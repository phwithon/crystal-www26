"""혼재 사전 실험: 정답 label로 SBERT+HDBSCAN 군집을 만들어 진짜·가짜가 함께 든 군집을 센다."""
import argparse
import json
import os
import re
import sys

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE, MIN_SAMPLES = 5, 1
BODY_CHARS = 1500
C_REAL, C_FAKE, C_NOISE, C_INK, C_MUTED, C_GRID = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#898781", "#e1e0d9"


def load_articles(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_text(a):
    t = (a.get("title") or "") + " " + (a.get("body") or "")[:BODY_CHARS]
    return re.sub(r"\s+", " ", t).strip()


def embed(texts, batch_size=64):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=dev)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)


def cluster(emb, mcs=MIN_CLUSTER_SIZE, ms=MIN_SAMPLES, method="eom"):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                           cluster_selection_method=method, core_dist_n_jobs=-1).fit_predict(emb)


def nearest_opposite(emb_q, emb_t, same_pool):
    """가짜 기사마다 가장 가까운 진짜 기사 코사인(군집 무관). sim>=0.99는 중복으로 따로 센다."""
    q = emb_q / np.linalg.norm(emb_q, axis=1, keepdims=True)
    t = emb_t / np.linalg.norm(emb_t, axis=1, keepdims=True)
    S = q @ t.T
    dup = (S >= 0.99).any(axis=1)
    S[S >= 0.99] = -1
    best = S.max(axis=1)
    return {"n_fake": len(best), "n_fake_with_dup": int(dup.sum()),
            **{f"cos>={th}": int((best >= th).sum()) for th in (0.5, 0.6, 0.7, 0.8, 0.9)},
            "mean_best": float(best.mean()), "median_best": float(np.median(best))}


def analyze(emb, labels, clusters, rng):
    """한 범위(subset)의 통계. labels: 0 real / 1 fake."""
    n = len(labels)
    cids = sorted(set(clusters) - {-1})
    size, mixed = {}, []
    for c in cids:
        idx = np.where(clusters == c)[0]
        size[c] = len(idx)
        nf = int(labels[idx].sum())
        if 0 < nf < len(idx):
            mixed.append((c, len(idx), len(idx) - nf, nf))
    mixed_ids = {m[0] for m in mixed}
    in_cluster = int((clusters != -1).sum())
    in_mixed = int(sum(m[1] for m in mixed))
    # 반대 진위 짝(같은 군집, sim<1.0) 가능 기사 수와 최고 유사도
    en = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    fake_with, real_with, best_sims = 0, 0, []
    for c in mixed_ids:
        idx = np.where(clusters == c)[0]
        S = en[idx] @ en[idx].T
        lab = labels[idx]
        for i, gi in enumerate(idx):
            opp = [j for j in range(len(idx)) if lab[j] != lab[i] and S[i, j] < 1.0]
            if not opp:
                continue
            best_sims.append(float(S[i, opp].max()))
            if lab[i] == 1:
                fake_with += 1
            else:
                real_with += 1
    rand = [float(en[i] @ en[j]) for i, j in rng.integers(0, n, size=(2000, 2)) if i != j]
    sizes = sorted(m[1] for m in mixed)
    return {
        "n_articles": n,
        "n_real": int((labels == 0).sum()), "n_fake": int((labels == 1).sum()),
        "n_clusters": len(cids),
        "n_in_cluster": in_cluster, "in_cluster_ratio": in_cluster / n if n else 0.0,
        "n_noise": n - in_cluster,
        "n_mixed_clusters": len(mixed), "mixed_cluster_ratio": len(mixed) / len(cids) if cids else 0.0,
        "n_in_mixed": in_mixed, "in_mixed_ratio_all": in_mixed / n if n else 0.0,
        "in_mixed_ratio_clustered": in_mixed / in_cluster if in_cluster else 0.0,
        "mixed_sizes": sizes,
        "mixed_size_hist": {"5-9": sum(5 <= s <= 9 for s in sizes), "10-19": sum(10 <= s <= 19 for s in sizes),
                            "20-49": sum(20 <= s <= 49 for s in sizes), "50+": sum(s >= 50 for s in sizes)},
        "fake_with_counterpart": fake_with, "real_with_counterpart": real_with,
        "fake_with_counterpart_ratio": fake_with / int((labels == 1).sum()) if (labels == 1).sum() else 0.0,
        "mean_best_opposite_sim": float(np.mean(best_sims)) if best_sims else None,
        "mean_random_sim": float(np.mean(rand)),
        "mixed": mixed, "mixed_ids": mixed_ids, "cluster_size": size,
    }


def mixed_examples(arts, sub_idx, clusters, stats, k=5):
    out = []
    for c, sz, nr, nf in sorted(stats["mixed"], key=lambda m: -min(m[2], m[3]))[:k]:
        idx = sub_idx[np.where(clusters == c)[0]]
        reals = [arts[i] for i in idx if arts[i]["label"] == 0][:2]
        fakes = [arts[i] for i in idx if arts[i]["label"] == 1][:1]
        out.append({"cluster": int(c), "size": sz, "real": nr, "fake": nf,
                    "titles": [(a["label"], a["dataset"], a["title"]) for a in fakes + reals]})
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.title.set_color(C_INK)
    ax.xaxis.label.set_color("#52514e")
    ax.yaxis.label.set_color("#52514e")


def plot_umap(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import umap
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6.5), facecolor="#fcfcfb")
    axes = np.atleast_1d(axes)
    for ax, (name, emb, labels, clusters, mixed_ids) in zip(axes, panels):
        xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(emb)
        noise = clusters == -1
        inmix = np.isin(clusters, list(mixed_ids))
        ax.set_facecolor("#fcfcfb")
        ax.scatter(*xy[noise].T, s=3, c=C_NOISE, alpha=0.25, lw=0, rasterized=True)
        for lab, col in ((0, C_REAL), (1, C_FAKE)):
            m = ~noise & ~inmix & (labels == lab)
            ax.scatter(*xy[m].T, s=5, c=col, alpha=0.45, lw=0, rasterized=True)
        for lab, col in ((0, C_REAL), (1, C_FAKE)):
            m = inmix & (labels == lab)
            ax.scatter(*xy[m].T, s=22, c=col, edgecolors=C_INK, linewidths=0.6, alpha=0.95, rasterized=True)
        ax.set_title(f"{name}  (n={len(labels):,}, mixed clusters={len(mixed_ids)}, articles in mixed={int(inmix.sum()):,})",
                     fontsize=11, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        style(ax)
    handles = [Line2D([], [], marker="o", ls="", ms=6, color=C_REAL, label="real (label 0)"),
               Line2D([], [], marker="o", ls="", ms=6, color=C_FAKE, label="fake (label 1)"),
               Line2D([], [], marker="o", ls="", ms=6, color=C_NOISE, alpha=0.5, label="noise (-1)"),
               Line2D([], [], marker="o", ls="", ms=8, color="w", markeredgecolor=C_INK, label="in mixed cluster (ring)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10, labelcolor="#52514e")
    fig.suptitle("SBERT all-MiniLM-L6-v2 → UMAP 2D, HDBSCAN(5/1); ring = cluster holding both real and fake",
                 fontsize=12, color=C_INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_sizes(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5), facecolor="#fcfcfb")
    axes = np.atleast_1d(axes)
    bins = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 100, 10 ** 6]
    labels_x = ["5", "6", "7", "8", "9", "10-11", "12-14", "15-19", "20-29", "30-49", "50-99", "100+"]
    for ax, (name, st) in zip(axes, panels):
        sizes = np.array(list(st["cluster_size"].values()))
        ids = np.array(list(st["cluster_size"].keys()))
        ismix = np.isin(ids, list(st["mixed_ids"]))
        h_single, _ = np.histogram(sizes[~ismix], bins=bins)
        h_mixed, _ = np.histogram(sizes[ismix], bins=bins)
        x = np.arange(len(labels_x))
        ax.set_facecolor("#fcfcfb")
        ax.bar(x, h_single, width=0.78, color=C_NOISE, alpha=0.55, lw=0, label="single-label cluster")
        ax.bar(x, h_mixed, width=0.78, bottom=h_single, color=C_FAKE, lw=0, label="mixed cluster", edgecolor="#fcfcfb")
        for xi, (a, b) in enumerate(zip(h_single, h_mixed)):
            if b:
                ax.text(xi, a + b + max(h_single.max(), 1) * 0.01, str(b), ha="center", va="bottom", fontsize=8, color="#52514e")
        ax.set_xticks(x); ax.set_xticklabels(labels_x, rotation=0, fontsize=8)
        ax.set_xlabel("cluster size (articles)")
        ax.set_ylabel("clusters")
        ax.set_title(f"{name}  (clusters={st['n_clusters']}, mixed={st['n_mixed_clusters']})", fontsize=11, loc="left")
        ax.grid(axis="y", color=C_GRID, lw=0.6); ax.set_axisbelow(True)
        style(ax)
    fig.legend(handles=[Patch(color=C_NOISE, alpha=0.55, label="single-label cluster"), Patch(color=C_FAKE, label="mixed cluster (number = count)")],
               loc="lower center", ncol=2, frameon=False, fontsize=10, labelcolor="#52514e")
    fig.suptitle("HDBSCAN(5/1) cluster size distribution", fontsize=12, color=C_INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default=os.path.join(ROOT, "crystal", "data", "articles.jsonl"))
    ap.add_argument("--emb", default=os.path.join(ROOT, "crystal", "data", "event", "embeddings.npy"))
    ap.add_argument("--out", default=os.path.join(ROOT, "_workspace"))
    ap.add_argument("--force-embed", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.emb), exist_ok=True)

    arts = load_articles(args.articles)
    ids = [a["article_id"] for a in arts]
    labels_all = np.array([int(a["label"]) for a in arts])
    ids_path = os.path.splitext(args.emb)[0] + "_ids.txt"
    if os.path.exists(args.emb) and os.path.exists(ids_path) and not args.force_embed \
            and open(ids_path, encoding="utf-8").read().split("\n")[:len(ids)] == ids:
        emb_all = np.load(args.emb)
    else:
        emb_all = embed([build_text(a) for a in arts])
        np.save(args.emb, emb_all)
        with open(ids_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids))
    print("embeddings", emb_all.shape, file=sys.stderr)

    rng = np.random.default_rng(0)
    scopes = {}
    for split_name, split_fn in (("test", lambda a: a["split"] == "test"), ("all", lambda a: True)):
        for ds in ("gossip+pol", "gossip", "pol"):
            idx = np.array([i for i, a in enumerate(arts) if split_fn(a) and (ds == "gossip+pol" or a["dataset"] == ds)])
            cl = cluster(emb_all[idx])
            st = analyze(emb_all[idx], labels_all[idx], cl, rng)
            st["examples"] = mixed_examples(arts, idx, cl, st)
            scopes[(split_name, ds)] = (idx, cl, st)
            print(split_name, ds, {k: v for k, v in st.items() if k in ("n_articles", "n_clusters", "n_mixed_clusters", "n_in_mixed", "fake_with_counterpart")}, file=sys.stderr)

    (ti, tc, ts), (ai, ac, as_) = scopes[("test", "gossip+pol")], scopes[("all", "gossip+pol")]
    variants = {}
    for sname, idx in (("test", ti), ("all", ai)):
        for mcs, ms, method in ((5, 1, "eom"), (5, 5, "eom"), (5, 1, "leaf"), (5, 5, "leaf"), (10, 5, "eom")):
            cl = cluster(emb_all[idx], mcs, ms, method)
            st = analyze(emb_all[idx], labels_all[idx], cl, rng)
            variants[f"{sname}/mcs{mcs}_ms{ms}_{method}"] = {
                k: st[k] for k in ("n_clusters", "in_cluster_ratio", "n_mixed_clusters", "n_in_mixed",
                                   "fake_with_counterpart", "fake_with_counterpart_ratio", "mean_best_opposite_sim")}
            variants[f"{sname}/mcs{mcs}_ms{ms}_{method}"]["max_cluster_size"] = max(st["cluster_size"].values()) if st["cluster_size"] else 0
            print(sname, mcs, ms, method, variants[f"{sname}/mcs{mcs}_ms{ms}_{method}"], file=sys.stderr)
    tf = ti[labels_all[ti] == 1]
    nearest = {"test_fake->test_real": nearest_opposite(emb_all[tf], emb_all[ti[labels_all[ti] == 0]], True),
               "test_fake->all_real": nearest_opposite(emb_all[tf], emb_all[labels_all == 0], False),
               "all_fake->all_real": nearest_opposite(emb_all[labels_all == 1], emb_all[labels_all == 0], False)}
    print(nearest, file=sys.stderr)
    plot_umap([("test pool", emb_all[ti], labels_all[ti], tc, ts["mixed_ids"]),
               ("all 18,126", emb_all[ai], labels_all[ai], ac, as_["mixed_ids"])], os.path.join(args.out, "01_event_umap.png"))
    plot_sizes([("test pool", ts), ("all 18,126", as_)], os.path.join(args.out, "01_event_cluster_sizes.png"))

    dump = {f"{s}/{d}": {k: v for k, v in st.items() if k not in ("mixed", "mixed_ids", "cluster_size")}
            for (s, d), (_, _, st) in scopes.items()}
    dump["variants"], dump["nearest_opposite"] = variants, nearest
    with open(os.path.join(args.out, "01_event_feasibility_stats.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
