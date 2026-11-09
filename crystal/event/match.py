"""본 매칭: is_displayable 기사마다 같은 HDBSCAN 군집(--method cluster)이면서 코사인 ≥ threshold인, final_label이 반대인 displayable 기사를 유사도 상위 TOP_K개 counterparts에 둔다 (--pool all).
2026-10-30: 전 진위 5단계 → 반대 진위 2건 (회의 결정)."""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from crystal.common import is_boilerplate, is_displayable  # noqa: E402,F401

DATA = os.path.join(ROOT, "crystal", "data")
ARTICLES = os.path.join(DATA, "articles.jsonl")
EMB = os.path.join(DATA, "event", "embeddings.npy")
FEEDS = os.path.join(DATA, "feeds.jsonl")
SEED_USERS = ("866694567411798017", "128578213", "145250021")
TOP_K = 2
BOILERPLATE_IDS = os.path.join(DATA, "event", "boilerplate_ids.txt")


def norm_title(t):
    return re.sub(r"[^0-9a-z가-힣]+", "", (t or "").lower())


def read_articles(path=ARTICLES):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_embeddings(ids):
    emb = np.load(EMB)
    emb_ids = open(os.path.splitext(EMB)[0] + "_ids.txt", encoding="utf-8").read().split("\n")
    pos = {a: i for i, a in enumerate(emb_ids)}
    emb = emb[[pos[i] for i in ids]]
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def match(arts, pool, method, threshold):
    """반환 {article_id: (event_id, counterparts)} (test 기사만)."""
    ids = [a["article_id"] for a in arts]
    emb = load_embeddings(ids)
    fl = np.array([int(a["final_label"]) for a in arts])
    pf = np.array([float(a["pred_prob_fake"]) for a in arts])
    bp = np.array([is_boilerplate(a["body"]) for a in arts])
    with open(BOILERPLATE_IDS, "w", encoding="utf-8") as f:
        f.write(chr(10).join(ids[i] for i in np.where(bp)[0]))
    disp = np.array([is_displayable(a["title"], a["body"]) for a in arts])
    q_idx = np.array([i for i, a in enumerate(arts) if disp[i] and (pool == "all" or a["split"] == "test")])
    p_idx = q_idx if pool == "test" else np.where(disp)[0]
    p_idx = p_idx[~bp[p_idx]]
    event = np.full(len(arts), -1, dtype=int)
    if method == "cluster":
        import hdbscan
        event[p_idx] = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric="euclidean",
                                       core_dist_n_jobs=-1).fit_predict(emb[p_idx])
    out = {}
    S = emb[q_idx] @ emb[p_idx].T
    for r, qi in enumerate(q_idx):
        s = S[r]
        ok = (fl[p_idx] != fl[qi]) & (np.round(s, 4) < 1.0) & (p_idx != qi) & (not bp[qi])  # 반대 진위, 반올림 뒤에도 sim < 1 (같은 글 복제본 제외)
        ok &= (event[p_idx] == event[qi]) & (event[qi] != -1) & (s >= threshold) if method == "cluster" else (s >= threshold)
        cand, seen, keep = np.where(ok)[0], {norm_title(arts[qi]["title"])}, []
        for c in cand[np.argsort(-s[cand])]:  # 같은 제목(정규화)의 중복 기사는 sim 높은 하나만
            nt = norm_title(arts[p_idx[c]]["title"])
            if nt not in seen:
                seen.add(nt); keep.append(c)
            if len(keep) == TOP_K:
                break
        cand = keep
        out[ids[qi]] = (int(event[qi]), [{"article_id": ids[p_idx[c]], "final_label": int(fl[p_idx[c]]),
                                          "pred_prob_fake": round(float(pf[p_idx[c]]), 4), "sim": round(float(s[c]), 4)} for c in cand])
    return out


def stats(arts, res):
    by = {a["article_id"]: a for a in arts}
    test = [a for a in arts if a["article_id"] in res]
    fake = [a for a in test if a["final_label"] == 1]
    real = [a for a in test if a["final_label"] == 0]
    has = lambda a: len(res[a["article_id"]][1]) > 0
    sims = [c["sim"] for a in test for c in res[a["article_id"]][1]]
    n_cp = [len(res[a["article_id"]][1]) for a in test]
    ex = sorted((a for a in fake if has(a)), key=lambda a: -res[a["article_id"]][1][0]["sim"])
    examples = []
    for a in ex[:5]:
        c = res[a["article_id"]][1][0]
        examples.append({"fake": (a["article_id"], a["title"][:90]), "pair": (c["article_id"], by[c["article_id"]]["title"][:90],
                                                                             by[c["article_id"]]["split"]), "sim": c["sim"]})
    feeds = {}
    with open(FEEDS, encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            if str(d["user_id"]) in SEED_USERS:
                top = d["feed"][:12]
                feeds[str(d["user_id"])] = {"n": len(top), "in_test": sum(i in res for i in top),
                                            "with_pair": sum(i in res and len(res[i][1]) > 0 for i in top),
                                            "fake_with_pair": sum(i in res and by[i]["final_label"] == 1 and len(res[i][1]) > 0 for i in top)}
    return {"n_test": len(test), "n_test_fake": len(fake), "n_test_real": len(real),
            "fake_with_pair": sum(map(has, fake)), "fake_with_pair_ratio": round(sum(map(has, fake)) / len(fake), 4),
            "real_with_pair": sum(map(has, real)), "real_with_pair_ratio": round(sum(map(has, real)) / len(real), 4),
            "n_event_clusters": len({res[a["article_id"]][0] for a in test} - {-1}),
            "pairs_total": len(sims), "n_counterparts_hist": {k: n_cp.count(k) for k in range(TOP_K + 1)},
            "sim_quantiles": {q: round(float(np.quantile(sims, q)), 4) for q in (0.0, 0.25, 0.5, 0.75, 1.0)} if sims else None,
            "sim_mean": round(float(np.mean(sims)), 4) if sims else None,
            "examples": examples, "seed_feeds_top12": feeds}


def apply(res, path=ARTICLES, retries=5):
    """읽기 직전 상태 위에 두 필드만 붙여 같은 순서로 덮어쓴다. 사이에 파일이 바뀌면 다시 읽는다."""
    for _ in range(retries):
        st0 = os.stat(path)
        rows = read_articles(path)
        n0, k0 = len(rows), sum("keywords" in r for r in rows)
        for r in rows:
            if r["article_id"] in res:
                r["event_id"], r["counterparts"] = res[r["article_id"]]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st1 = os.stat(path)
        if (st1.st_mtime_ns, st1.st_size) != (st0.st_mtime_ns, st0.st_size):
            os.remove(tmp); time.sleep(1); continue
        os.replace(tmp, path)
        rows2 = read_articles(path)
        n1, k1 = len(rows2), sum("keywords" in r for r in rows2)
        if n1 == n0 and k1 >= k0 and sum("counterparts" in r for r in rows2) >= len(res):
            print(f"applied: lines {n0}->{n1}, keywords {k0}->{k1}, counterparts {len(res)}", file=sys.stderr)
            return True
        time.sleep(1)
    raise RuntimeError("apply failed: articles.jsonl changed during write")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["test", "all"], default="test")
    ap.add_argument("--method", choices=["cluster", "cos"], default="cos")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--out", default=None, help="결과 jsonl (article_id, event_id, counterparts)")
    ap.add_argument("--apply", nargs="?", const="", default=None, metavar="FILE",
                    help="articles.jsonl에 붙인다. FILE을 주면 그 jsonl을, 없으면 지금 계산한 결과를")
    ap.add_argument("--stats", default=None, help="확인 수치 json 저장 경로")
    args = ap.parse_args()

    arts = read_articles()
    if args.apply:
        with open(args.apply, encoding="utf-8") as f:
            res = {d["article_id"]: (d["event_id"], d["counterparts"]) for d in map(json.loads, f)}
    else:
        res = match(arts, args.pool, args.method, args.threshold)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for aid, (eid, cps) in res.items():
                f.write(json.dumps({"article_id": aid, "event_id": eid, "counterparts": cps}, ensure_ascii=False) + "\n")
    st = stats(arts, res)
    print(json.dumps({k: v for k, v in st.items() if k != "examples"}, ensure_ascii=False), file=sys.stderr)
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)
    if args.apply is not None:
        apply(res)


if __name__ == "__main__":
    main()
