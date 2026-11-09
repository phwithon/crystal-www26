"""10-fold 교차 예측 (2026-10-21). gossip+pol 합본, 중복 news_id 첫 줄만(18,124). (dataset, label) 층화로 묶음 10개(seed 42),
fold i: test = 묶음 i, val = 묶음 (i+1)%10, train = 나머지 8묶음. 학습 설정은 train.py와 동일(roberta-base, 3 epoch, lr 2e-5, batch 16, bf16, best = val macro F1).
산출: folds/fold{i}/best, temperature.json(그 fold의 val로 calibrate.fit_temperature), predictions.csv, result.json, done(완료 표시) → kfold.csv, kfold_predictions.csv, kfold_results.json.
실행: $env:KMP_DUPLICATE_LIB_OK='TRUE'; python -X utf8 roberta_veracity/train_kfold.py"""
import gc
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

SEED, MODEL, MAXLEN, BS, EPOCHS, LR, K = 42, "roberta-base", 512, 16, 3, 2e-5, 10
D = r"C:\Users\USER\Desktop\졸프_데이터\data\DATA\FAKENEWSNET_문장추가"
OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)
sys.path.insert(0, ROOT)
from crystal.veracity.calibrate import fit_temperature  # noqa: E402

dev = torch.device("cuda")


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


frames = []
for name in ("gossip", "pol"):
    df = pd.read_csv(os.path.join(D, f"{name}.csv"), usecols=["news_id", "title", "description", "label"], dtype={"news_id": str})
    df["dataset"] = name
    frames.append(df)
df = pd.concat(frames, ignore_index=True).drop_duplicates("news_id", keep="first").reset_index(drop=True)
df["label"] = df["label"].astype(int)
df["title"] = df["title"].fillna("").astype(str).str.strip()
df["description"] = df["description"].fillna("").astype(str).str.strip()
df.loc[df["description"] == df["title"], "description"] = ""
fold = np.empty(len(df), dtype=int)
rng = np.random.default_rng(SEED)
for _, idx in df.groupby(["dataset", "label"]).indices.items():
    idx = rng.permutation(idx)
    fold[idx] = np.arange(len(idx)) % K
df["fold"] = fold
df[["news_id", "dataset", "label", "fold"]].to_csv(os.path.join(OUT, "kfold.csv"), index=False)
print(f"articles {len(df)}  fold sizes {np.bincount(fold).tolist()}", flush=True)
print(df.groupby(["fold", "dataset"])["label"].agg(["count", "mean"]).round(3).to_string(), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)


def batches(d, shuffle):
    d = d.reset_index(drop=True)
    items = list(zip(d["title"], d["description"], d["label"].astype(int), d["dataset"], d["news_id"]))

    def collate(b):
        enc = tok([x[0] for x in b], [x[1] for x in b], truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([x[2] for x in b])
        return enc, [x[3] for x in b], [x[4] for x in b]

    return DataLoader(items, batch_size=BS if shuffle else 64, shuffle=shuffle, collate_fn=collate)


def metrics(ys, ps, ds):
    ys, ps, ds = np.array(ys), np.array(ps), np.array(ds)

    def m(mask):
        y, p = ys[mask], ps[mask]
        tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1_fake = 2 * prec * rec / max(prec + rec, 1e-9)
        prec0 = tn / max(tn + fn, 1); rec0 = tn / max(tn + fp, 1)
        f1_real = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-9)
        return {"n": int(mask.sum()), "acc": float((y == p).mean()) if mask.any() else None, "f1_fake": float(f1_fake),
                "macro_f1": float((f1_fake + f1_real) / 2), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    return {"all": m(np.ones(len(ys), bool)), "gossip": m(ds == "gossip"), "pol": m(ds == "pol")}


@torch.no_grad()
def run_logits(model, loader):
    model.eval()
    out = []  # (news_id, dataset, label, logit_real, logit_fake)
    for enc, names, ids in loader:
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**{k: v for k, v in enc.items() if k != "labels"}).logits.float().cpu()
        out += [(i, n, int(y), float(l[0]), float(l[1])) for i, n, y, l in zip(ids, names, enc["labels"].tolist(), logits)]
    model.train()
    return out


def train_fold(i):
    fdir = os.path.join(OUT, "folds", f"fold{i}")
    if os.path.exists(os.path.join(fdir, "done")):
        print(f"fold {i}: done, skip", flush=True)
        return json.load(open(os.path.join(fdir, "result.json"), encoding="utf-8"))
    os.makedirs(fdir, exist_ok=True)
    seed_all()
    t_fold = time.time()
    vi = (i + 1) % K
    tr, va, te = df[(df["fold"] != i) & (df["fold"] != vi)], df[df["fold"] == vi], df[df["fold"] == i]
    print(f"fold {i}: train {len(tr)} val {len(va)} (bucket {vi}) test {len(te)}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    train_dl, val_dl, test_dl = batches(tr, True), batches(va, False), batches(te, False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(train_dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best, best_state, best_ep, epochs = -1, None, 0, []
    for ep in range(1, EPOCHS + 1):
        t0, tot = time.time(), 0.0
        for n, (enc, _, _) in enumerate(train_dl, 1):
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
            if n % 300 == 0:
                print(f"  fold {i} epoch {ep} step {n}/{len(train_dl)} loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
        vl = run_logits(model, val_dl)
        v = metrics([r[2] for r in vl], [int(r[4] > r[3]) for r in vl], [r[1] for r in vl])
        epochs.append({"epoch": ep, "loss": round(tot / len(train_dl), 4), "val_acc": round(v["all"]["acc"], 4),
                       "val_macro_f1": round(v["all"]["macro_f1"], 4), "seconds": round(time.time() - t0)})
        print(f"fold {i} epoch {ep} loss {tot / len(train_dl):.4f} ({time.time() - t0:.0f}s) | val acc {v['all']['acc']:.4f} macroF1 {v['all']['macro_f1']:.4f}", flush=True)
        if v["all"]["macro_f1"] > best:
            best, best_ep = v["all"]["macro_f1"], ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.save_pretrained(os.path.join(fdir, "best")); tok.save_pretrained(os.path.join(fdir, "best"))
    vl = run_logits(model, val_dl)
    temp = fit_temperature(torch.tensor([[r[3], r[4]] for r in vl]), torch.tensor([r[2] for r in vl]))
    json.dump(temp, open(os.path.join(fdir, "temperature.json"), "w", encoding="utf-8"), indent=1)
    T = temp["T"]
    tl = run_logits(model, test_dl)
    z = torch.tensor([[r[3], r[4]] for r in tl])
    prob = torch.softmax(z / T, -1)[:, 1].tolist()
    pred = (z[:, 1] > z[:, 0]).long().tolist()
    pd.DataFrame({"news_id": [r[0] for r in tl], "fold": i, "logit_real": [round(r[3], 6) for r in tl], "logit_fake": [round(r[4], 6) for r in tl],
                  "prob_fake": [round(p, 6) for p in prob], "pred": pred}).to_csv(os.path.join(fdir, "predictions.csv"), index=False)
    res = {"fold": i, "val_bucket": vi, "n_train": len(tr), "n_val": len(va), "n_test": len(te), "best_epoch": best_ep, "epochs": epochs,
           "val": metrics([r[2] for r in vl], [int(r[4] > r[3]) for r in vl], [r[1] for r in vl]),
           "test": metrics([r[2] for r in tl], pred, [r[1] for r in tl]), "temperature": temp, "seconds": round(time.time() - t_fold)}
    json.dump(res, open(os.path.join(fdir, "result.json"), "w", encoding="utf-8"), indent=1)
    print(f"fold {i} TEST acc {res['test']['all']['acc']:.4f} macroF1 {res['test']['all']['macro_f1']:.4f} T {T} ({res['seconds']}s)", flush=True)
    del model, opt, sched, best_state, train_dl, val_dl, test_dl
    gc.collect(); torch.cuda.empty_cache()
    open(os.path.join(fdir, "done"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    return res


t_all = time.time()
results = [train_fold(i) for i in range(K)]
preds = pd.concat([pd.read_csv(os.path.join(OUT, "folds", f"fold{i}", "predictions.csv"), dtype={"news_id": str}) for i in range(K)], ignore_index=True)
preds.to_csv(os.path.join(OUT, "kfold_predictions.csv"), index=False)
m = preds.merge(df[["news_id", "dataset", "label"]], on="news_id")
assert len(m) == len(df) == len(preds), (len(m), len(df), len(preds))
pooled = metrics(m["label"].tolist(), m["pred"].tolist(), m["dataset"].tolist())
summary = {"n": len(m), "pooled": pooled,
           "folds": [{"fold": r["fold"], "n_test": r["n_test"], "best_epoch": r["best_epoch"], "acc": round(r["test"]["all"]["acc"], 4),
                      "macro_f1": round(r["test"]["all"]["macro_f1"], 4), "f1_fake": round(r["test"]["all"]["f1_fake"], 4),
                      "val_macro_f1": round(r["val"]["all"]["macro_f1"], 4), "T": r["temperature"]["T"],
                      "val_ece_before": r["temperature"]["val_ece_before"], "val_ece_after": r["temperature"]["val_ece_after"], "seconds": r["seconds"]} for r in results],
           "mean_fold_acc": round(float(np.mean([r["test"]["all"]["acc"] for r in results])), 4),
           "std_fold_acc": round(float(np.std([r["test"]["all"]["acc"] for r in results])), 4),
           "train_seconds_total": int(sum(r["seconds"] for r in results)), "wall_seconds_this_run": round(time.time() - t_all)}
json.dump(summary, open(os.path.join(OUT, "kfold_results.json"), "w", encoding="utf-8"), indent=1)
print("KFOLD", json.dumps({k: v for k, v in summary.items() if k != "folds"}), flush=True)
