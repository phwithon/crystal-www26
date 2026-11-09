"""RoBERTa(12층 + 분류층) 진위 분류 파인튜닝. gossip + pol 합쳐서 학습, test 는 합친 것과 데이터셋별로 보고."""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

SEED, MODEL, MAXLEN, BS, EPOCHS, LR = 42, "roberta-base", 512, 16, 3, 2e-5
D = r"C:\Users\USER\Desktop\졸프_데이터\data\DATA\FAKENEWSNET_문장추가"
OUT = os.path.dirname(os.path.abspath(__file__))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = torch.device("cuda")

# ---- 데이터: 두 데이터셋 합치고 (데이터셋, 라벨)별로 8:1:1 층화 분할
frames = []
for name in ("gossip", "pol"):
    df = pd.read_csv(os.path.join(D, f"{name}.csv"), usecols=["news_id", "title", "description", "label"], dtype={"news_id": str})
    df["dataset"] = name
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
df["title"] = df["title"].fillna("").astype(str).str.strip()
df["description"] = df["description"].fillna("").astype(str).str.strip()
df.loc[df["description"] == df["title"], "description"] = ""
split = np.empty(len(df), dtype=object)
rng = np.random.default_rng(SEED)
for _, idx in df.groupby(["dataset", "label"]).indices.items():
    idx = rng.permutation(idx)
    a, b = int(len(idx) * 0.8), int(len(idx) * 0.9)
    split[idx[:a]], split[idx[a:b]], split[idx[b:]] = "train", "val", "test"
df["split"] = split
df[["news_id", "dataset", "label", "split"]].to_csv(os.path.join(OUT, "split.csv"), index=False)
print(df.groupby(["split", "dataset"])["label"].agg(["count", "mean"]).round(3), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)


def batches(part, shuffle):
    d = df[df["split"] == part].reset_index(drop=True)
    items = list(zip(d["title"], d["description"], d["label"].astype(int), d["dataset"]))

    def collate(b):
        enc = tok([x[0] for x in b], [x[1] for x in b], truncation="only_second", max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([x[2] for x in b])
        return enc, [x[3] for x in b]

    return DataLoader(items, batch_size=BS if shuffle else 64, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps, ds = [], [], []
    for enc, names in loader:
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**{k: v for k, v in enc.items() if k != "labels"}).logits
        ps += logits.argmax(-1).tolist(); ys += enc["labels"].tolist(); ds += names
    ys, ps, ds = np.array(ys), np.array(ps), np.array(ds)

    def m(mask):
        y, p = ys[mask], ps[mask]
        tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1_fake = 2 * prec * rec / max(prec + rec, 1e-9)
        tn = ((p == 0) & (y == 0)).sum()
        prec0 = tn / max(tn + fn, 1); rec0 = tn / max(tn + fp, 1)
        f1_real = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-9)
        return {"n": int(mask.sum()), "acc": float((y == p).mean()), "f1_fake": float(f1_fake), "macro_f1": float((f1_fake + f1_real) / 2)}

    model.train()
    return {"all": m(np.ones(len(ys), bool)), "gossip": m(ds == "gossip"), "pol": m(ds == "pol")}


model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
train_dl, val_dl, test_dl = batches("train", True), batches("val", False), batches("test", False)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = len(train_dl) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

best, best_state = -1, None
for ep in range(1, EPOCHS + 1):
    t0, tot = time.time(), 0.0
    for i, (enc, _) in enumerate(train_dl, 1):
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**enc).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        tot += loss.item()
        if i % 200 == 0:
            print(f"  epoch {ep} step {i}/{len(train_dl)} loss {tot / i:.4f} ({time.time() - t0:.0f}s)", flush=True)
    v = evaluate(model, val_dl)
    print(f"epoch {ep} loss {tot / len(train_dl):.4f} ({time.time() - t0:.0f}s) | val acc {v['all']['acc']:.4f} macroF1 {v['all']['macro_f1']:.4f}", flush=True)
    if v["all"]["macro_f1"] > best:
        best = v["all"]["macro_f1"]
        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

model.load_state_dict(best_state)
model.save_pretrained(os.path.join(OUT, "best")); tok.save_pretrained(os.path.join(OUT, "best"))
res = evaluate(model, test_dl)
json.dump(res, open(os.path.join(OUT, "test_results.json"), "w"), indent=2)
print("TEST", json.dumps(res), flush=True)
