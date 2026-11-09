"""인물명 마스킹 RoBERTa (2026-11-03). mask_names.py가 만든 masked.csv(제목·본문의 PERSON → [NAME])로 fold0 분할을 다시 학습한다.
용도: 기여도 계산(LIME/ContextCite)용 모델. 원래 모델은 특정 인물명을 가짜 신호로 크게 쓰기 때문에, 인물명을 가린 모델로 문장·단어 기여도를 잰다.
판정(final_label)은 원래 10-fold 모델(train_kfold.py)을 그대로 쓴다. 분할·설정은 train_kfold.py의 fold0과 동일
(kfold.csv의 fold 열: test=0, val=1, train=나머지; roberta-base, 3 epoch, lr 2e-5, batch 16, bf16, best=val macro F1).
참고 결과(2026-11-03): test acc 0.8489 (원래 fold0 0.8577).
실행: $env:KMP_DUPLICATE_LIB_OK='TRUE'; python -X utf8 roberta_veracity/train_masked.py [--masked crystal/data/keyword/masked.csv] [--fold 0] [--out roberta_veracity/folds/fold0_masked]"""
import argparse
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
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def macro_f1(ys, ps):
    ys, ps = np.asarray(ys), np.asarray(ps)
    f = []
    for c in (0, 1):
        tp = ((ps == c) & (ys == c)).sum(); fp = ((ps == c) & (ys != c)).sum(); fn = ((ps != c) & (ys == c)).sum()
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked", default=os.path.join(ROOT, "crystal", "data", "keyword", "masked.csv"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "folds", f"fold{a.fold}_masked")
    os.makedirs(os.path.join(out, "best"), exist_ok=True)
    seed_all()
    kf = pd.read_csv(os.path.join(HERE, "kfold.csv"), dtype={"news_id": str})
    m = pd.read_csv(a.masked, dtype={"news_id": str})
    df = kf.merge(m[["news_id", "title", "description"]], on="news_id", how="inner")
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df.loc[df["description"] == df["title"], "description"] = ""
    df["label"] = df["label"].astype(int)
    test, val = df[df["fold"] == a.fold], df[df["fold"] == (a.fold + 1) % K]
    train = df[~df["fold"].isin([a.fold, (a.fold + 1) % K])]
    print(f"train {len(train)} val {len(val)} test {len(test)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)

    def loader(d, shuffle):
        items = list(zip(d["title"], d["description"], d["label"]))

        def collate(b):
            enc = tok([x[0] for x in b], [x[1] for x in b], truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt")
            enc["labels"] = torch.tensor([x[2] for x in b])
            return enc
        return DataLoader(items, batch_size=BS, shuffle=shuffle, collate_fn=collate)

    def predict(model, d):
        ps = []
        model.eval()
        with torch.no_grad():
            for b in loader(d, False):
                b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    ps += model(**b).logits.float().argmax(-1).tolist()
        return ps

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    tl = loader(train, True)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * len(tl) * EPOCHS), len(tl) * EPOCHS)
    best, t0 = -1, time.time()
    for ep in range(EPOCHS):
        model.train()
        for b in tl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = model(**b).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
        f1 = macro_f1(val["label"].tolist(), predict(model, val))
        print(f"epoch {ep + 1} val macro F1 {f1:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best = f1; model.save_pretrained(os.path.join(out, "best")); tok.save_pretrained(os.path.join(out, "best"))
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(out, "best")).to(dev)
    ps = predict(model, test)
    res = {"fold": a.fold, "val_macro_f1": best, "test_acc": float((np.asarray(ps) == test["label"].to_numpy()).mean()),
           "test_macro_f1": macro_f1(test["label"].tolist(), ps), "n_test": len(test)}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(res, flush=True)


if __name__ == "__main__":
    main()
