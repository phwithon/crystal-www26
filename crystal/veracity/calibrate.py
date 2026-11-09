"""Temperature scaling (fit_temperature는 train_kfold.py가 fold마다 재사용): val 기사(split == "val")의 logit과 정답 label로 NLL을 최소화하는 스칼라 T를 맞춘다 (LBFGS, 초기값 1.5).
결과 crystal/data/veracity/temperature.json {"T", "val_nll_before", "val_nll_after", "val_ece_before", "val_ece_after", "n_val"}. ECE는 10-bin.
실행: $env:KMP_DUPLICATE_LIB_OK='TRUE'; python -X utf8 crystal/veracity/calibrate.py"""
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import encode_batch, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTICLES = os.path.join(ROOT, "crystal", "data", "articles.jsonl")
OUT = os.path.join(ROOT, "crystal", "data", "veracity", "temperature.json")
BS = 64


def ece(prob_fake, y, bins=10):
    """10-bin ECE. 예측 클래스 확신도(max prob) 기준."""
    conf = torch.where(prob_fake >= 0.5, prob_fake, 1 - prob_fake)
    pred = (prob_fake >= 0.5).long()
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    e = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return float(e)


@torch.no_grad()
def val_logits():
    tok, model = load()
    rows = [json.loads(l) for l in open(ARTICLES, encoding="utf-8")]
    val = [r for r in rows if r["split"] == "val"]
    val.sort(key=lambda r: len(r["title"]) + len(r["body"]))
    logits = []
    for s in range(0, len(val), BS):
        b = val[s:s + BS]
        ids, mask = encode_batch(tok, [r["title"] for r in b], [r["body"] for r in b])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits.append(model(input_ids=ids, attention_mask=mask).logits.float().cpu())
    return torch.cat(logits), torch.tensor([int(r["label"]) for r in val])


def fit_temperature(z, y):
    """logit z [N,2], 정답 y [N] → {"T", nll/ece 전후, "n_val"} (LBFGS, log T, 초기 1.5)."""
    logT = torch.zeros(1, requires_grad=True)
    with torch.no_grad():
        logT.fill_(float(torch.log(torch.tensor(1.5))))
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(logT.exp().detach())
    with torch.no_grad():
        nll0, nll1 = float(F.cross_entropy(z, y)), float(F.cross_entropy(z / T, y))
        p0, p1 = torch.softmax(z, -1)[:, 1], torch.softmax(z / T, -1)[:, 1]
    return {"T": round(T, 6), "val_nll_before": round(nll0, 6), "val_nll_after": round(nll1, 6),
            "val_ece_before": round(ece(p0, y), 6), "val_ece_after": round(ece(p1, y), 6), "n_val": int(len(y))}


def main():
    torch.manual_seed(42)
    z, y = val_logits()
    res = fit_temperature(z, y)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
