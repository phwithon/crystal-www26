"""ContextCite 방식 문장 기여도 (2026-11-03, 직접 구현). 라이브러리(생성형 LM용)를 쓰지 않고 그 방식을 분류기에 맞게 구현했다:
본문을 문장으로 나눈 뒤 문장을 무작위로 지운 입력 N=256개를 만들어 분류기의 (판정 라벨 log-odds)를 재고, 문장 포함 여부(0/1) → log-odds를
LassoCV 선형 대리모델로 맞춘다. 계수가 큰 문장 = 판정에 기여한 문장. 상위 5개를 원문 좌표로 저장한다.
기본 모델은 인물명 마스킹 fold0 모델(train_masked.py)이며, 마스킹 문장을 원문에서 되찾아 좌표를 매긴다([NAME] ↔ 임의 이름).
실행: $env:KMP_DUPLICATE_LIB_OK='TRUE'; python -X utf8 crystal/keyword/contextcite.py --model <모델 폴더> --ids <ids.json> [--masked masked.csv] [--out ...]
출력: crystal/data/keyword/cc_sentences_top5.jsonl  {article_id, final_label, sentence(1위), top[≤5]{start,end,phrase,w}, weights, r2}"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LassoCV
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from crystal.veracity.loader import normalize  # noqa: E402

ARTICLES = os.path.join(ROOT, "crystal", "data", "articles.jsonl")
SENT = re.compile(r"[^.!?]+[.!?]+[\"”’']?|[^.!?]+$")


def split(text, limit):
    """limit(토큰 512개 안에 들어온 문자 수)까지의 본문을 문장으로 나눈다. 16자 이하 조각은 뺀다."""
    out = []
    for m in SENT.finditer(text[:limit]):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s > 15:
            out.append((s, e))
    return out


def to_orig(msent, ob):
    """마스킹 문장을 원문에서 찾는다 ([NAME]은 1~60자 아무 문자열과 매칭)."""
    parts = [re.escape(p) for p in msent.split("[NAME]")]
    m = re.search(r".{1,60}?".join(parts), ob, flags=re.S)
    return (m.start(), m.end()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True, help="대상 article_id 목록 JSON")
    ap.add_argument("--masked", default="", help="mask_names.py 산출 csv. 없으면 원문으로 계산")
    ap.add_argument("--out", default=os.path.join(ROOT, "crystal", "data", "keyword", "cc_sentences_top5.jsonl"))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ids = json.load(open(a.ids, encoding="utf-8"))
    rows = {}
    with open(ARTICLES, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["article_id"] in ids:
                rows[r["article_id"]] = r
    masked = None
    if a.masked:
        import pandas as pd
        masked = pd.read_csv(a.masked, dtype={"news_id": str}).set_index("news_id")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(a.device).eval()
    done = {json.loads(l)["article_id"] for l in open(a.out, encoding="utf-8")} if os.path.exists(a.out) else set()
    todo = [i for i in ids if i not in done]
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as f:
        for n, aid in enumerate(todo, 1):
            r = rows[aid]; target = int(r["final_label"]); ob = r["body"]
            if masked is not None and aid in masked.index:
                t, b = normalize(str(masked.loc[aid, "title"] or ""), str(masked.loc[aid, "description"] or ""))
            else:
                t, b = normalize(r["title"], ob)
            enc = tok(t, b, truncation="only_second", max_length=512, return_offsets_mapping=True)
            bt = [e for (s, e), q in zip(enc["offset_mapping"], enc.sequence_ids()) if q == 1]
            sents = split(b, max(bt)) if bt else []
            rec = {"article_id": aid, "final_label": target, "sentence": None, "top": [], "weights": []}
            if len(sents) >= 2:
                rng = np.random.default_rng(42)
                masks = rng.integers(0, 2, size=(a.n, len(sents)))
                pairs = [(t, " ".join(b[s:e] for (s, e), k in zip(sents, m) if k)) for m in masks]
                ys = []
                for i in range(0, a.n, 32):
                    x = tok([p[0] for p in pairs[i:i + 32]], [p[1] for p in pairs[i:i + 32]], truncation="only_second",
                            max_length=512, padding=True, return_tensors="pt").to(a.device)
                    with torch.no_grad():
                        lp = torch.log_softmax(model(**x).logits.float(), -1)
                    ys += (lp[:, target] - lp[:, 1 - target]).tolist()
                reg = LassoCV(cv=5, random_state=0).fit(masks, np.array(ys))
                w = reg.coef_
                rec["r2"] = round(float(reg.score(masks, np.array(ys))), 3)
                order = np.argsort(-w)
                rec["weights"] = [{"text": b[sents[i][0]:sents[i][1]][:200], "w": round(float(w[i]), 4)} for i in order[:5]]
                for i in order:
                    if w[i] <= 0 or len(rec["top"]) >= 5:
                        break
                    loc = to_orig(b[sents[i][0]:sents[i][1]], ob)
                    if loc:
                        rec["top"].append({"start": loc[0], "end": loc[1], "phrase": ob[loc[0]:loc[1]], "w": round(float(w[i]), 4)})
                rec["sentence"] = rec["top"][0] if rec["top"] else None
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
