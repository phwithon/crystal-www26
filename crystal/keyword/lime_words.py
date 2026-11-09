"""LIME 단어 기여도 (2026-11-03). RoBERTa 분류기(기본: 인물명 마스킹 fold0 모델)에 LIME을 적용해
판정 라벨 쪽으로 기여가 큰 단어 top-K를 뽑고, 원문 본문 좌표로 저장한다 (crystal/data/keyword/lime_masked_keywords.jsonl 형식).
IG(ig.py)와 함께 '단서 후보'를 보는 용도이며, 최종 데모 색칠은 build_highlights.py(ContextCite 문장 + lexicon 단어)가 만든다.
실행: $env:KMP_DUPLICATE_LIB_OK='TRUE'; python -X utf8 crystal/keyword/lime_words.py --model <모델 폴더> --ids <ids.json> [--out ...] [--samples 500] [--top 10]"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from crystal.veracity.loader import normalize  # noqa: E402

ARTICLES = os.path.join(ROOT, "crystal", "data", "articles.jsonl")
WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]+")


class Clf:
    """LIME용 확률 함수: 제목을 고정하고 본문 변형을 받는다."""

    def __init__(self, model_dir, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
        self.device, self.title = device, ""

    def __call__(self, bodies):
        out = []
        for i in range(0, len(bodies), 32):
            x = self.tok([self.title] * len(bodies[i:i + 32]), list(bodies[i:i + 32]), truncation="only_second",
                         max_length=512, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out.append(torch.softmax(self.model(**x).logits, -1).float().cpu().numpy())
        return np.concatenate(out)


def sent_of(text, pos):
    s = max([m.end() for m in re.finditer(r"[.!?][\"”’']?\s+", text[:pos])] or [0])
    m = re.search(r"[.!?](\s|$)", text[pos:])
    return text[s:pos + m.end() if m else len(text)].strip()


def explain(clf, title, body, label, samples, top):
    clf.title = title
    exp = LimeTextExplainer(class_names=["true", "fake"], split_expression=r"\W+", bow=True, random_state=42)
    e = exp.explain_instance(body, clf, labels=(label,), num_features=top * 3, num_samples=samples)
    words = [(w, s) for w, s in e.as_list(label=label) if s > 0 and WORD.fullmatch(w)]
    return words[:top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="RoBERTa 폴더 (예: roberta_veracity/folds/fold0_masked/best)")
    ap.add_argument("--ids", required=True, help="대상 article_id 목록 JSON")
    ap.add_argument("--masked", default="", help="마스킹 텍스트 csv (mask_names.py 산출). 없으면 원문으로 계산")
    ap.add_argument("--out", default=os.path.join(ROOT, "crystal", "data", "keyword", "lime_masked_keywords.jsonl"))
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--top", type=int, default=10)
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
    clf = Clf(a.model)
    done = {json.loads(l)["article_id"] for l in open(a.out, encoding="utf-8")} if os.path.exists(a.out) else set()
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as f:
        for n, aid in enumerate([i for i in ids if i not in done], 1):
            r = rows[aid]
            if masked is not None and aid in masked.index:
                t, b = normalize(str(masked.loc[aid, "title"] or ""), str(masked.loc[aid, "description"] or ""))
            else:
                t, b = normalize(r["title"], r["body"])
            label = int(r["final_label"])
            words = explain(clf, t, b, label, a.samples, a.top)
            kws = []
            for w, s in words:  # 본문에서 단어 위치(마스킹 전 원문 기준, 첫 등장) 찾기
                m = re.search(r"(?<![\w])" + re.escape(w) + r"(?![\w])", r["body"], flags=re.I)
                if not m:
                    continue
                kws.append({"phrase": r["body"][m.start():m.end()], "field": "body", "start": m.start(), "end": m.end(),
                            "score": round(float(s), 4), "sentence": sent_of(r["body"], m.start())})
            f.write(json.dumps({"article_id": aid, "final_label": label, "method": "lime", "keywords": kws}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"  {n} ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
