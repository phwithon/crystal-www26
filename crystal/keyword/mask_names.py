"""인물명 마스킹 (2026-11-03). spaCy en_core_web_sm의 PERSON 개체를 "[NAME]"으로 바꾼 제목·본문을 masked.csv로 만든다.
이유: 원래 RoBERTa는 특정 인물명(Bieber, Gomez 등)을 가짜 신호로 크게 쓰기 때문에, 기여도(LIME/ContextCite) 계산용 모델은
인물명을 가린 텍스트로 다시 학습한다(roberta_veracity/train_masked.py). 판정 자체는 원래 10-fold 모델을 그대로 쓴다.
실행: python -X utf8 crystal/keyword/mask_names.py [--out crystal/data/keyword/masked.csv] [--limit N]"""
import argparse
import json
import os
import sys

import pandas as pd
import spacy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTICLES = os.path.join(ROOT, "crystal", "data", "articles.jsonl")


def mask(nlp, text):
    text = text or ""
    if not text.strip():
        return text
    doc = nlp(text[:100000])
    out, pos = [], 0
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        out.append(text[pos:ent.start_char]); out.append("[NAME]"); pos = ent.end_char
    out.append(text[pos:])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "crystal", "data", "keyword", "masked.csv"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger"])
    rows = []
    with open(ARTICLES, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            r = json.loads(line)
            rows.append({"news_id": r["article_id"], "dataset": r.get("dataset", ""), "label": r.get("label"),
                         "title": mask(nlp, r.get("title")), "description": mask(nlp, r.get("body"))})
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
