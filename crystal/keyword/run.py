"""전체 기사(18,124, 본문 없는 기사 포함)에 IG → top-K → spaCy 확장으로 keywords를 붙여 articles.jsonl을 같은 순서로 덮어쓴다.
IG는 기사마다 자기 fold 모델(loader.load(fold))로 계산하며 fold별로 묶어 모델을 한 번씩만 올린다.
실행: $env:KMP_DUPLICATE_LIB_OK='TRUE'; python -X utf8 crystal/keyword/run.py [--score signed|l2]   (ig_raw.jsonl로 중단 후 이어서 실행 가능)"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from crystal.keyword.expand import Expander  # noqa: E402
from crystal.keyword.ig import DEFAULT_SCORE, SCORE_MODES, IGAnalyzer, select_top_words  # noqa: E402
from crystal.veracity.loader import load, normalize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTICLES = os.path.join(ROOT, "crystal", "data", "articles.jsonl")
RAW = os.path.join(ROOT, "crystal", "data", "keyword", "ig_raw.jsonl")
WS = os.path.join(ROOT, "_workspace", "03_keyword")
TOP_K, SEED = 10, 42


def run_ig(rows):
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    done = {json.loads(l)["article_id"] for l in open(RAW, encoding="utf-8")} if os.path.exists(RAW) else set()
    todo = [r for r in rows if r["article_id"] not in done]
    print(f"IG: {len(done)} done, {len(todo)} todo", flush=True)
    if not todo:
        return
    torch.manual_seed(SEED)
    t0, n = time.time(), 0
    with open(RAW, "a", encoding="utf-8") as f:
        for fold in sorted({r["fold"] for r in todo}):
            tok, model = load(fold)
            an = IGAnalyzer(tok, model)
            for r in [x for x in todo if x["fold"] == fold]:
                words, meta = an.analyze(r["title"], r["body"], r["final_label"])
                t_off, b_off = len(r["title"]) - len(r["title"].lstrip()), len(r["body"]) - len(r["body"].lstrip())
                for w in words:  # 위치를 원문(strip 전) 기준으로
                    o = t_off if w["field"] == "title" else b_off
                    w["start"], w["end"] = w["start"] + o, w["end"] + o
                    w["score"], w["signed"] = round(w["score"], 6), round(w["signed"], 5)
                f.write(json.dumps({"article_id": r["article_id"], "fold": fold, "final_label": r["final_label"], "meta": meta, "words": words}, ensure_ascii=False) + "\n")
                n += 1
                if n % 100 == 0 or n == len(todo):
                    f.flush()
                    print(f"  {n}/{len(todo)} fold {fold} ({time.time() - t0:.0f}s)", flush=True)
            del an, model
            torch.cuda.empty_cache()


def run_expand(rows, score):
    raw = {}
    for l in open(RAW, encoding="utf-8"):
        d = json.loads(l)
        raw[d["article_id"]] = d
    ex = Expander()
    os.makedirs(WS, exist_ok=True)
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        d = raw[r["article_id"]]
        top = select_top_words(d["words"], TOP_K, score)
        body = "" if normalize(r["title"], r["body"])[1] == "" else r["body"]
        r["keywords"], stats = ex.expand(r["title"], body, top)
        for k in r["keywords"]:
            assert (r["title"] if k["field"] == "title" else r["body"])[k["start"]:k["end"]] == k["phrase"], r["article_id"]
        json.dump({"article": {"id": r["article_id"], "title": r["title"], "text": r["body"]},
                   "prediction": {"label": {0: "Real", 1: "Fake"}[r["final_label"]], "prob_fake": r["pred_prob_fake"]},
                   "keywords": r["keywords"], "expansion": stats,
                   "metadata": {"model": f"roberta_veracity/folds/fold{d['fold']}/best", "ig_steps": 50, "top_k": TOP_K, "score": score, "spacy": "en_core_web_sm", "ig_meta": d["meta"]}},
                  open(os.path.join(WS, f"article_{r['article_id']}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if i % 300 == 0 or i == len(rows):
            print(f"  expand {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)


def write_keywords(kw):
    """쓰기 직전에 다시 읽어 그 시점 내용 위에 keywords만 바꾼다 (다른 모듈이 같은 파일에 쓰는 중일 수 있음)."""
    for _ in range(5):
        rows = [json.loads(l) for l in open(ARTICLES, encoding="utf-8")]
        n_cp = sum("counterparts" in r for r in rows)
        for r in rows:
            if r["article_id"] in kw:
                r["keywords"] = kw[r["article_id"]]
        tmp = ARTICLES + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        chk = [json.loads(l) for l in open(ARTICLES, encoding="utf-8")]
        if len(chk) == len(rows) and sum("counterparts" in r for r in chk) <= n_cp:
            os.replace(tmp, ARTICLES)
            print(f"wrote {len(rows)} rows, keywords on {len(kw)}, counterparts on {n_cp}")
            return
        print("articles.jsonl changed while writing, retry", flush=True)
    raise RuntimeError("articles.jsonl keeps changing; not written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", choices=SCORE_MODES, default=DEFAULT_SCORE)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(ARTICLES, encoding="utf-8")]
    run_ig(rows)
    run_expand(rows, args.score)
    write_keywords({r["article_id"]: r["keywords"] for r in rows})


if __name__ == "__main__":
    main()
