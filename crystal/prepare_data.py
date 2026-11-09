"""원천 csv -> crystal/data/articles.jsonl, users.jsonl (데이터 계약 형식).
2026-10-21: 본문 없는 기사를 제거하지 않는다. 중복 news_id만 첫 줄 유지(18,124건). 화면·추천 제외는 crystal/common.is_displayable이 맡는다.
split은 K-fold 이후 전건 "test"(모든 기사가 out-of-fold), fold는 veracity/predict.py가 붙인다."""
import ast
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from crystal.common import is_displayable  # noqa: E402

SRC = r"C:\Users\USER\Desktop\졸프_데이터\data\DATA\FAKENEWSNET_문장추가"
OUT = os.path.join(ROOT, "crystal", "data")
os.makedirs(OUT, exist_ok=True)

users = defaultdict(list)
n_art, seen, dup, not_disp = 0, set(), 0, Counter()
with open(os.path.join(OUT, "articles.jsonl"), "w", encoding="utf-8") as fa:
    for d in ("gossip", "pol"):
        df = pd.read_csv(os.path.join(SRC, f"{d}.csv"), dtype=str).fillna("")
        for r in df.itertuples(index=False):
            if r.news_id in seen:
                dup += 1
                continue
            seen.add(r.news_id)
            if not is_displayable(r.title, r.description):
                not_disp[d] += 1
            fa.write(json.dumps({"article_id": r.news_id, "dataset": d, "title": r.title, "body": r.description,
                                 "label": int(r.label), "split": "test"}, ensure_ascii=False) + "\n")
            n_art += 1
            uids, times = ast.literal_eval(r.user_ids or "[]"), ast.literal_eval(r.user_times or "[]")
            for u, t in zip(uids, times):
                users[str(u)].append({"article_id": r.news_id, "time": int(t)})

with open(os.path.join(OUT, "users.jsonl"), "w", encoding="utf-8") as fu:
    for u, h in users.items():
        fu.write(json.dumps({"user_id": u, "history": sorted(h, key=lambda x: x["time"])}) + "\n")

lens = [len(h) for h in users.values()]
print(f"duplicate_id dropped {dup}  not_displayable {dict(not_disp)} (kept)")
print(f"articles {n_art}  users {len(users)}  interactions {sum(lens)}  history median {sorted(lens)[len(lens)//2]}  >=5: {sum(l >= 5 for l in lens)}")
