"""유저 전부(396,264명)에 대해 is_displayable 기사 전체(split 제한 없음)를 후보로 상위 K를 feeds.jsonl에 쓴다.
이미 본 기사는 제외. 진위로 거르지 않는다. 기본 CROWN (CRYSTAL_RECSYS=itemknn 또는 --model itemknn 이면 ItemKNN)."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from crystal.common import is_displayable  # noqa: E402
from crystal.recsys.dataset import DATA, distinct_history, iter_users, load_articles  # noqa: E402
from crystal.recsys.recommender import load_recommender  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, default=50)
p.add_argument("--batch", type=int, default=None, help="유저 배치 (기본: itemknn 20000, crown 2048)")
p.add_argument("--model", default=None, help="기본: 환경변수 CRYSTAL_RECSYS 또는 crown")
p.add_argument("--out", default=os.path.join(DATA, "feeds.jsonl"))
args = p.parse_args()
name = args.model or os.environ.get("CRYSTAL_RECSYS", "crown")
t0 = time.time()

articles = load_articles()
rec = load_recommender(name, articles=articles)
cands = [a["article_id"] for a in articles if is_displayable(a["title"], a["body"])]
cand_pos = {a: i for i, a in enumerate(cands)}

if name == "itemknn":
    batch = args.batch or 20000
    pop = rec.pop_of(cands).astype(np.float32)
    print(f"users {len(rec.user_ids)}  candidates {len(cands)}  load {time.time() - t0:.0f}s", flush=True)
    n_fallback = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for s0 in range(0, len(rec.user_ids), batch):
            sl = slice(s0, s0 + batch)
            s = rec.rank_rows(sl, cands)
            n_fallback += int((s.max(1) <= 0).sum())
            for b, h in enumerate(rec.histories[sl]):
                seen = [cand_pos[a] for a in h if a in cand_pos]
                if seen:
                    s[b, seen] = -np.inf
            top = np.lexsort((-np.broadcast_to(pop, s.shape), -s), axis=1)[:, :args.k]  # 유사도 합 내림차순, 동점은 인기도
            for u, row in zip(rec.user_ids[sl], top):
                fout.write(json.dumps({"user_id": u, "feed": [cands[j] for j in row]}) + "\n")
    print(f"wrote {len(rec.user_ids)} users -> {args.out}  similarity-all-zero users (popularity fallback) {n_fallback}  {time.time() - t0:.0f}s", flush=True)
else:
    batch = args.batch or 2048
    print(f"candidates {len(cands)}  model load+news encode {time.time() - t0:.0f}s", flush=True)
    rows = np.arange(batch)[:, None]
    n_users = 0
    t1 = time.time()

    def flush(buf, fout):
        hists = [[h["article_id"] for h in u["history"]] for u in buf]
        s = rec.rank_batch(hists, cands)  # [B, C] CROWN 점수 (히스토리 최근 50개)
        for b, u in enumerate(buf):
            seen = [cand_pos[a] for a in distinct_history(u["history"]) if a in cand_pos]
            if seen:
                s[b, seen] = -np.inf
        part = np.argpartition(-s, args.k - 1, axis=1)[:, :args.k]
        r = rows[:len(buf)]
        top = part[r, np.argsort(-s[r, part], axis=1, kind="stable")]  # 점수 내림차순
        for u, row in zip(buf, top):
            fout.write(json.dumps({"user_id": u["user_id"], "feed": [cands[j] for j in row]}) + "\n")

    with open(args.out + ".tmp", "w", encoding="utf-8") as fout:
        buf = []
        for u in iter_users():
            buf.append(u)
            if len(buf) == batch:
                flush(buf, fout)
                n_users += len(buf)
                buf = []
                if n_users == batch:  # 첫 배치로 예상 소요 출력
                    per = (time.time() - t1) / batch
                    print(f"first batch {batch} users {time.time() - t1:.1f}s -> 전체 396,264명 예상 {per * 396264 / 60:.0f}분", flush=True)
                elif n_users % (batch * 20) == 0:
                    print(f"{n_users} users  {time.time() - t1:.0f}s", flush=True)
        if buf:
            flush(buf, fout)
            n_users += len(buf)
    os.replace(args.out + ".tmp", args.out)
    print(f"wrote {n_users} users -> {args.out}  {time.time() - t0:.0f}s", flush=True)
