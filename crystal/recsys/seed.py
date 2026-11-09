"""시드 유저 재선정: 첫 12건에 FAKE 3~5건 + 반대 진위 짝이 있는 FAKE 기사 ≥1 + 7일/전체 fake 비율 차 ≥ gap. 짝 집합 두 개(test 풀, 전체 풀) 각각 출력.
python -X utf8 crystal/recsys/seed.py"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from crystal.recsys.dataset import DATA, iter_users, load_articles  # noqa: E402
from crystal.recsys.stats import load_feeds, user_stats  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--label-field", default="final_label")
p.add_argument("--page", type=int, default=12)
p.add_argument("--gaps", type=float, nargs="+", default=[0.3, 0.2])
p.add_argument("--top", type=int, default=5)
p.add_argument("--json", default=None, help="선정 후보를 json으로 저장 (드라이버용)")
a = p.parse_args()

by_id = {x["article_id"]: x for x in load_articles()}
pool_all = {}
with open(os.path.join(DATA, "event", "counterparts_pool_all.jsonl"), encoding="utf-8") as f:
    for d in map(json.loads, f):
        pool_all[d["article_id"]] = d.get("counterparts") or []
sets = {"test 풀 짝 (articles.jsonl counterparts)": {k: len(v["counterparts"]) for k, v in by_id.items()
                                                  if v.get("split") == "test" and v.get(a.label_field) == 1 and v.get("counterparts")},
        "전체 풀 짝 (event/counterparts_pool_all.jsonl)": {k: len(v) for k, v in pool_all.items()
                                                       if by_id.get(k, {}).get(a.label_field) == 1 and v}}
for name, s in sets.items():
    print(f"{name}: 짝 있는 FAKE 기사 {len(s)}건", flush=True)

feeds = load_feeds()
rows = {name: [] for name in sets}  # (gap, uid, stats, [(fake_id, n_cp)])
out_json = {}
for u in iter_users():
    feed = feeds.get(u["user_id"])
    if not feed:
        continue
    page = feed[:a.page]
    st = user_stats(u["history"], feed, by_id, a.label_field, a.page)
    if not (3 <= st["feed"]["n_fake"] <= 5) or st["weekly"]["total"] < 3 or st["overall"]["total"] < 10:
        continue
    if st["weekly"]["total"] == st["overall"]["total"]:
        continue
    gap = abs(st["weekly"]["fake"] - st["overall"]["fake"])
    if gap < min(a.gaps):
        continue
    for name, s in sets.items():
        cp = [(x, s[x]) for x in page if x in s]
        if cp:
            rows[name].append((gap, u["user_id"], st, cp))

for name in sets:
    cands = sorted(rows[name], key=lambda r: -r[0])
    used = None
    for g in sorted(a.gaps, reverse=True):
        sel = [r for r in cands if r[0] >= g]
        if sel:
            used = g
            break
    print(f"\n### {name}\n조건: 첫 {a.page}건 FAKE 3~5, 짝 있는 FAKE ≥1, 7일 total ≥3, 전체 total ≥10, fake 비율 차 ≥ {used or min(a.gaps)}"
          + (f" (완화: {max(a.gaps)} → {used})" if used is not None and used < max(a.gaps) else "") + f". 후보 {len(sel) if used else 0}명.")
    if not used:
        print("후보 없음")
        continue
    print("\n| user_id | 첫 12건 FAKE | 2~5페이지 FAKE | 짝 있는 FAKE 기사 (짝 수) | 7일 TRUE·FAKE/TOTAL (fake) | 전체 TRUE·FAKE/TOTAL (fake) | 차 |\n|---|---|---|---|---|---|---|")
    out_json[name] = {"gap_used": used, "candidates": []}
    for gap, uid, st, cp in sel[:a.top]:
        w, o = st["weekly"], st["overall"]
        feed = feeds[uid]
        pages = [sum(by_id[x][a.label_field] == 1 for x in feed[k * a.page:(k + 1) * a.page] if x in by_id) for k in range((len(feed) + a.page - 1) // a.page)]
        print(f"| {uid} | {st['feed']['n_fake']} | {'/'.join(map(str, pages[1:5]))} | {', '.join(f'{x} ({n})' for x, n in cp)} | {w['n_true']}·{w['n_fake']}/{w['total']} ({w['fake']:.2f}) "
              f"| {o['n_true']}·{o['n_fake']}/{o['total']} ({o['fake']:.2f}) | {gap:.2f} |")
        out_json[name]["candidates"].append({"user_id": uid, "gap": round(gap, 4), "feed_fake_first_page": st["feed"]["n_fake"], "pages_fake": pages,
                                             "fake_with_pair": cp, "weekly": w, "overall": o})
if a.json:
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=1)
