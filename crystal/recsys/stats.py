"""진위 분포: 히스토리 최근 7일(마지막 상호작용 기준)·전체·현재 피드의 라벨 비율 {true, fake, n_true, n_fake, total}.
라벨 필드는 기본 final_label. python -X utf8 crystal/recsys/stats.py --label-field label 로 시드 유저 후보를 찍는다."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from crystal.recsys.dataset import DATA, iter_users, load_articles  # noqa: E402

DAY = 86400


def label_dist(article_ids, articles_by_id, label_field="final_label"):
    """라벨 필드가 없는 기사는 빼고 센다. total 0이면 비율은 0.0."""
    labels = [articles_by_id[a][label_field] for a in article_ids if a in articles_by_id and articles_by_id[a].get(label_field) is not None]
    n_fake = sum(1 for l in labels if l == 1)
    n_true = len(labels) - n_fake
    total = len(labels)
    return {"true": n_true / total if total else 0.0, "fake": n_fake / total if total else 0.0,
            "n_true": n_true, "n_fake": n_fake, "total": total}


def history_dist(history, articles_by_id, days=None, label_field="final_label"):
    """days가 있으면 유저의 마지막 상호작용 시각부터 days일 안의 기사만. 같은 기사는 한 번만 센다."""
    if days is not None and history:
        cut = history[-1]["time"] - days * DAY
        history = [h for h in history if h["time"] >= cut]
    return label_dist(list(dict.fromkeys(h["article_id"] for h in history)), articles_by_id, label_field)


def feed_dist(feed, articles_by_id, label_field="final_label"):
    return label_dist(feed, articles_by_id, label_field)


def user_stats(history, feed, articles_by_id, label_field="final_label", page=None):
    """사이드바 세 칸. page가 있으면 피드 분포는 첫 page건만 (figure2_a: 한 페이지 12건)."""
    return {"weekly": history_dist(history, articles_by_id, 7, label_field),
            "overall": history_dist(history, articles_by_id, None, label_field),
            "feed": feed_dist(feed[:page] if page else feed, articles_by_id, label_field)}


def load_feeds(path=os.path.join(DATA, "feeds.jsonl")):
    with open(path, encoding="utf-8") as f:
        return {d["user_id"]: d["feed"] for d in map(json.loads, f)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label-field", default="final_label")
    p.add_argument("--page", type=int, default=12)
    p.add_argument("--min-fake", type=int, default=3)
    p.add_argument("--max-fake", type=int, default=5)
    p.add_argument("--min-weekly", type=int, default=3)
    p.add_argument("--min-overall", type=int, default=10)
    p.add_argument("--min-gap", type=float, default=0.3)
    p.add_argument("--top", type=int, default=3)
    a = p.parse_args()
    by_id = {x["article_id"]: x for x in load_articles()}
    feeds = load_feeds()
    cands = []
    for u in iter_users():
        feed = feeds.get(u["user_id"])
        if not feed:
            continue
        s = user_stats(u["history"], feed, by_id, a.label_field, a.page)
        if not (a.min_fake <= s["feed"]["n_fake"] <= a.max_fake):
            continue
        if s["weekly"]["total"] < a.min_weekly or s["overall"]["total"] < a.min_overall or s["weekly"]["total"] == s["overall"]["total"]:
            continue
        gap = abs(s["weekly"]["fake"] - s["overall"]["fake"])
        if gap >= a.min_gap:
            cands.append((gap, u["user_id"], s))
    cands.sort(key=lambda x: -x[0])
    print(f"label_field={a.label_field}  candidates {len(cands)}")
    for gap, uid, s in cands[:a.top]:
        print(f"\nuser {uid}  gap {gap:.2f}")
        for k in ("weekly", "overall", "feed"):
            print(f"  {k:8s} TRUE {s[k]['n_true']} · FAKE {s[k]['n_fake']}  TOTAL {s[k]['total']}  fake {s[k]['fake']:.2f}")
