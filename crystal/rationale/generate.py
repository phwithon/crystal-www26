"""시드 유저 피드 앞 페이지 기사 + 그 counterparts의 설명을 미리 만들어 캐시와 articles.jsonl에 넣는다.
실행: $env:KMP_DUPLICATE_LIB_OK='TRUE'; python -X utf8 -m crystal.rationale.generate --users 721944259,586105624 --pages 2"""
import argparse
import json
import os
import sys
import time

from crystal.common import is_displayable
from crystal.rationale import core

FEEDS_PATH = os.path.join(core.DATA_DIR, "feeds.jsonl")
PAGE_SIZE = 12
EXPECT_LINES = 18124  # 2026-10-21 K-fold: 전건 keywords, counterparts는 is_displayable 기사만


def read_feeds(user_ids):
    want = {f'"user_id": "{u}"'.encode() for u in user_ids}
    out = {}
    with open(FEEDS_PATH, "rb") as f:
        for line in f:
            head = line[:80]
            if any(w in head for w in want):
                d = json.loads(line)
                out[d["user_id"]] = d["feed"]
                if len(out) == len(user_ids):
                    break
    return out


def targets(feeds, pages, articles):
    feed_ids = []
    for ids in feeds.values():
        for i in ids[:pages * PAGE_SIZE]:
            if i in articles and i not in feed_ids:
                feed_ids.append(i)
    cp_ids = []
    for i in feed_ids:
        for c in articles[i].get("counterparts") or []:
            if c["article_id"] in articles and c["article_id"] not in feed_ids and c["article_id"] not in cp_ids:
                cp_ids.append(c["article_id"])
    return feed_ids, cp_ids


def run_generation(llm, cache, articles, feed_ids, cp_ids, stats):
    done = {"keyword": 0, "verdict": 0, "contrast": 0, "skipped_quota": []}
    for n, aid in enumerate(feed_ids + cp_ids, 1):
        a = articles[aid]
        try:  # 429 재시도 뒤에도 실패하면 그 기사만 건너뛰고 계속
            kws = [k for k in core.unique_phrases(a) if cache.get(aid, "keyword", k["phrase"]) is None]
            if kws:
                core.gen_keywords(llm, cache, a, kws, articles, stats)
                done["keyword"] += 1
            if cache.get(aid, "verdict", "") is None:
                core.gen_verdict(llm, cache, a, articles, stats)
                done["verdict"] += 1
            for c in a.get("counterparts") or []:
                b = articles.get(c["article_id"])
                if b and cache.get(aid, "contrast", b["article_id"]) is None:
                    core.gen_contrast(llm, cache, a, b, stats)
                    done["contrast"] += 1
        except core.QuotaExceeded as e:
            done["skipped_quota"].append(aid)
            print(f"  skip {aid}: {e}", flush=True)
        print(f"[{n}/{len(feed_ids) + len(cp_ids)}] {aid} calls={llm.calls} discarded={stats['discarded']}", flush=True)
    return done


def apply_cache(cache, path=core.ARTICLES_PATH):
    """캐시를 articles.jsonl에 같은 순서로 반영. 임시 파일 → os.replace, 전후 줄 수 검사."""
    by_article = {}
    for (aid, kind, key), r in cache.rows.items():
        by_article.setdefault(aid, []).append((kind, key, r["text"]))
    tmp = path + ".tmp"
    n_in = n_out = n_kw = n_cp = n_disp = 0
    with open(path, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as g:
        for line in f:
            n_in += 1
            a = json.loads(line)
            for kind, key, text in by_article.get(a["article_id"], []):
                if kind == "keyword":
                    a.setdefault("rationales", {})
                    if a["rationales"] is None:
                        a["rationales"] = {}
                    a["rationales"][key] = text
                elif kind == "verdict":
                    a["verdict_rationale"] = text
                elif kind == "contrast":
                    for c in a.get("counterparts") or []:
                        if c["article_id"] == key:
                            c["claim_contrast"] = json.loads(text) if isinstance(text, str) else text
            n_kw += "keywords" in a  # 빈 리스트(키워드 0개)도 필드가 있으면 센다
            n_cp += a.get("counterparts") is not None
            n_disp += is_displayable(a.get("title"), a.get("body"))
            g.write(json.dumps(a, ensure_ascii=False) + "\n")
            n_out += 1
    if not (n_in == n_out == EXPECT_LINES and n_kw == EXPECT_LINES and n_cp == n_disp):
        os.remove(tmp)
        raise RuntimeError(f"line check failed: in={n_in} out={n_out} keywords={n_kw} counterparts={n_cp} displayable={n_disp}")
    os.replace(tmp, path)
    return n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="721944259,586105624")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--model", default=core.DEFAULT_MODEL)
    ap.add_argument("--fallback", default=core.FALLBACK_MODEL)
    ap.add_argument("--interval", type=float, default=4.0, help="최소 호출 간격(초)")
    ap.add_argument("--apply-only", action="store_true", help="생성 없이 캐시만 articles.jsonl에 반영")
    args = ap.parse_args()

    t0 = time.time()
    articles = core.load_articles()
    cache = core.Cache()
    feeds = read_feeds(args.users.split(","))
    feed_ids, cp_ids = targets(feeds, args.pages, articles)
    n_pairs = sum(len(articles[i].get("counterparts") or []) for i in feed_ids)
    print(f"targets: feed articles={len(feed_ids)} counterpart articles={len(cp_ids)} pairs={n_pairs} cache rows={len(cache.rows)}")
    stats, done, status = core.new_stats(), {}, "complete"
    llm = None
    if not args.apply_only:
        llm = core.Gemini(args.model, args.fallback, interval=args.interval)
        try:
            done = run_generation(llm, cache, articles, feed_ids, cp_ids, stats)
        except core.QuotaExceeded as e:
            status = f"stopped: {e}"
        except KeyboardInterrupt:
            status = "stopped: interrupted"
    n = apply_cache(cache)
    missing_v = [i for i in feed_ids + cp_ids if cache.get(i, "verdict", "") is None]
    missing_k = sum(1 for i in feed_ids for k in core.unique_phrases(articles[i]) if cache.get(i, "keyword", k["phrase"]) is None)
    summary = {"status": status, "feed_articles": len(feed_ids), "counterpart_articles": len(cp_ids), "pairs": n_pairs,
               "calls": llm.calls if llm else 0, "api_errors": llm.errors if llm else 0, "quota_429": llm.quota_hits if llm else 0,
               "model": args.model, "fallback": args.fallback, "calls_by_model": llm.used if llm else {}, "error_codes": llm.codes if llm else {}, "generated_units": done, **stats,
               "cache_rows": len(cache.rows), "missing_verdict": missing_v, "missing_keyword_rationales": missing_k,
               "articles_written": n, "seconds": round(time.time() - t0)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
