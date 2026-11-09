"""데모 색칠 파일 생성 (2026-11-03~04). cc_sentences_top5.jsonl(ContextCite 문장) + lexicon.py(단서 사전) → crystal/data/keyword/cc_keywords.jsonl
규칙:
  1) 문장: 본문 길이에 따라 1~5개 (300단어 미만 1, <500 2, <800 3, <1200 4, 그 이상 5). 순위대로 넣되,
     이미 칠한 문장과 붙어 있거나 겹치면 건너뛰고 다음 순위 문장을 쓴다(5순위까지). 단어 중간에서 시작하거나 끝이 잘린 문장은 원문 문장 경계로 늘린다.
  2) 단어: FAKE 판정 기사는 FAKE 목록, TRUE 판정 기사는 TRUE 목록을 본문에서 매칭(등장할 때마다, 최대 15개). 칠한 문장 안의 단어는 칠하지 않는다.
  3) 설명: 단어는 lexicon 템플릿, 문장은 --explain 으로 준 JSON(id|start → 글)에서 가져와 rationale_cache_lime_masked.jsonl에 넣는다.
     설명 JSON이 없으면 문장 설명은 캐시에 넣지 않는다(웹에서 Gemini가 생성).
실행: python -X utf8 crystal/keyword/build_highlights.py [--sentences cc_sentences_top5.jsonl] [--explain a.json b.json ...] [--out cc_keywords.jsonl] [--max-words 15]
웹: CRYSTAL_KEYWORDS=cc 로 서버를 켜면 이 파일의 색칠이 기사에 적용된다(crystal/web/data.py)."""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from crystal.keyword.lexicon import FAKE, TRUE  # noqa: E402

ARTICLES = os.path.join(ROOT, "crystal", "data", "articles.jsonl")
KW = os.path.join(ROOT, "crystal", "data", "keyword")
CACHE = os.path.join(ROOT, "crystal", "data", "rationale_cache_lime_masked.jsonl")
END = re.compile(r"[.!?][\"”’'\)]*$")


def n_of(words):
    return 1 if words < 300 else 2 if words < 500 else 3 if words < 800 else 4 if words < 1200 else 5


def sent_of(text, pos):
    s = max([m.end() for m in re.finditer(r"[.!?][\"”’']?\s+", text[:pos])] or [0])
    m = re.search(r"[.!?](\s|$)", text[pos:])
    return text[s:pos + m.end() if m else len(text)].strip()


def fix_bounds(body, s, e, others):
    """단어 중간 시작 → 문장 처음으로, 잘린 끝 → 문장 끝으로(다른 후보 문장을 삼키지 않는 범위에서)."""
    if s > 0 and body[s - 1].isalnum():
        s = max([m.end() for m in re.finditer(r"[.!?][\"”’']?\s+", body[:s])] or [0])
    if not END.search(body[s:e]):
        m = re.search(r"[.!?][\"”’'\)]*(?=\s|$)", body[e:e + 400])
        if m and not any(s < o["start"] < e + m.end() for o in others):
            e += m.end()
    return s, e


def touching(body, a, b):
    return not (a[1] <= b[0] or a[0] >= b[1]) or body[min(a[1], b[1]):max(a[0], b[0])].strip() == ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default=os.path.join(KW, "cc_sentences_top5.jsonl"))
    ap.add_argument("--explain", nargs="*", default=[], help="문장 설명 JSON들 ({'<article_id>|<start>': text})")
    ap.add_argument("--out", default=os.path.join(KW, "cc_keywords.jsonl"))
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--max-words", type=int, default=15)
    a = ap.parse_args()
    expl = {}
    for p in a.explain:
        expl.update(json.load(open(p, encoding="utf-8")))
    cc = {}
    for l in open(a.sentences, encoding="utf-8"):
        d = json.loads(l); cc[d["article_id"]] = d
    rows = {}
    for l in open(ARTICLES, encoding="utf-8"):
        r = json.loads(l)
        if r["article_id"] in cc:
            rows[r["article_id"]] = r
    out, cache_add = [], []
    for aid, c in cc.items():
        r = rows[aid]; body = r["body"]; lex = FAKE if r["final_label"] == 1 else TRUE
        cands = c.get("top") or ([c["sentence"]] if c.get("sentence") else [])
        kws, chosen = [], []
        for s in cands:  # 순위대로, 붙어 있으면 건너뛰기
            if len(chosen) >= n_of(len(body.split())):
                break
            ns, ne = fix_bounds(body, s["start"], s["end"], cands)
            if any(touching(body, (ns, ne), (x["start"], x["end"])) for x in chosen):
                continue
            k = {"phrase": body[ns:ne], "field": "body", "start": ns, "end": ne, "score": round(1.0 - 0.1 * len(chosen), 2),
                 "sentence": body[ns:ne], "kind": "sentence"}
            chosen.append(k); kws.append(k)
            text = expl.get(f"{aid}|{s['start']}")
            if text:
                cache_add.append({"article_id": aid, "kind": "keyword", "key": k["phrase"], "text": text,
                                  "model": "manual", "prompt_version": "cc-sentence-v3"})
        taken = [(k["start"], k["end"]) for k in kws]
        found = []
        for pat, tmpl in lex:
            for m in re.finditer(r"(?<![\w])(?:" + pat + r")(?![\w])", body, flags=re.I):
                s0, e0 = m.start(), m.end()
                if any(not (e0 <= x or s0 >= y) for x, y in taken):
                    continue
                taken.append((s0, e0))
                found.append(({"phrase": body[s0:e0], "field": "body", "start": s0, "end": e0, "score": 0.6,
                               "sentence": sent_of(body, s0), "kind": "lexicon"}, tmpl))
        found.sort(key=lambda x: x[0]["start"])
        seen = set()
        for k, tmpl in found[:a.max_words]:
            kws.append(k)
            if k["phrase"] not in seen:
                seen.add(k["phrase"])
                cache_add.append({"article_id": aid, "kind": "keyword", "key": k["phrase"], "text": tmpl.format(p=k["phrase"]),
                                  "model": "lexicon", "prompt_version": "lexicon-v1"})
        kws.sort(key=lambda k: k["start"])
        for k in kws:
            assert body[k["start"]:k["end"]] == k["phrase"], (aid, k["phrase"][:30])
        out.append({"article_id": aid, "final_label": r["final_label"], "method": "contextcite+lexicon", "keywords": kws})
    with open(a.out, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    cur = [json.loads(l) for l in open(a.cache, encoding="utf-8")] if os.path.exists(a.cache) else []
    newkeys = {(c["article_id"], c["kind"], c["key"]) for c in cache_add}
    cur = [c for c in cur if (c["article_id"], c["kind"], c["key"]) not in newkeys] + cache_add
    with open(a.cache + ".tmp", "w", encoding="utf-8") as f:
        for c in cur:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(a.cache + ".tmp", a.cache)
    n_s = sum(sum(k["kind"] == "sentence" for k in d["keywords"]) for d in out)
    n_w = sum(sum(k["kind"] == "lexicon" for k in d["keywords"]) for d in out)
    print(f"articles {len(out)} | sentences {n_s} | words {n_w} | cache entries added {len(cache_add)} -> {a.out}")


if __name__ == "__main__":
    main()
