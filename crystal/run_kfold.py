"""K-fold 전환 드라이버 (2026-10-21). 단계 1~9를 순서대로 돌리고 완료 표시(_workspace/09_done/step{n}.ok)가 있는 단계는 건너뛴다.
어디서 멈추든 finally에서 보고서(_workspace/09_kfold_report.md)·CLAUDE.md 이력 1줄을 쓴다. 종료 명령(shutdown)은 이 스크립트에 넣지 못해 호출자가 끝난 뒤 실행한다.
실행: $env:KMP_DUPLICATE_LIB_OK='TRUE'; python -X utf8 crystal/run_kfold.py   (로그 _workspace/09_kfold.log)"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from crystal.common import is_displayable  # noqa: E402

DATA = os.path.join(ROOT, "crystal", "data")
RV = os.path.join(ROOT, "roberta_veracity")
WS = os.path.join(ROOT, "_workspace")
LOG = os.path.join(WS, "09_kfold.log")
DONE = os.path.join(WS, "09_done")
SUMMARY = os.path.join(WS, "09_summary.json")
REPORT = os.path.join(WS, "09_kfold_report.md")
ARTICLES = os.path.join(DATA, "articles.jsonl")
PY = [sys.executable, "-X", "utf8"]
ENV = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
PAGE, K_FEED = 12, 50
os.makedirs(DONE, exist_ok=True)
summary = json.load(open(SUMMARY, encoding="utf-8")) if os.path.exists(SUMMARY) else {}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save():
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


def run(args, name):
    """자식 프로세스 출력은 로그에 그대로 붙이고, 파싱용으로 줄 목록을 돌려준다."""
    log(f"RUN {name}: {' '.join(args)}")
    t0, lines = time.time(), []
    with open(LOG, "a", encoding="utf-8") as lf:
        p = subprocess.Popen(args, cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            lf.write(line); lf.flush()
            lines.append(line.rstrip("\n"))
        p.wait()
    log(f"END {name}: exit {p.returncode} ({time.time() - t0:.0f}s)")
    if p.returncode != 0:
        raise RuntimeError(f"{name} failed with exit {p.returncode}; last lines: " + " | ".join(lines[-5:]))
    return lines, round(time.time() - t0)


def step(n, name, fn):
    mark = os.path.join(DONE, f"step{n}.ok")
    if os.path.exists(mark):
        log(f"step {n} {name}: done earlier, skip")
        return
    log(f"===== step {n} {name} =====")
    summary[f"step{n}"] = fn()
    summary["last_completed_step"] = n
    save()
    open(mark, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"step {n} {name}: OK")


def read_articles():
    with open(ARTICLES, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def disp(a):
    return is_displayable(a.get("title"), a.get("body"))


# ---------- 단계 ----------
def s1_prepare():
    lines, sec = run(PY + [os.path.join(ROOT, "crystal", "prepare_data.py")], "prepare_data")
    rows = read_articles()
    n_users = sum(1 for _ in open(os.path.join(DATA, "users.jsonl"), encoding="utf-8"))
    out = {"n_articles": len(rows), "n_displayable": sum(map(disp, rows)), "n_users": n_users, "seconds": sec,
           "dataset": dict(Counter(r["dataset"] for r in rows)), "label": dict(Counter(r["label"] for r in rows)), "stdout": lines[-2:]}
    assert len(rows) == 18124, out
    return out


def s2_train():
    lines, sec = run(PY + [os.path.join(RV, "train_kfold.py")], "train_kfold")
    res = json.load(open(os.path.join(RV, "kfold_results.json"), encoding="utf-8"))
    n_pred = sum(1 for _ in open(os.path.join(RV, "kfold_predictions.csv"), encoding="utf-8")) - 1
    assert n_pred == 18124 and res["n"] == 18124, (n_pred, res["n"])
    return {"results": res, "seconds_this_run": sec}


def s3_predict():
    _, sec = run(PY + [os.path.join(ROOT, "crystal", "veracity", "predict.py")], "predict")
    rows = read_articles()
    probs = [r["pred_prob_fake"] for r in rows]
    hist = [0] * 10
    for p in probs:
        hist[min(int(p * 10), 9)] += 1
    by_ds = {}
    for d in ("gossip", "pol"):
        rs = [r for r in rows if r["dataset"] == d]
        by_ds[d] = {"n": len(rs), "acc": round(sum(r["pred_label"] == r["label"] for r in rs) / len(rs), 4)}
    return {"n": len(rows), "pooled_acc": round(sum(r["pred_label"] == r["label"] for r in rows) / len(rows), 4), "by_dataset": by_ds,
            "pred_fake": sum(r["pred_label"] for r in rows), "label_fake": sum(r["label"] for r in rows),
            "prob_mid_005_095": sum(0.05 <= p <= 0.95 for p in probs), "prob_mid_ratio": round(sum(0.05 <= p <= 0.95 for p in probs) / len(probs), 4),
            "prob_hist_0.1": hist, "fold_counts": dict(sorted(Counter(r["fold"] for r in rows).items())),
            "all_split_test": all(r["split"] == "test" for r in rows), "seconds": sec}


def s4_keyword():
    raw = os.path.join(DATA, "keyword", "ig_raw.jsonl")
    sub = os.path.join(DONE, "step4_raw_discarded")
    if not os.path.exists(sub):  # 기존 ig_raw는 단일 모델 결과라 폐기 (한 번만)
        if os.path.exists(raw):
            os.remove(raw); log("discarded old ig_raw.jsonl (single-model IG)")
        open(sub, "w").write("1")
    lines, sec = run(PY + [os.path.join(ROOT, "crystal", "keyword", "run.py")], "keyword_run")
    rows = read_articles()
    n_kw = sum(len(r.get("keywords") or []) for r in rows)
    ig_line = [l for l in lines if l.startswith("IG:")]
    n_raw = sum(1 for _ in open(raw, encoding="utf-8"))
    return {"n_articles_with_field": sum("keywords" in r for r in rows), "n_keywords": n_kw, "n_zero": sum(not r.get("keywords") for r in rows),
            "n_ig_raw": n_raw, "ig_header": ig_line, "seconds": sec,
            "dist": dict(sorted(Counter(len(r.get("keywords") or []) for r in rows).items())),
            "title": sum(k["field"] == "title" for r in rows for k in r.get("keywords") or []),
            "body": sum(k["field"] == "body" for r in rows for k in r.get("keywords") or [])}


def s5_match():
    st_path = os.path.join(WS, "09_event_stats.json")
    _, sec = run(PY + [os.path.join(ROOT, "crystal", "event", "match.py"), "--pool", "all", "--method", "cos",
                       "--out", os.path.join(DATA, "event", "counterparts_pool_all.jsonl"), "--stats", st_path, "--apply"], "match")
    st = json.load(open(st_path, encoding="utf-8"))
    st.pop("examples", None); st.pop("seed_feeds_top12", None)
    rows = read_articles()
    return {"stats": st, "n_with_counterparts_field": sum("counterparts" in r for r in rows), "seconds": sec}


def pick_seed():
    sj = os.path.join(WS, "09_seed.json")
    lines, _ = run(PY + [os.path.join(ROOT, "crystal", "recsys", "seed.py"), "--json", sj, "--top", "20"], "seed")
    key = "전체 풀 짝 (event/counterparts_pool_all.jsonl)"
    res = json.load(open(sj, encoding="utf-8"))
    cands = res.get(key, {}).get("candidates", [])
    gap_used = res.get(key, {}).get("gap_used")
    relaxed = None
    if len(cands) < 2:  # 예비까지 2명이 안 되면 0.2로 완화해 예비만 보충
        sj2 = os.path.join(WS, "09_seed_gap02.json")
        run(PY + [os.path.join(ROOT, "crystal", "recsys", "seed.py"), "--json", sj2, "--top", "20", "--gaps", "0.2"], "seed_gap0.2")
        relaxed = json.load(open(sj2, encoding="utf-8")).get(key, {}).get("candidates", [])
        ids = {c["user_id"] for c in cands}
        cands = cands + [c for c in relaxed if c["user_id"] not in ids]
    if not cands:
        raise RuntimeError("seed user: no candidate even at gap 0.2")
    return {"seed": cands[0], "backup": cands[1] if len(cands) > 1 else None, "gap_used": gap_used, "n_candidates_tier": len(res.get(key, {}).get("candidates", [])),
            "n_candidates_relaxed": len(relaxed) if relaxed is not None else None, "table": [l for l in lines if l.startswith("|") or l.startswith("###") or l.startswith("조건")]}


def s6_recsys():
    lines, sec = run(PY + [os.path.join(ROOT, "crystal", "recsys", "infer.py"), "--k", str(K_FEED), "--batch", "5000"], "infer")
    head = [l for l in lines if l.startswith("users ")]
    tail = [l for l in lines if l.startswith("wrote ")]
    n_feeds = sum(1 for _ in open(os.path.join(DATA, "feeds.jsonl"), encoding="utf-8"))
    return {"infer_head": head, "infer_tail": tail, "n_feeds": n_feeds, "seconds": sec, **pick_seed()}


def s7_rationale():
    users = [summary["step6"]["seed"]["user_id"]] + ([summary["step6"]["backup"]["user_id"]] if summary["step6"].get("backup") else [])
    cache = os.path.join(DATA, "rationale_cache.jsonl")
    sub = os.path.join(DONE, "step7_cache_archived")
    if not os.path.exists(sub):  # 기존 캐시는 단일 모델의 키워드·확률 기준이라 보관만 하고 새로 만든다
        if os.path.exists(cache):
            shutil.move(cache, os.path.join(WS, "09_rationale_cache_single_model.jsonl")); log("archived old rationale_cache.jsonl -> _workspace/09_rationale_cache_single_model.jsonl")
        open(sub, "w").write("1")
    lines, sec = run(PY + ["-m", "crystal.rationale.generate", "--users", ",".join(users), "--pages", "2"], "rationale_generate")
    txt = "\n".join(lines)
    i = txt.rfind('\n{\n "status"')
    res = json.loads(txt[i + 1:]) if i >= 0 else {"parse_error": lines[-3:]}
    n_cache = sum(1 for _ in open(cache, encoding="utf-8")) if os.path.exists(cache) else 0
    return {"users": users, "summary": res, "cache_lines": n_cache, "seconds": sec}


def http(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def s8_web():
    seed = summary["step6"]["seed"]["user_id"]
    dp = os.path.join(ROOT, "crystal", "web", "data.py")
    s = open(dp, encoding="utf-8").read()
    s2 = re.sub(r'REAL_SEED_USER = "\d+"', f'REAL_SEED_USER = "{seed}"', s)
    if s2 != s:
        open(dp, "w", encoding="utf-8").write(s2); log(f"data.py REAL_SEED_USER -> {seed}")
    shutil.rmtree(os.path.join(ROOT, "crystal", "web", "_cache"), ignore_errors=True)
    env = {**ENV, "CRYSTAL_DATA": "real", "CRYSTAL_SEED_USER": seed, "CRYSTAL_SHOW_LEVEL_CAPTION": "1"}
    with open(os.path.join(WS, "09_web_server.log"), "w", encoding="utf-8") as lf:
        p = subprocess.Popen(PY + ["-m", "uvicorn", "crystal.web.app:app", "--port", "8766"], cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            settings = None
            for _ in range(120):
                time.sleep(3)
                if p.poll() is not None:
                    raise RuntimeError("web server exited early")
                try:
                    settings = http("http://127.0.0.1:8766/api/settings")
                    break
                except Exception:
                    pass
            if settings is None:
                raise RuntimeError("web server did not answer /api/settings in 360s")
            pages = []
            for pg in range(1, 6):
                fd = http(f"http://127.0.0.1:8766/api/feed?user_id={seed}&page={pg}")
                pages.append({"page": pg, "n": len(fd["items"]), "fake": sum(i.get("final_label") == 1 for i in fd["items"]),
                              "probs": [i.get("pred_prob_fake") for i in fd["items"]], "stats": fd["stats"] if pg == 1 else None, "pages": fd["pages"], "total": fd["total"]})
            first_fake = next((i["article_id"] for i in http(f"http://127.0.0.1:8766/api/feed?user_id={seed}&page=1")["items"] if i.get("final_label") == 1), None)
            art = http(f"http://127.0.0.1:8766/api/article/{first_fake}") if first_fake else None
            cp = (art or {}).get("counterparts") or []
            cmp_ = http(f"http://127.0.0.1:8766/api/compare?a={first_fake}&b={cp[0]['article_id']}") if cp else None
            out = {"settings": {k: settings.get(k) for k in ("data_source", "n_articles", "n_users", "default_user", "show_level_caption")},
                   "feed_pages": pages, "first_fake": first_fake,
                   "article": {"prob": art.get("pred_prob_fake"), "keywords": len(art.get("keywords") or []), "rationales": len(art.get("rationales") or {}),
                               "verdict_status": art.get("verdict_status"), "counterparts": len(cp)} if art else None,
                   "compare": {"contrast_status": cmp_.get("contrast_status")} if cmp_ else None}
        finally:
            p.kill(); p.wait()
            log("web server (8766) stopped")
    return out


def s9_qa():
    from crystal.rationale.core import consistent
    rows = read_articles()
    by = {r["article_id"]: r for r in rows}
    qa = {}
    # 1 articles
    d = {"lines": len(rows), "dup_ids": len(rows) - len(by), "n_displayable": sum(map(disp, rows))}
    d["missing_pred_fields"] = sum(any(k not in r for k in ("pred_label", "pred_prob_fake", "final_label", "veracity_source", "factcheck", "fold")) for r in rows)
    d["split_not_test"] = sum(r.get("split") != "test" for r in rows)
    d["fold_out_of_range"] = sum(not (isinstance(r.get("fold"), int) and 0 <= r["fold"] <= 9) for r in rows)
    d["final_ne_pred"] = sum(r.get("final_label") != r.get("pred_label") for r in rows)
    d["prob_pred_mismatch"] = sum((r["pred_prob_fake"] >= 0.5) != (r["pred_label"] == 1) for r in rows)
    d["factcheck_not_null"] = sum(r.get("factcheck") is not None for r in rows)
    d["source_not_model"] = sum(r.get("veracity_source") != "model" for r in rows)
    d["keywords_field_missing"] = sum("keywords" not in r for r in rows)
    d["counterparts_on_displayable"] = sum("counterparts" in r for r in rows if disp(r))
    d["counterparts_on_non_displayable"] = sum("counterparts" in r for r in rows if not disp(r))
    d["pass"] = d["dup_ids"] == 0 and d["missing_pred_fields"] == 0 and d["split_not_test"] == 0 and d["fold_out_of_range"] == 0 and d["final_ne_pred"] == 0 \
        and d["prob_pred_mismatch"] == 0 and d["keywords_field_missing"] == 0 and d["counterparts_on_displayable"] == d["n_displayable"] and d["counterparts_on_non_displayable"] == 0
    qa["1_articles"] = d
    # 2 keywords
    d = {"n_keywords": 0, "pos_mismatch": 0, "phrase_not_in_sentence": 0, "overlap": 0, "over_10": 0, "score_out": 0, "zero_articles": 0, "max_score_not_1": 0}
    for r in rows:
        kws = r.get("keywords") or []
        if not kws:
            d["zero_articles"] += 1
            continue
        d["n_keywords"] += len(kws)
        d["over_10"] += len(kws) > 10
        d["max_score_not_1"] += abs(max(k["score"] for k in kws) - 1.0) > 1e-6
        for k in kws:
            d["pos_mismatch"] += (r["title"] if k["field"] == "title" else r["body"])[k["start"]:k["end"]] != k["phrase"]
            d["phrase_not_in_sentence"] += k["phrase"] not in (k.get("sentence") or "")
            d["score_out"] += not (0 <= k["score"] <= 1)
        for fld in ("title", "body"):
            spans = sorted((k["start"], k["end"]) for k in kws if k["field"] == fld)
            d["overlap"] += sum(spans[i + 1][0] < spans[i][1] for i in range(len(spans) - 1))
    d["pass"] = d["pos_mismatch"] == 0 and d["phrase_not_in_sentence"] == 0 and d["overlap"] == 0 and d["over_10"] == 0 and d["score_out"] == 0
    qa["2_keywords"] = d
    # 3 counterparts
    nt = lambda t: re.sub(r"[^0-9a-z가-힣]+", "", (t or "").lower())
    d = {"pairs": 0, "same_label": 0, "label_ne_row": 0, "sim_out": 0, "not_desc": 0, "self_ref": 0, "missing": 0, "cp_not_displayable": 0, "dup_title": 0, "over_3": 0,
         "fake_with_pair": 0, "real_with_pair": 0, "event_not_minus1": 0}
    for r in rows:
        cps = r.get("counterparts")
        if cps is None:
            continue
        d["pairs"] += len(cps)
        d["over_3"] += len(cps) > 3
        d["event_not_minus1"] += r.get("event_id") != -1
        if cps:
            d["fake_with_pair" if r["final_label"] == 1 else "real_with_pair"] += 1
        sims = [c["sim"] for c in cps]
        d["not_desc"] += sims != sorted(sims, reverse=True)
        titles = [nt(by[c["article_id"]]["title"]) for c in cps if c["article_id"] in by]
        d["dup_title"] += len(titles) - len(set(titles))
        for c in cps:
            b = by.get(c["article_id"])
            if b is None:
                d["missing"] += 1; continue
            d["same_label"] += c["final_label"] == r["final_label"]
            d["label_ne_row"] += c["final_label"] != b["final_label"]
            d["sim_out"] += not (0.8 <= c["sim"] < 1.0)
            d["self_ref"] += c["article_id"] == r["article_id"]
            d["cp_not_displayable"] += not disp(b)
    d["pass"] = all(d[k] == 0 for k in ("same_label", "label_ne_row", "sim_out", "not_desc", "self_ref", "missing", "cp_not_displayable", "dup_title", "over_3"))
    qa["3_counterparts"] = d
    # 4 feeds
    d = {"n_feeds": 0, "len_not_50": 0, "dup_in_feed": 0, "not_displayable": 0, "unknown": 0, "sampled_users": 0, "history_leak": 0}
    seeds = [str(summary.get("step6", {}).get("seed", {}).get("user_id")), str((summary.get("step6", {}).get("backup") or {}).get("user_id"))]
    seed_feed = {}
    with open(os.path.join(DATA, "feeds.jsonl"), encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            d["n_feeds"] += 1
            fd = x["feed"]
            d["len_not_50"] += len(fd) != K_FEED
            d["dup_in_feed"] += len(fd) != len(set(fd))
            for a in fd:
                if a not in by:
                    d["unknown"] += 1
                elif not disp(by[a]):
                    d["not_displayable"] += 1
            if x["user_id"] in seeds:
                seed_feed[x["user_id"]] = fd
    n_users, hist = 0, {}
    with open(os.path.join(DATA, "users.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            n_users += 1
            if i % 300 == 0 or any(f'"user_id": "{s}"' in line[:60] for s in seeds):
                u = json.loads(line)
                hist[u["user_id"]] = {h["article_id"] for h in u["history"]}
    d["sampled_users"] = len(hist)
    with open(os.path.join(DATA, "feeds.jsonl"), encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x["user_id"] in hist:
                d["history_leak"] += len(hist[x["user_id"]] & set(x["feed"]))
    d["n_users"] = n_users
    d["seed_first24"] = {u: {"unique": len(set(fd[:24])) == 24, "displayable": all(disp(by[a]) for a in fd[:24] if a in by),
                             "fake_per_page": [sum(by[a]["final_label"] == 1 for a in fd[k * PAGE:(k + 1) * PAGE] if a in by) for k in range(5)]} for u, fd in seed_feed.items()}
    d["pass"] = d["n_feeds"] == n_users and d["len_not_50"] == 0 and d["dup_in_feed"] == 0 and d["not_displayable"] == 0 and d["unknown"] == 0 and d["history_leak"] == 0
    qa["4_feeds"] = d
    # 5 cache + 6 consistency
    cp = os.path.join(DATA, "rationale_cache.jsonl")
    d = {"lines": 0, "bad_format": 0, "kind": Counter(), "text_mismatch": 0, "unknown_article": 0, "key_not_keyword": 0, "verdict_inconsistent": 0,
         "keyword_inconsistent": 0, "contrast_same_label": 0, "leak_patterns": 0, "model_mention": 0}
    leak = re.compile(r"AIza|sk-[A-Za-z0-9]|api key|GEMINI|Bearer|\.env", re.I)
    model_re = re.compile(r"classifier|the model|probability|confidence", re.I)
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                d["lines"] += 1
                if any(k not in x for k in ("article_id", "kind", "key", "text", "model", "prompt_version")) or x["kind"] not in ("keyword", "verdict", "contrast"):
                    d["bad_format"] += 1; continue
                d["kind"][x["kind"]] += 1
                a = by.get(x["article_id"])
                if a is None:
                    d["unknown_article"] += 1; continue
                t = x["text"]
                if x["kind"] == "keyword":
                    d["text_mismatch"] += (a.get("rationales") or {}).get(x["key"]) != t
                    d["key_not_keyword"] += x["key"] not in {k["phrase"] for k in a.get("keywords") or []}
                    d["keyword_inconsistent"] += not consistent(t, a["final_label"])
                elif x["kind"] == "verdict":
                    d["text_mismatch"] += a.get("verdict_rationale") != t
                    d["verdict_inconsistent"] += not consistent(t, a["final_label"])
                else:
                    b = by.get(x["key"])
                    obj = json.loads(t) if isinstance(t, str) else t
                    d["text_mismatch"] += not any(c["article_id"] == x["key"] and c.get("claim_contrast") == obj for c in a.get("counterparts") or [])
                    d["contrast_same_label"] += (b is None) or (b["final_label"] == a["final_label"])
                    t = json.dumps(obj)
                d["leak_patterns"] += bool(leak.search(t))
                d["model_mention"] += bool(model_re.search(t))
    d["kind"] = dict(d["kind"])
    d["articles_with_rationales"] = sum(bool(r.get("rationales")) for r in rows)
    d["articles_with_verdict"] = sum(bool(r.get("verdict_rationale")) for r in rows)
    d["pairs_with_contrast"] = sum(bool(c.get("claim_contrast")) for r in rows for c in r.get("counterparts") or [])
    d["pass"] = d["bad_format"] == 0 and d["text_mismatch"] == 0 and d["key_not_keyword"] == 0 and d["verdict_inconsistent"] == 0 and d["keyword_inconsistent"] == 0 \
        and d["contrast_same_label"] == 0 and d["leak_patterns"] == 0
    qa["5_6_cache_consistency"] = d
    qa["all_pass"] = all(v["pass"] for v in qa.values() if isinstance(v, dict))
    with open(os.path.join(WS, "09_qa.json"), "w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=1)
    return qa


# ---------- 보고서 ----------
def fmt(x):
    return f"{x:,}" if isinstance(x, int) else (f"{x:.4f}" if isinstance(x, float) else str(x))


def write_report(failure):
    S = summary
    L = ["# 09 K-fold 교차 예측 전환 보고 (오케스트레이터 보조, 2026-10-21)", "",
         "사용자 결정: 추천 후보를 test 10%로 제한하면 순위가 왜곡되므로 10-fold 교차 예측으로 전건 out-of-fold 판정을 만들고 후보를 전체(is_displayable) 기사로 연다. "
         "본문 없는 기사는 데이터(학습·히스토리)에 남기고 화면·추천·짝에서만 뺀다(`crystal/common.is_displayable`).",
         "드라이버 `crystal/run_kfold.py`, 로그 `_workspace/09_kfold.log`, 수치 `_workspace/09_summary.json`. 실행 전부 `KMP_DUPLICATE_LIB_OK=TRUE`, `python -X utf8`.", ""]
    done = S.get("last_completed_step", 0)
    L += [f"## 상태: {'전 단계 완료' if done >= 9 and not failure else f'단계 {done}까지 완료, 이후 실패'}", ""]
    if failure:
        L += ["### 실패", "```", failure.strip(), "```", ""]
    if "step1" in S:
        s = S["step1"]
        lab = {str(k): v for k, v in s["label"].items()}
        L += ["## 1. 데이터 준비 (`crystal/prepare_data.py`)",
              f"- 제거는 중복 news_id만. `articles.jsonl` {fmt(s['n_articles'])}건 (gossip {fmt(s['dataset'].get('gossip', 0))} / pol {fmt(s['dataset'].get('pol', 0))}; label 0: {fmt(lab.get('0', 0))} / 1: {fmt(lab.get('1', 0))}), "
              f"is_displayable {fmt(s['n_displayable'])}건 (비표시 {fmt(s['n_articles'] - s['n_displayable'])}건은 데이터에 남김). `users.jsonl` {fmt(s['n_users'])}명.",
              f"- prepare 출력: {' / '.join(s['stdout'])}", ""]
    if "step2" in S:
        r = S["step2"]["results"]
        p = r["pooled"]
        L += ["## 2. 10-fold 학습 (`roberta_veracity/train_kfold.py`)",
              "- fold i: test = 묶음 i, val = 묶음 (i+1)%10, train = 나머지 8묶음 ((dataset, label) 층화, seed 42). 설정은 train.py와 동일.",
              f"- **전체 pooled {fmt(p['all']['n'])}건: acc {p['all']['acc']:.4f}, macro F1 {p['all']['macro_f1']:.4f}, fake F1 {p['all']['f1_fake']:.4f}** (기존 단일 모델 test 1,813건 acc 0.8753 / macro F1 0.8294 / fake F1 0.7408). "
              f"TP {p['all']['tp']} / FP {p['all']['fp']} / FN {p['all']['fn']} / TN {p['all']['tn']}.",
              f"- gossip {fmt(p['gossip']['n'])}건 acc {p['gossip']['acc']:.4f} macro F1 {p['gossip']['macro_f1']:.4f}; pol {fmt(p['pol']['n'])}건 acc {p['pol']['acc']:.4f} macro F1 {p['pol']['macro_f1']:.4f}.",
              f"- fold acc 평균 {r['mean_fold_acc']:.4f} (표준편차 {r['std_fold_acc']:.4f}), 총 학습 시간 {r['train_seconds_total']}초 ({r['train_seconds_total'] / 3600:.2f}h).", "",
              "| fold | n_test | best epoch | val macro F1 | test acc | test macro F1 | fake F1 | T | val ECE 전→후 | 초 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for f in r["folds"]:
            L.append(f"| {f['fold']} | {f['n_test']} | {f['best_epoch']} | {f['val_macro_f1']:.4f} | {f['acc']:.4f} | {f['macro_f1']:.4f} | {f['f1_fake']:.4f} | {f['T']:.4f} | {f['val_ece_before']:.4f} → {f['val_ece_after']:.4f} | {f['seconds']} |")
        L.append("")
    if "step3" in S:
        s = S["step3"]
        L += ["## 3. 진위 판정 반영 (`crystal/veracity/predict.py`)",
              f"- `kfold_predictions.csv`를 읽어 전건 {fmt(s['n'])}건에 pred_label·pred_prob_fake(자기 fold T 보정)·final_label·fold 부여, split 전건 \"test\": {s['all_split_test']}. pooled acc {s['pooled_acc']:.4f} (gossip {s['by_dataset']['gossip']['acc']:.4f}, pol {s['by_dataset']['pol']['acc']:.4f}).",
              f"- 판정 fake {fmt(s['pred_fake'])} (정답 fake {fmt(s['label_fake'])}). fold별 건수 {s['fold_counts']}.",
              f"- `pred_prob_fake` 0.05~0.95 구간 {fmt(s['prob_mid_005_095'])}/{fmt(s['n'])} ({s['prob_mid_ratio'] * 100:.1f}%; 08 보고 test 1,531건 기준 69.2%). 0.1 구간 히스토그램 {s['prob_hist_0.1']}.", ""]
    if "step4" in S:
        s = S["step4"]
        L += ["## 4. 키워드 (`crystal/keyword/run.py`, fold별 자기 모델로 IG)",
              f"- 기존 `ig_raw.jsonl`(단일 모델) 폐기 후 전건 재계산: {s['ig_header']}, ig_raw {fmt(s['n_ig_raw'])}줄, 소요 {s['seconds']}초 ({s['seconds'] / 3600:.2f}h, IG+확장+쓰기).",
              f"- keywords 필드 {fmt(s['n_articles_with_field'])}건, 키워드 {fmt(s['n_keywords'])}개 (제목 {fmt(s['title'])} / 본문 {fmt(s['body'])}), 0개 기사 {fmt(s['n_zero'])}. 개수 분포 {s['dist']}.", ""]
    if "step5" in S:
        st = S["step5"]["stats"]
        L += ["## 5. 사건 짝 (`match.py --pool all --method cos`, 짝 대상·후보 모두 is_displayable)",
              f"- 대상 {fmt(st['n_test'])}건 (fake {fmt(st['n_test_fake'])} / real {fmt(st['n_test_real'])}). 짝 있는 fake {fmt(st['fake_with_pair'])} ({st['fake_with_pair_ratio'] * 100:.1f}%), real {fmt(st['real_with_pair'])} ({st['real_with_pair_ratio'] * 100:.1f}%). "
              f"**쌍 {fmt(st['pairs_total'])}** (08: 151, test 1,531건 대상), 짝 수 0/1/2/3 {st['n_counterparts_hist']}, sim 분위 {st['sim_quantiles']}, 평균 {st['sim_mean']}. counterparts 필드 {fmt(S['step5']['n_with_counterparts_field'])}건.", ""]
    if "step6" in S:
        s = S["step6"]
        sd, bk = s["seed"], s.get("backup")
        L += ["## 6. 피드·시드 (`infer.py` K=50, 후보 = is_displayable 전체; `seed.py`)",
              f"- {' / '.join(s['infer_head'] + s['infer_tail'])}. `feeds.jsonl` {fmt(s['n_feeds'])}줄.",
              f"- 시드 조건 충족 후보(전체 풀 짝, 차 ≥{s['gap_used']}) {s['n_candidates_tier']}명" + (f", 0.2 완화 시 {s['n_candidates_relaxed']}명" if s.get('n_candidates_relaxed') is not None else "") + ".",
              f"- **시드 주 {sd['user_id']}**: 첫 12건 FAKE {sd['feed_fake_first_page']}, 1~5페이지 FAKE {sd['pages_fake'][:5]}, 짝 있는 FAKE {sd['fake_with_pair']}, 7일 {sd['weekly']['n_true']}·{sd['weekly']['n_fake']}/{sd['weekly']['total']} ({sd['weekly']['fake']:.2f}), 전체 {sd['overall']['n_true']}·{sd['overall']['n_fake']}/{sd['overall']['total']} ({sd['overall']['fake']:.2f}), 차 {sd['gap']:.2f}.",
              (f"- **시드 예비 {bk['user_id']}**: 첫 12건 FAKE {bk['feed_fake_first_page']}, 1~5페이지 FAKE {bk['pages_fake'][:5]}, 짝 있는 FAKE {bk['fake_with_pair']}, 7일 {bk['weekly']['n_true']}·{bk['weekly']['n_fake']}/{bk['weekly']['total']}, 전체 {bk['overall']['n_true']}·{bk['overall']['n_fake']}/{bk['overall']['total']}, 차 {bk['gap']:.2f}." if bk else "- 예비 없음."),
              "", "seed.py 출력:", ""] + s["table"] + [""]
    if "step7" in S:
        s, r = S["step7"], S["step7"]["summary"]
        L += ["## 7. 설명 (`crystal.rationale.generate --pages 2`, `gemini-3.5-flash-lite`)",
              f"- 기존 캐시는 단일 모델의 키워드·확률·판정 기준이라 `_workspace/09_rationale_cache_single_model.jsonl`로 보관하고 새로 생성. 대상 유저 {s['users']}.",
              f"- 대상 피드 기사 {r.get('feed_articles')} + 짝 기사 {r.get('counterpart_articles')}, 쌍 {r.get('pairs')}. **Gemini 호출 {r.get('calls')}회** (모델별 {r.get('calls_by_model')}, 오류 {r.get('api_errors')}, 429 {r.get('quota_429')}, 코드 {r.get('error_codes')}), 생성 단위 {r.get('generated_units')}, 버림 {r.get('discarded')}, {r.get('seconds')}초, 상태 {r.get('status')}.",
              f"- 캐시 {fmt(s['cache_lines'])}줄, articles 반영 {r.get('articles_written')}건. 누락 verdict {r.get('missing_verdict')}, 누락 키워드 설명 {r.get('missing_keyword_rationales')}.", ""]
    if "step8" in S:
        s = S["step8"]
        L += ["## 8. 웹 (8766 포트로 확인 후 종료, `_cache/` 삭제, `REAL_SEED_USER` 갱신)",
              f"- `/api/settings`: {s['settings']}.",
              "- `/api/feed` 페이지별 FAKE 수: " + ", ".join(f"p{p['page']} {p['fake']}/{p['n']}" for p in s["feed_pages"]) + f" (pages {s['feed_pages'][0]['pages']}, total {s['feed_pages'][0]['total']}). 1페이지 사이드바 {s['feed_pages'][0]['stats']}.",
              f"- 1페이지 확률(pred_prob_fake) {s['feed_pages'][0]['probs']}.",
              f"- `/api/article/{s['first_fake']}`: {s['article']}; compare: {s['compare']}.", ""]
    if "step9" in S:
        q = S["step9"]
        L += ["## 9. QA (07의 1~6 항목을 새 전제로)", "| # | 항목 | 결과 | 수치 |", "|---|---|---|---|"]
        for k, v in q.items():
            if isinstance(v, dict):
                L.append(f"| {k} | {'통과' if v['pass'] else '**실패**'} | {json.dumps({a: b for a, b in v.items() if a != 'pass'}, ensure_ascii=False)} |")
        L += [f"전체 통과: {q['all_pass']}", ""]
    L += ["## 코드 변경",
          "- 신규 `crystal/common.py`(is_boilerplate, is_displayable), `roberta_veracity/train_kfold.py`, `crystal/run_kfold.py`.",
          "- `prepare_data.py` 제거 규칙 삭제(중복 id만), split 전건 \"test\". `veracity/loader.py` `load(fold)`, `predict.py` kfold csv 읽기 + fold 필드, `calibrate.py` `fit_temperature` 분리.",
          "- `keyword/run.py` 전건·fold별 모델, `event/match.py` is_displayable 대상·후보, `recsys/infer.py` 후보 displayable 전체, `dataset.py` N_ARTICLES 18124, `seed.py` 페이지별 FAKE·`--json`, `rationale/generate.py` EXPECT 갱신·429 기사 건너뜀, `web/data.py` pool displayable·REAL_SEED_USER.",
          "- 데이터 계약 문서에 `fold` 필드와 split 설명 추가.", "",
          "## 남은 문제",
          "1. `_workspace/03_keyword/article_*.json`이 전건(18,124) 생성되어 폴더가 큼.",
          "2. `match.py`가 `boilerplate_ids.txt`를 실행마다 덮어씀(지금은 보일러플레이트가 데이터에 있어 34건이 다시 기록됨).",
          "3. 기존 단일 모델 `roberta_veracity/best`·`split.csv`·`test_results.json`·`crystal/data/veracity/temperature.json`은 참고용으로 남아 있음(파이프라인은 더 이상 쓰지 않음).",
          "4. 기존 설명 캐시는 보관만 했고(`_workspace/09_rationale_cache_single_model.jsonl`) 새 캐시는 시드 두 명 2페이지분만 있음.",
          "5. 01~08 보고서·스킬 문서의 수치(15,524 / 1,531 / 0.8707 등)는 이 보고 기준으로 갱신 필요."]
    if failure:
        L.append("6. 위 '실패' 단계부터 다시 실행하면 완료 표시가 있는 단계는 건너뛴다 (`_workspace/09_done/`).")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log(f"report written: {REPORT}")


def write_claude_md(failure):
    p = os.path.join(ROOT, "CLAUDE.md")
    s = open(p, encoding="utf-8").read()
    if "| 2026-10-21 | 10-fold 교차 예측" in s:
        return
    r = summary.get("step2", {}).get("results", {}).get("pooled", {}).get("all", {})
    acc = f"pooled acc {r['acc']:.4f} / macro F1 {r['macro_f1']:.4f}" if r else "학습 미완"
    st = "전 단계 완료" if summary.get("last_completed_step", 0) >= 9 and not failure else f"단계 {summary.get('last_completed_step', 0)}까지 완료 후 실패 (보고서 참조)"
    seed = summary.get("step6", {}).get("seed", {}).get("user_id", "-")
    line = (f"| 2026-10-21 | 10-fold 교차 예측으로 전환: 본문 없는 기사 유지(18,124건, 중복 id 2건만 제거), fold별 8:1:1 학습·온도 보정(`train_kfold.py`), 전건 out-of-fold 판정(split 전건 test + `fold` 필드), "
            f"화면·추천·짝은 `common.is_displayable` 기사만, 추천 후보 전체로 확장, IG 전건 fold별 모델로 재계산, 캐시 재생성, 시드 {seed}. {acc}. {st}. 드라이버 `crystal/run_kfold.py`, 보고 `_workspace/09_kfold_report.md` "
            f"| prepare_data, common(신규), train_kfold(신규), veracity(loader·predict·calibrate), keyword/run, event/match, recsys(infer·dataset·seed), rationale/generate, web/data, data-contract "
            f"| 사용자 결정 10/21: test 10% 후보 제한이 추천 순위를 왜곡 |\n")
    open(p, "w", encoding="utf-8").write(s.rstrip("\n") + "\n" + line)
    log("CLAUDE.md updated")


def main():
    failure = None
    log("driver start")
    try:
        step(1, "prepare", s1_prepare)
        step(2, "train_kfold", s2_train)
        step(3, "predict", s3_predict)
        step(4, "keyword", s4_keyword)
        step(5, "match", s5_match)
        step(6, "recsys+seed", s6_recsys)
        step(7, "rationale", s7_rationale)
        step(8, "web", s8_web)
        step(9, "qa", s9_qa)
    except BaseException:
        failure = traceback.format_exc()
        summary["failure"] = failure
        save()
        log("FAILED:\n" + failure)
    finally:
        try:
            write_report(failure)
            write_claude_md(failure)
        except BaseException:
            log("report/CLAUDE.md writing failed:\n" + traceback.format_exc())
    log("driver end" + (" (failed)" if failure else " (all steps done)"))
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
