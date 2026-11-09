"""CRYSTAL 웹: 피드·기사·비교·검색·설명 API와 정적 화면. 실행: python -X utf8 -m uvicorn crystal.web.app:app"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data as D

STATIC = os.path.join(D.HERE, "static")
SETTINGS = {"show_feed_labels": os.environ.get("CRYSTAL_SHOW_FEED_LABELS", "0") != "0",
            "show_level_caption": os.environ.get("CRYSTAL_SHOW_LEVEL_CAPTION", "0") != "0", "page_size": 12}
GOALS = {"feed": "지금 받은 피드와 지난 소비의 진위 분포를 사용자가 인식하게 한다.",
         "article": "이 기사가 왜 가짜/진짜로 판정됐는지 근거(판정에 가장 기여한 구문과 설명)를 보여 준다.",
         "event": "같은 사건을 다룬 반대 진위 기사를 나란히 보여 교정한다."}
store = D.load_store()
app = FastAPI(title="CRYSTAL")


def prob_of(a):
    """사람에게 보여주는 확률: 판정된 라벨의 확률 (FAKE면 pred_prob_fake, TRUE면 1-pred_prob_fake). 2026-11-03 사용자 요청."""
    p = a.get("pred_prob_fake")
    if p is None:
        return None
    return round(p if a.get("final_label") == 1 else 1 - p, 4)


def card(a):
    return {"article_id": a["article_id"], "title": a.get("title", ""), "final_label": a.get("final_label"),
            "pred_prob_fake": a.get("pred_prob_fake"), "prob": prob_of(a)}


def get_article(aid):
    a = store.articles.get(aid)
    if a is None:
        raise HTTPException(404, f"article {aid} not found")
    return a


def article_payload(a):
    out = {k: a.get(k) for k in ("article_id", "dataset", "title", "body", "final_label", "pred_prob_fake",
                                 "veracity_source", "factcheck", "keywords", "rationales", "verdict_rationale", "event_id")}
    out["body"] = out["body"] or ""
    out["keywords"] = out["keywords"] or []
    out["rationales"] = out["rationales"] or {}
    out["event_id"] = -1 if out["event_id"] is None else out["event_id"]
    out["prob"] = prob_of(a)
    out["keyword_errors"] = []
    for k in out["keywords"]:
        text = out["title"] if k.get("field") == "title" else out["body"]
        actual = text[k["start"]:k["end"]]
        if actual != k["phrase"]:
            out["keyword_errors"].append({"phrase": k["phrase"], "field": k.get("field"), "start": k["start"], "end": k["end"], "actual": actual})
    cps = []
    for c in a.get("counterparts") or []:
        b = store.articles.get(c["article_id"])
        cps.append({**c, "title": b.get("title", "") if b else "", "prob": prob_of(b) if b else None,
                    "pred_prob_fake": b.get("pred_prob_fake") if b else None})
    out["counterparts"] = cps
    if not out["verdict_rationale"]:
        t, out["verdict_status"] = store.lookup(a["article_id"], "verdict", "")
        out["verdict_rationale"] = t or D.PLACEHOLDER
    else:
        out["verdict_status"] = "article"
    return out


@app.get("/api/settings")
def settings():
    return {**SETTINGS, "goals": GOALS, "data_source": store.source, "data_reason": store.reason,
            "default_user": store.default_user(), "n_articles": len(store.pool), "n_users": len(store.feed_off)}


@app.get("/api/feed")
def feed(user_id: str = "", page: int = 1):
    uid = user_id or store.default_user()
    ids = store.feed(uid)
    if ids is None:
        raise HTTPException(404, f"user {uid} not found")
    size = SETTINGS["page_size"]
    pages = max(1, (len(ids) + size - 1) // size)
    page = min(max(page, 1), pages)
    page_ids = [i for i in ids[(page - 1) * size:page * size] if i in store.articles]
    return {"user_id": uid, "page": page, "pages": pages, "page_size": size, "total": len(ids),
            "show_feed_labels": SETTINGS["show_feed_labels"], "goal": GOALS["feed"],
            "items": [card(store.articles[i]) for i in page_ids], "stats": store.stats(uid, page_ids)}


@app.get("/api/article/{aid}")
def article(aid: str):
    return {**article_payload(get_article(aid)), "goal": GOALS["article"]}


@app.get("/api/compare")
def compare(a: str, b: str):
    pa, pb = article_payload(get_article(a)), article_payload(get_article(b))
    contrast, status = None, "article"
    for src, other in ((pa, b), (pb, a)):
        for c in src["counterparts"]:
            if c["article_id"] == other and c.get("claim_contrast"):
                contrast = c["claim_contrast"]
                break
        if contrast:
            break
    if not contrast:
        contrast, status = store.lookup(a, "contrast", b)
        contrast = contrast or {"this": D.PLACEHOLDER, "related": D.PLACEHOLDER}
    return {"a": pa, "b": pb, "claim_contrast": contrast, "contrast_status": status, "goal": GOALS["event"]}


@app.get("/api/search")
def search(q: str = "", limit: int = 20):
    q = q.strip().lower()
    hits = [card(x) for x in store.pool if q and q in (x.get("title") or "").lower()]
    return {"q": q, "items": hits[:limit], "total": len(hits)}


class RationaleReq(BaseModel):
    article_id: str
    kind: str = "keyword"
    key: str = ""


@app.post("/api/rationale")
def rationale(req: RationaleReq):
    get_article(req.article_id)
    text, status = store.get_rationale(req.article_id, req.kind, req.key)
    return {"article_id": req.article_id, "kind": req.kind, "key": req.key, "text": text, "status": status}


class ClickReq(BaseModel):
    user_id: str
    article_id: str


@app.post("/api/log-click")
def log_click(req: ClickReq):
    store.log_click(req.user_id, req.article_id)
    return {"ok": True}


@app.get("/api/demo-users")
def demo_users():
    return {"users": store.demo_users(), "default_user": store.default_user()}


class ResetReq(BaseModel):
    user_id: str


@app.post("/api/reset-user")
def reset_user(req: ResetReq):
    store.reset_user(req.user_id)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/article/{aid}")
def article_page(aid: str):
    return FileResponse(os.path.join(STATIC, "article.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
