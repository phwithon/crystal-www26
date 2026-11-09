"""crystal/data가 갖춰져 있으면 실데이터, 아니면 crystal/web/sample. 유저·피드는 줄 오프셋으로 찾는다."""
import hashlib
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REAL_DIR = os.path.join(ROOT, "crystal", "data")
SAMPLE_DIR = os.path.join(HERE, "sample")
CACHE_DIR = os.path.join(HERE, "_cache")
PLACEHOLDER = "Analyzing context..."
# 2026-11-03 CRYSTAL_KEYWORDS=lime: crystal/data/keyword/lime_keywords.jsonl에 있는 기사는 그 키워드로 덮어쓰고,
# 키워드·판정 설명은 rationale_cache_lime.jsonl만 쓴다 (IG 구문 기준으로 만든 설명과 섞이지 않게). 기본 "ig"는 기존 동작 그대로.
KEYWORDS_MODE = os.environ.get("CRYSTAL_KEYWORDS", "ig")
REAL_SEED_USER = "25376730"  # 2026-10-21 CROWN 피드 기준 재선정 (_workspace/11_seed.json 1순위)
_UID = re.compile(rb'"user_id":\s*"([^"]+)"')
# 2026-10-30 데모 유저 (전체 히스토리 30건 이상, 최근 7일 3건 이상 중 가짜 비율 75% / 진짜 비율 75%)
DEMO_USERS = [
    {"user_id": "620481545", "name": "Fake news reader"},  # 65건 중 가짜 75%
    {"user_id": "2361665978", "name": "True news reader"},  # 80건 중 진짜 75%
]

try:
    from crystal.common import is_displayable
except Exception:
    def is_displayable(title, body):
        b = (body or "").strip()
        return b != "" and b != (title or "").strip() and not (b.startswith(".css") or "{display:" in b[:300])
try:
    from crystal.recsys.stats import label_dist, history_dist
except Exception:
    def label_dist(article_ids, articles_by_id, label_field="final_label"):
        labels = [articles_by_id[a][label_field] for a in article_ids
                  if a in articles_by_id and articles_by_id[a].get(label_field) is not None]
        n_fake = sum(1 for l in labels if l == 1)
        total = len(labels)
        return {"true": (total - n_fake) / total if total else 0.0, "fake": n_fake / total if total else 0.0,
                "n_true": total - n_fake, "n_fake": n_fake, "total": total}

    def history_dist(history, articles_by_id, days=None, label_field="final_label"):
        if days is not None and history:
            cut = history[-1]["time"] - days * 86400
            history = [h for h in history if h["time"] >= cut]
        return label_dist(list(dict.fromkeys(h["article_id"] for h in history)), articles_by_id, label_field)


def _line_index(path):
    st = os.stat(path)
    cpath = os.path.join(CACHE_DIR, hashlib.md5(path.encode("utf-8")).hexdigest()[:10] + "-" + os.path.basename(path) + ".idx.json")
    if st.st_size > 5_000_000 and os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            c = json.load(f)
        if c["size"] == st.st_size and c["mtime"] == st.st_mtime:
            return c["offsets"]
    offsets, pos = {}, 0
    with open(path, "rb") as f:
        for line in f:
            m = _UID.search(line[:256])
            if m:
                offsets[m.group(1).decode("utf-8")] = pos
            pos += len(line)
    if st.st_size > 5_000_000:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump({"size": st.st_size, "mtime": st.st_mtime, "offsets": offsets}, f)
    return offsets


def _real_ready():
    for name in ("articles.jsonl", "users.jsonl", "feeds.jsonl"):
        if not os.path.exists(os.path.join(REAL_DIR, name)):
            return False, f"{name} 없음"
    with open(os.path.join(REAL_DIR, "articles.jsonl"), encoding="utf-8") as f:
        for line in f:
            if '"split": "test"' in line and '"keywords": [{' in line:
                return True, "test 기사에 keywords 있음"
    return False, "test 기사에 keywords 없음"


class Store:
    def __init__(self, data_dir, source, reason):
        self.dir, self.source, self.reason = data_dir, source, reason
        self.articles = {}
        with open(os.path.join(data_dir, "articles.jsonl"), encoding="utf-8") as f:
            for line in f:
                a = json.loads(line)
                self.articles[a["article_id"]] = a
        self.lime_ids = set()
        lp = os.path.join(data_dir, "keyword", {"lime": "lime_keywords.jsonl", "lime_masked": "lime_masked_keywords.jsonl", "ig_masked": "ig_masked_keywords.jsonl", "ig_sent": "ig_sent_keywords.jsonl", "cc": "cc_keywords.jsonl"}.get(KEYWORDS_MODE, ""))
        if KEYWORDS_MODE in ("lime", "lime_masked", "ig_masked", "ig_sent", "cc") and os.path.exists(lp):
            with open(lp, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["article_id"] in self.articles:
                        self.articles[r["article_id"]]["keywords"] = r["keywords"]
                        self.lime_ids.add(r["article_id"])
        self.pool = [a for a in self.articles.values() if a.get("split", "test") == "test" and is_displayable(a.get("title"), a.get("body"))]
        self.users_path = os.path.join(data_dir, "users.jsonl")
        self.feeds_path = os.path.join(data_dir, "feeds.jsonl")
        self.user_off = _line_index(self.users_path)
        self.feed_off = _line_index(self.feeds_path)
        self.cache, self.cache_lime = {}, {}
        lime_cache_name = "rationale_cache_lime_masked.jsonl" if KEYWORDS_MODE in ("lime_masked", "ig_masked", "ig_sent", "cc") else "rationale_cache_lime.jsonl"  # ig_masked도 같은 캐시(구문 키 공유)
        for name, dst in (("rationale_cache.jsonl", self.cache), (lime_cache_name, self.cache_lime)):
            p = os.path.join(data_dir, name)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        dst[(r["article_id"], r["kind"], r["key"])] = r["text"]
        self.clicks = {}
        self.clicks_path = os.path.join(data_dir, "clicks.jsonl")
        if os.path.exists(self.clicks_path):
            with open(self.clicks_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    self.clicks.setdefault(r["user_id"], []).append({"article_id": r["article_id"], "time": r["time"]})

    @staticmethod
    def _read(path, off):
        with open(path, "rb") as f:
            f.seek(off)
            return json.loads(f.readline().decode("utf-8"))

    def default_user(self):
        env = os.environ.get("CRYSTAL_SEED_USER")
        for uid in (env, DEMO_USERS[0]["user_id"], REAL_SEED_USER):
            if uid and uid in self.feed_off:
                return uid
        return next(iter(self.feed_off), None)

    def history(self, uid):
        if uid not in self.user_off:
            return None
        return self._read(self.users_path, self.user_off[uid])["history"] + self.clicks.get(uid, [])

    _NAME_STOP = {"the", "a", "an", "is", "are", "was", "this", "that", "what", "why", "how", "here", "new", "watch",
                  "see", "inside", "every", "all", "signs", "report", "reports", "exclusive", "video", "photo", "photos"}

    def _entity_key(self, aid):
        """기사 제목에 나오는 인물성 토큰 집합 (대문자 단어). 다양화 용도라 근사치면 충분하다."""
        import re as _re
        title = self.articles.get(aid, {}).get("title") or ""
        toks = {w for w in _re.findall(r"[A-Z][a-z]+", title) if w.lower() not in self._NAME_STOP}
        return toks or {aid}

    def feed(self, uid):
        """미리 계산한 추천 목록에서 데모 중 읽은 기사를 뺀 뒤, 같은 인물이 12건 페이지당 2건을 넘지 않게
        그리디로 재배열한다 (2026-11-03: 데모 피드가 한 인물로 도배되는 문제). 순위는 CROWN 순서를 유지한 채 미룰 뿐이다."""
        if uid not in self.feed_off:
            return None
        # 2026-11-03: 피드 순서 고정. 읽은 기사도 빼지 않는다 (클릭은 log_click으로 히스토리·분포 사이드바에만 반영)
        read = set()
        base = [i for i in self._read(self.feeds_path, self.feed_off[uid])["feed"] if i not in read]
        # 2026-11-03 데모 1페이지 큐레이션: demo_front.json의 기사들을 맨 앞에 고정 (품질 좋은 설명·키워드 + 라벨 혼합)
        fp = os.path.join(self.dir, "keyword", "demo_front.json")
        if os.path.exists(fp):
            try:
                with open(fp, encoding="utf-8") as _f:
                    _front = json.load(_f).get(uid) or []
            except ValueError:
                _front = []
            fr = [i for i in _front if i in self.articles and i not in read]
            _dem = set((json.load(open(fp, encoding="utf-8")).get("_demote") or {}).get(uid) or [])
            if _dem:  # 2026-11-03: 사용자 지시로 뒤로 뺀 기사는 맨 끝(10페이지 밖)으로
                base = [i for i in base if i not in _dem] + [i for i in base if i in _dem]
                self._dem_tail = [i for i in base if i in _dem]
            _dem = set((_demote_all := (json.load(open(fp, encoding="utf-8")).get("_demote") or {})).get(uid) or [])
            if _dem:  # 2026-11-03: 사용자 지시로 뒤로 뺀 기사 — 맨 끝으로 보내 상한(10페이지) 밖으로
                base = [i for i in base if i not in _dem] + [i for i in base if i in _dem]
                self._dem_tail = [i for i in base if i in _dem]
            if fr:  # 큐레이션 12건은 그대로 1페이지로 쓰고, 나머지만 다양화 그리디를 태운다 (제목 중복은 큐레이션 쪽만 남김)
                ft = {(self.articles[i].get("title") or "").strip().lower() for i in fr}
                rest_base = [i for i in base if i not in set(fr)
                             and (self.articles.get(i, {}).get("title") or "").strip().lower() not in ft]
                _dt = getattr(self, "_dem_tail", [])
                _out = fr + [i for i in self._diversify(rest_base) if i not in _dt] + _dt
                return _out[:self.MAX_FEED]
            base = fr + [i for i in base if i not in set(fr)]
        return self._diversify(base)[:self.MAX_FEED]

    MAX_FEED = 120  # 2026-11-03: 피드는 10페이지(12개×10)까지만
    MIN_FEED_WORDS = 260  # 2026-11-03: 짧은 기사는 피드 뒤로 보냄 (첫 화면 품질)

    def _diversify(self, base):
        short = [i for i in base if len((self.articles.get(i, {}).get("body") or "").split()) < self.MIN_FEED_WORDS]
        if short:
            short_set = set(short)
            base = [i for i in base if i not in short_set] + short
        seen_titles, dedup = set(), []
        for i in base:  # 제목이 같은 중복 기사(수집본에 동일 기사 복수 id)는 처음 것만
            t = (self.articles.get(i, {}).get("title") or i).strip().lower()
            if t in seen_titles:
                continue
            seen_titles.add(t)
            dedup.append(i)
        base = dedup
        out, deferred, count, page = [], [], {}, 12
        queue = list(base)
        while queue or deferred:
            if len(out) % page == 0:
                count = {}
                queue = deferred + queue  # 새 페이지에서는 미뤄둔 것 우선 복귀
                deferred = []
            if not queue:
                break
            aid = queue.pop(0)
            keys = self._entity_key(aid)
            if any(count.get(k, 0) >= 2 for k in keys):
                deferred.append(aid)
                continue
            for k in keys:
                count[k] = count.get(k, 0) + 1
            out.append(aid)
        return out + deferred

    def log_click(self, uid, aid):
        """열람을 유저 히스토리 끝에 이어 붙인다. 시각은 데이터 속 마지막 상호작용 + 60초씩 (실제 현재 시각을 쓰면 7일 창이 데모 클릭만 남음)."""
        if aid not in self.articles:
            return
        hist = self.history(uid) or []
        t = (hist[-1]["time"] if hist else int(time.time())) + 60
        self.clicks.setdefault(uid, []).append({"article_id": aid, "time": t})
        with open(self.clicks_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"user_id": uid, "article_id": aid, "time": t}, ensure_ascii=False) + "\n")

    def reset_user(self, uid):
        """데모 중 쌓인 열람 기록을 지워 처음 상태로 되돌린다."""
        self.clicks.pop(uid, None)
        with open(self.clicks_path, "w", encoding="utf-8") as f:
            for u, rows in self.clicks.items():
                for r in rows:
                    f.write(json.dumps({"user_id": u, **r}, ensure_ascii=False) + "\n")

    def demo_users(self):
        out = []
        for d in DEMO_USERS:
            if d["user_id"] not in self.user_off:
                continue
            hist = self.history(d["user_id"]) or []
            out.append({**d, "overall": history_dist(hist, self.articles, None), "n_read": len(self.clicks.get(d["user_id"], []))})
        return out

    def stats(self, uid, page_ids):
        hist = self.history(uid) or []
        return {"weekly": history_dist(hist, self.articles, 7),
                "overall": history_dist(hist, self.articles, None),
                "feed": label_dist(page_ids, self.articles)}

    def _cache_for(self, article_id, kind):
        """LIME/IG 마스킹 모드 기사의 keyword 설명만 전용 캐시를 본다. verdict·contrast는 원래 캐시 (2026-11-03 사용자 지시: Why this verdict는 원래대로)."""
        return self.cache_lime if (article_id in self.lime_ids and kind == "keyword") else self.cache

    def lookup(self, article_id, kind, key):
        a = self.articles.get(article_id)
        if a:
            if kind == "keyword" and (a.get("rationales") or {}).get(key):
                return a["rationales"][key], "article"
            if kind == "verdict" and a.get("verdict_rationale"):
                return a["verdict_rationale"], "article"
            if kind == "contrast":
                for c in a.get("counterparts") or []:
                    if c.get("article_id") == key and c.get("claim_contrast"):
                        return c["claim_contrast"], "article"
        for k in ((article_id, kind, key), (article_id, kind, "")):
            t = self._cache_for(article_id, kind).get(k)
            if t:
                if kind == "contrast" and isinstance(t, str):
                    try:
                        t = json.loads(t)
                    except ValueError:
                        pass
                if kind == "contrast" and not (isinstance(t, dict) and "this" in t):
                    continue  # contrast-v1 형식(fake/verified) 캐시는 무시
                return t, "cache"
        return None, "pending"

    def get_rationale(self, article_id, kind, key):
        """캐시 조회 자리. 실제 Gemini 호출은 rationale 모듈이 여기에 붙인다."""
        t, status = self.lookup(article_id, kind, key)
        if t is not None:
            return t, status
        if self.source == "real":
            try:
                from crystal.rationale.online import get_rationale
                t = get_rationale(article_id, kind, key, self.articles)
                self._cache_for(article_id, kind)[(article_id, kind, key)] = t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
                return t, "online"
            except Exception:
                pass
        if kind == "contrast":
            return {"this": PLACEHOLDER, "related": PLACEHOLDER}, "pending"
        return PLACEHOLDER, "pending"


def load_store():
    mode = os.environ.get("CRYSTAL_DATA", "auto")
    if mode == "sample":
        return Store(SAMPLE_DIR, "sample", "CRYSTAL_DATA=sample")
    ok, reason = _real_ready()
    if mode == "real" or ok:
        return Store(REAL_DIR, "real", reason if ok else f"CRYSTAL_DATA=real 강제 ({reason})")
    return Store(SAMPLE_DIR, "sample", reason)
