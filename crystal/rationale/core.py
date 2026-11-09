"""Gemini verbalizer 공통부: 프롬프트, 호출(재시도·모델 대체), 판정 일치 검증, 캐시."""
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(ROOT, "crystal", "data")
ARTICLES_PATH = os.path.join(DATA_DIR, "articles.jsonl")
CACHE_PATH = os.path.join(DATA_DIR, "rationale_cache.jsonl")
LIME_CACHE_PATH = os.path.join(DATA_DIR, "rationale_cache_lime.jsonl")  # 2026-11-03 LIME 키워드 모드 전용 설명 캐시 (IG 기준 설명과 분리)
LIME_MASKED_CACHE_PATH = os.path.join(DATA_DIR, "rationale_cache_lime_masked.jsonl")  # 인물명 마스킹 모델 키워드 전용
PROMPT_DIR = os.path.join(HERE, "prompts")
DEFAULT_MODEL = os.environ.get("CRYSTAL_GEMINI_MODEL", "gemini-3.5-flash-lite")  # 2026-11-03 사용자 지시: 캐시에 없을 때만 온라인 생성, flash-lite
FALLBACK_MODEL = "gemini-3.5-flash"  # flash-lite 불가 시
BODY_LIMIT = 2500
OPENING_LIMIT = 500
GEN_CONFIG = {"temperature": 0.2, "response_mime_type": "application/json"}


class QuotaExceeded(RuntimeError):
    pass


def load_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GEMINI_API_KEY not found in environment or .env")


def load_prompt(kind):
    with open(os.path.join(PROMPT_DIR, kind + ".md"), encoding="utf-8") as f:
        head, body = f.read().split("\n", 1)
    return head.split(":", 1)[1].strip(), body.strip()


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace("{{" + k + "}}", str(v))
    return template


def label_word(final_label):
    return "Fake" if final_label == 1 else "True"


def verdict_str(a):
    return f"{label_word(a['final_label']).upper()} (confidence {max(a['pred_prob_fake'], 1 - a['pred_prob_fake']):.1%})"


def opening(a, limit=OPENING_LIMIT):
    text = (a.get("body") or "").strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def unique_phrases(a):
    seen, out = set(), []
    for k in a.get("keywords") or []:
        if k["phrase"] not in seen:
            seen.add(k["phrase"])
            out.append(k)
    return out


def evidence_text(a, articles):
    lines = []
    for c in a.get("counterparts") or []:
        b = articles.get(c["article_id"])
        if b:
            lines.append(f"- [{label_word(b['final_label']).upper()}] {b['title']}\n  Opening: {opening(b)}")
    return "\n".join(lines) if lines else "none"


def build_keyword_prompt(a, kws, articles):
    version, tpl = load_prompt("keyword")
    phrases = "\n".join(f'- phrase: "{k["phrase"]}"\n  sentence: "{k["sentence"]}"' for k in kws)
    return version, fill(tpl, verdict=verdict_str(a), title=a["title"], body=opening(a, BODY_LIMIT),
                         evidence=evidence_text(a, articles), phrases=phrases)


def build_verdict_prompt(a, articles):
    version, tpl = load_prompt("verdict")
    kws = unique_phrases(a)
    phrases = "\n".join(f'- "{k["phrase"]}" ({k["field"]})' for k in kws) if kws else "none"
    return version, fill(tpl, verdict=verdict_str(a), title=a["title"], body=opening(a, BODY_LIMIT),
                         phrases=phrases, evidence=evidence_text(a, articles), label_word=label_word(a["final_label"]))


def build_contrast_prompt(a, b):
    version, tpl = load_prompt("contrast")
    return version, fill(tpl, a_verdict=label_word(a["final_label"]).upper(), a_title=a["title"], a_body=opening(a),
                         b_verdict=label_word(b["final_label"]).upper(), b_title=b["title"], b_body=opening(b))


_SUBJ = r"\b(?:article|story|claim|report|piece|headline|news)\b[^.;]{0,60}?\b(?:is|was|are|appears|seems|has been|proves|confirmed|verified|labeled|labelled|deemed|considered)\b[^.;]{0,25}?\b"
_CONTRA = {1: re.compile(_SUBJ + r"(?:true|accurate|genuine|credible|authentic|real|factual|legitimate)\b", re.I),
           0: re.compile(_SUBJ + r"(?:fake|false|fabricated|misleading|untrue|a hoax)\b", re.I)}
_LABEL = {1: re.compile(r"\blabel(?:l)?ed\s+(?:as\s+)?true\b", re.I), 0: re.compile(r"\blabel(?:l)?ed\s+(?:as\s+)?fake\b", re.I)}


def consistent(text, final_label):
    """생성 문장이 주어진 판정과 반대 진위를 말하면 False."""
    if not isinstance(text, str) or not text.strip():
        return False
    return not (_CONTRA[final_label].search(text) or _LABEL[final_label].search(text))


CONTRAST_KEYS = ("this", "related")


def contrast_ok(obj):
    return isinstance(obj, dict) and all(isinstance(obj.get(k), str) and obj[k].strip() for k in CONTRAST_KEYS)


class Gemini:
    def __init__(self, model=DEFAULT_MODEL, fallback=FALLBACK_MODEL, interval=0.0, max_retry=3):
        from google import genai
        self.client = genai.Client(api_key=load_api_key())
        self.model, self.fallback, self.interval, self.max_retry = model, fallback, interval, max_retry
        self.calls, self.errors, self.quota_hits, self.last, self.codes, self.used = 0, 0, 0, 0.0, {}, {}

    def _once(self, model, prompt):
        from google.genai import types
        wait = self.interval - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        self.calls += 1
        kw = dict(GEN_CONFIG)
        if "lite" not in model:  # lite 계열은 thinking_config를 받지 않음(400)
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)  # 두 문장 설명에 thinking 불필요(속도·비용)
        r = self.client.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(**kw))
        return r.text or ""

    def generate(self, prompt):
        from google.genai import errors
        code, last = None, None
        for model in (self.model, self.fallback):
            if not model:
                continue
            for attempt in range(self.max_retry):
                try:
                    out = self._once(model, prompt)
                    self.used[model] = self.used.get(model, 0) + 1
                    return out, model
                except errors.APIError as e:
                    self.errors += 1
                    code, last = getattr(e, "code", None), e
                    self.codes[f"{model}:{code}"] = self.codes.get(f"{model}:{code}", 0) + 1
                    if code == 429:
                        self.quota_hits += 1
                    if code not in (429, 500, 503) or attempt == self.max_retry - 1:
                        break
                    time.sleep(5 * (attempt + 1))
            if code not in (404, 429, 500, 503):
                raise RuntimeError(f"{model}: {code} {getattr(last, 'message', last)}")
        raise QuotaExceeded(f"gave up after {self.max_retry} retries on each of {self.model}, {self.fallback} (last code {code})")


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[1] if "\n" in text else text.strip("`")
        text = text.rsplit("```", 1)[0] if "```" in text else text
    return json.loads(text)


class Cache:
    def __init__(self, path=CACHE_PATH):
        self.path, self.rows = path, {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    self.rows[(r["article_id"], r["kind"], r["key"])] = r

    def get(self, article_id, kind, key):
        r = self.rows.get((article_id, kind, key))
        return r["text"] if r else None

    def put(self, article_id, kind, key, text, model, prompt_version):
        r = {"article_id": article_id, "kind": kind, "key": key, "text": text, "model": model, "prompt_version": prompt_version}
        self.rows[(article_id, kind, key)] = r
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_articles(path=ARTICLES_PATH):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            out[a["article_id"]] = a
    return out


def gen_keywords(llm, cache, a, kws, articles, stats):
    """기사 하나의 키워드 설명을 한 번에 만든다. 판정 불일치 구문은 1회 재생성."""
    pending, out = list(kws), {}
    for _ in range(2):
        if not pending:
            break
        version, prompt = build_keyword_prompt(a, pending, articles)
        text, model = llm.generate(prompt)
        try:
            obj = parse_json(text)
        except ValueError:
            obj = None
        if not isinstance(obj, dict):
            stats["parse_fail"] += 1
            continue
        nxt = []
        for k in pending:
            t = obj.get(k["phrase"])
            if isinstance(t, str) and consistent(t, a["final_label"]):
                out[k["phrase"]] = t
                cache.put(a["article_id"], "keyword", k["phrase"], t.strip(), model, version)
            else:
                stats["discarded"] += 1
                nxt.append(k)
        pending = nxt
    stats["dropped"] += len(pending)
    return out


def gen_verdict(llm, cache, a, articles, stats):
    version, prompt = build_verdict_prompt(a, articles)
    for _ in range(2):
        text, model = llm.generate(prompt)
        try:
            t = parse_json(text).get("verdict_rationale")
        except (ValueError, AttributeError):
            stats["parse_fail"] += 1
            continue
        if isinstance(t, str) and consistent(t, a["final_label"]):
            cache.put(a["article_id"], "verdict", "", t.strip(), model, version)
            return t
        stats["discarded"] += 1
    stats["dropped"] += 1
    return None


def gen_contrast(llm, cache, a, b, stats):
    version, prompt = build_contrast_prompt(a, b)
    for _ in range(2):
        text, model = llm.generate(prompt)
        try:
            obj = parse_json(text)
        except ValueError:
            stats["parse_fail"] += 1
            continue
        if contrast_ok(obj):
            obj = {k: obj[k].strip() for k in CONTRAST_KEYS}
            cache.put(a["article_id"], "contrast", b["article_id"], json.dumps(obj, ensure_ascii=False), model, version)
            return obj
        stats["discarded"] += 1
    stats["dropped"] += 1
    return None


def new_stats():
    return {"discarded": 0, "dropped": 0, "parse_fail": 0}
