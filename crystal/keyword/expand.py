"""상위 단어를 spaCy 의존 구문으로 확장 (PRISM keyword_expander 규칙: ADJ amod→ADJ NOUN, NOUN→amod ADJ 또는 compound 붙임, VERB→dobj 또는 prep+pobj).
확장 구문을 원문에서 못 찾으면 단어로 되돌린다. 결과 keywords[{phrase, field, start, end, score, sentence}], field[start:end] == phrase 보장."""
import re

import spacy

SPACY_MODEL = "en_core_web_sm"


def _constituents(token):
    pos = token.pos_
    if pos == "ADJ":
        head = token.head
        if token.dep_ == "amod" and head.pos_ == "NOUN":
            mods = [token] + [c for c in head.children if c.pos_ == "ADJ" and c != token]
            return sorted(mods, key=lambda x: x.i) + [head], "ADJ → ADJ+NOUN"
        return [token], "ADJ → ADJ"
    if pos == "NOUN":
        mods = [c for c in token.children if c.pos_ == "ADJ" and c.dep_ == "amod"]
        if mods:
            return sorted(mods, key=lambda x: x.i) + [token], "NOUN → ADJ+NOUN"
        comps = [c for c in token.children if c.dep_ == "compound"]
        if comps:
            return sorted(comps, key=lambda x: x.i) + [token], "NOUN → COMPOUND+NOUN"
        return [token], "NOUN → NOUN"
    if pos == "VERB":
        for c in token.children:
            if c.dep_ == "dobj":
                return [token, c], "VERB → VERB+OBJ"
        for c in token.children:
            if c.dep_ == "prep":
                for g in c.children:
                    if g.dep_ == "pobj":
                        return [token, c, g], "VERB → VERB+PREP+NOUN"
        return [token], "VERB → VERB"
    return [token], f"{pos} → {pos}"


def _clean(s):
    return re.sub(r"\s+", " ", re.sub(r"\.+", "", s)).strip()


class Expander:
    def __init__(self):
        self.nlp = spacy.load(SPACY_MODEL)

    def _locate(self, text, parts):
        """구문의 원문 위치. 토큰이 연속이면 그 범위, 아니면 공백 유연 검색, 못 찾으면 None."""
        if all(parts[i + 1].i == parts[i].i + 1 for i in range(len(parts) - 1)):
            return parts[0].idx, parts[-1].idx + len(parts[-1].text)
        words = _clean(" ".join(p.text for p in parts)).split(" ")
        sent = parts[0].sent  # 같은 문장 안에서만 찾는다
        m = re.search(r"\s+".join(re.escape(w) for w in words), text[sent.start_char:sent.end_char])
        return (sent.start_char + m.start(), sent.start_char + m.end()) if m else None

    def expand(self, title, body, top_words):
        """top_words: ig.select_top_words 결과. 반환 (keywords, 확장 통계 리스트[{word, expanded, expansion_type}])."""
        texts = {"title": title, "body": body}
        docs = {f: self.nlp(texts[f]) if texts[f] else None for f in texts}
        out, stats = [], []
        for w in top_words:
            text, doc = texts[w["field"]], docs[w["field"]]
            tok = None
            if doc is not None:
                for t in doc:
                    if t.idx <= w["start"] and t.idx + len(t.text) >= w["end"] and t.text.strip():
                        tok = t
                        break
            if tok is None:
                start, end, sent, etype = w["start"], w["end"], None, "NO_TOKEN"
            else:
                parts, etype = _constituents(tok)
                span = self._locate(text, parts) if len(parts) > 1 else (tok.idx, tok.idx + len(tok.text))
                if span is None or not _clean(text[span[0]:span[1]]):
                    span, etype = (tok.idx, tok.idx + len(tok.text)), etype + " (fallback)"
                start, end = span
                sent = tok.sent.text.strip()
            phrase = text[start:end]
            if sent is None:
                sent = _sentence_of(text, start)
            out.append({"phrase": phrase, "field": w["field"], "start": start, "end": end, "score": round(float(w["score"]), 6), "sentence": sent})
            stats.append({"word": w["word"], "expanded": phrase, "expansion_type": etype})
        return _dedup(out), stats


def _sentence_of(text, pos):
    s = max([m.end() for m in re.finditer(r"[.!?]\s+", text[:pos])] or [0])
    m = re.search(r"[.!?](\s|$)", text[pos:])
    return text[s:pos + m.end() if m else len(text)].strip()


def _dedup(kws):
    """겹치는 구간은 점수 높은 것만 남긴다 (입력은 점수 내림차순)."""
    kept = []
    for k in kws:
        if any(o["field"] == k["field"] and k["start"] < o["end"] and o["start"] < k["end"] for o in kept):
            continue
        kept.append(k)
    return kept
