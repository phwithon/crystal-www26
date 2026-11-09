"""Integrated Gradients 단어 기여도 (RoBERTa 진위 예측기, 목표 클래스 final_label).
PRISM IG_khan/ig_analyzer.py 이식: hook 대상 word_embeddings, baseline = pad 토큰 임베딩, steps 50 사다리꼴, 토큰 L2 norm → word_ids()로 서브워드 합산 → min-max."""
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from crystal.veracity.loader import MAXLEN, normalize  # noqa: E402

STEPS, CHUNK = 50, 26  # 51개 보간점을 CHUNK개씩 배치로 forward/backward

NLTK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
}
SPECIAL_TOKENS = {'<unk>', '<sep>', '<pad>', '.', ',', '!', '?', '"', "'", '-', ':', ';', '(', ')', '[', ']'}
PRE_DEFINED_STOPWORDS = {
    '—', '–', '\x97', '“', '”', '--', '|', '…',
    'getty', 'images',
    'associated', 'newsweek', 'reported', 'reporting', 'reached', 'said', 'says', 'according', 'publisher',
    'though', 'thus', 'likewise', 'however', 'also', 'moreover', 'although',
    'related', 'read', 'watch', 'breaking', 'updated',
    'caption', 'hide', 'subscriber', 'login', '©', '*', '&', '​', '﻿',
    'click', 'newsletter', 'app',
}
DEFAULT_IGNORE_TOKENS = NLTK_STOPWORDS | SPECIAL_TOKENS | PRE_DEFINED_STOPWORDS
_HAS_ALNUM = re.compile(r"[^\W_]")


def is_ignored(word):
    w = word.lower().lstrip("'’")
    if w in DEFAULT_IGNORE_TOKENS or word.lower() in DEFAULT_IGNORE_TOKENS or not _HAS_ALNUM.search(w):
        return True
    return (word.startswith('[') and word.endswith(']')) or '@' in word


def _token_attr(model, emb, ids, mask, target):
    x = emb(ids)
    base = emb(torch.full_like(ids, model.config.pad_token_id))
    diff = x - base
    acc = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, STEPS + 1, device=ids.device)
    w = torch.full((STEPS + 1,), 1.0, device=ids.device)
    w[0] = w[-1] = 0.5
    for s in range(0, STEPS + 1, CHUNK):
        a, ww = alphas[s:s + CHUNK], w[s:s + CHUNK]
        inp = (base + a[:, None, None] * diff).detach().requires_grad_(True)
        logits = model(inputs_embeds=inp, attention_mask=mask.expand(len(a), -1)).logits
        grad = torch.autograd.grad(logits[:, target].sum(), inp)[0]
        acc += (grad * ww[:, None, None]).sum(0, keepdim=True)
    attr = (diff * acc / STEPS)[0]  # [L, H]
    return attr.norm(dim=-1), attr.sum(dim=-1), logits


class IGAnalyzer:
    def __init__(self, tok, model):
        self.tok, self.model = tok, model
        self.emb = model.roberta.embeddings.word_embeddings
        model.requires_grad_(False)

    def analyze(self, title, body, target):
        """반환: words [{word, field, start, end, score(min-max L2), signed}], 메타 {truncated, completeness, logit_target}."""
        t, b = normalize(title, body)
        enc = self.tok(t, b, truncation="only_second", max_length=MAXLEN, return_offsets_mapping=True, return_tensors="pt")
        ids, mask = enc["input_ids"].to(self.model.device), enc["attention_mask"].to(self.model.device)
        with torch.no_grad():
            lx = self.model(input_ids=ids, attention_mask=mask).logits[0, target].item()
            lb = self.model(input_ids=torch.full_like(ids, self.model.config.pad_token_id), attention_mask=mask).logits[0, target].item()
        l2, signed, _ = _token_attr(self.model, self.emb, ids, mask, target)
        l2, signed = l2.tolist(), signed.tolist()
        seq, wid, off = enc.sequence_ids(), enc.word_ids(), enc["offset_mapping"][0].tolist()
        words, cur = [], None
        for i, (sq, wi, (s, e)) in enumerate(zip(seq, wid, off)):
            if sq is None or e <= s:
                continue
            key = (sq, wi)
            if cur is None or cur["_k"] != key:
                cur = {"_k": key, "field": "title" if sq == 0 else "body", "start": s, "end": e, "l2": 0.0, "signed": 0.0}
                words.append(cur)
            cur["end"] = max(cur["end"], e)
            cur["l2"] += l2[i]
            cur["signed"] += signed[i]
        src = {"title": t, "body": b}
        lo, hi = (min(w["l2"] for w in words), max(w["l2"] for w in words)) if words else (0, 0)
        out = []
        for w in words:
            out.append({"word": src[w["field"]][w["start"]:w["end"]], "field": w["field"], "start": w["start"], "end": w["end"],
                        "score": (w["l2"] - lo) / (hi - lo) if hi > lo else 0.0, "signed": w["signed"]})
        n_body_tok = sum(1 for sq in seq if sq == 1)
        meta = {"n_tokens": int(mask.sum()), "truncated": int(mask.sum()) >= MAXLEN and n_body_tok > 0,
                "completeness": sum(signed) - (lx - lb), "logit_target": lx}
        return out, meta


SCORE_MODES, DEFAULT_SCORE = ("l2", "signed"), "signed"


def select_top_words(words, k=10, score=DEFAULT_SCORE):
    """불용어·구두점 제외, 같은 단어(소문자)는 점수 최댓값으로 묶어 상위 k개. 위치는 첫 등장(제목 → 본문).
    score="signed": 목표 클래스 방향 부호 합이 양수인 단어만, 기사 최대값으로 나눠 0~1. "l2": min-max L2(PRISM 원래 방식)."""
    best, first = {}, {}
    for w in words:
        if is_ignored(w["word"]) or (score == "signed" and w["signed"] <= 0):
            continue
        key, val = w["word"].lower(), w["signed"] if score == "signed" else w["score"]
        if key not in first:
            first[key] = w
        if key not in best or val > best[key][0]:
            best[key] = (val, w)
    ranked = sorted(best, key=lambda k_: -best[k_][0])[:k]
    top = max((best[k_][0] for k_ in ranked), default=1.0) or 1.0
    return [dict(first[k_], score=best[k_][0] / top if score == "signed" else best[k_][0], signed=best[k_][1]["signed"]) for k_ in ranked]
