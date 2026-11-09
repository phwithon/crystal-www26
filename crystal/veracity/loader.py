"""RoBERTa 진위 예측기 로더. load(fold)는 roberta_veracity/folds/fold{fold}/best (K-fold, 기사마다 자기 fold 모델), load()는 단일 모델 roberta_veracity/best.
사용: tok, model = load(fold); ids, mask = encode(tok, title, body)  →  model(inputs_embeds=model.roberta.embeddings.word_embeddings(ids), attention_mask=mask).logits
IG hook 대상: model.roberta.embeddings.word_embeddings. 라벨 0=real, 1=fake. 입력 정규화는 train.py와 동일(strip, 본문==제목이면 "").
"""
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "roberta_veracity")
CKPT = os.path.join(RV, "best")
MAXLEN = 512


def ckpt_path(fold=None):
    return CKPT if fold is None else os.path.join(RV, "folds", f"fold{int(fold)}", "best")


def load(fold=None, device="cuda"):
    path = ckpt_path(fold)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    return tok, model


def normalize(title, body):
    title, body = (title or "").strip(), (body or "").strip()
    return title, ("" if body == title else body)


def encode(tok, title, body, device="cuda"):
    """한 기사 → (input_ids, attention_mask), 각 [1, L]. 학습과 같은 형식(title, body, truncation=only_second)."""
    t, b = normalize(title, body)
    enc = tok(t, b, truncation="only_second", max_length=MAXLEN, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def encode_batch(tok, titles, bodies, device="cuda"):
    pairs = [normalize(t, b) for t, b in zip(titles, bodies)]
    enc = tok([p[0] for p in pairs], [p[1] for p in pairs], truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)
