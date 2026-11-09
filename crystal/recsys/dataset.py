"""데이터 계약 파일 읽기와 유저-기사 희소 행렬."""
import json
import os

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "crystal", "data")
N_ARTICLES = 18124  # 2026-10-21 중복 id 2건만 제거, 본문 없는 기사 유지 (prepare_data.py)


def load_articles(path=os.path.join(DATA, "articles.jsonl")):
    """시작할 때 한 번 읽어 메모리에 들고 쓴다 (다른 모듈이 이 파일을 덮어쓸 수 있음)."""
    with open(path, encoding="utf-8") as f:
        arts = [json.loads(l) for l in f]
    assert len(arts) == N_ARTICLES, f"articles.jsonl {len(arts)}건, 기대 {N_ARTICLES}건 (쓰는 중일 수 있음)"
    return arts


def iter_users(path=os.path.join(DATA, "users.jsonl")):
    with open(path, encoding="utf-8") as f:
        for l in f:
            yield json.loads(l)


def distinct_history(history):
    """시간순 히스토리에서 같은 기사 재상호작용을 빼고 첫 등장 순서로 article_id 목록을 돌려준다."""
    return list(dict.fromkeys(h["article_id"] for h in history))


def user_item_matrix(users, idx_of, n_items):
    """[U, n_items] 이진 CSR과 user_id 목록, 유저별 distinct 히스토리. 히스토리 없는 유저는 뺀다."""
    rows, cols, uids, hists = [], [], [], []
    for u in users:
        h = [a for a in distinct_history(u["history"]) if a in idx_of]
        if not h:
            continue
        rows.extend([len(uids)] * len(h)); cols.extend(idx_of[a] for a in h)
        uids.append(u["user_id"]); hists.append(h)
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(uids), n_items))
    return X, uids, hists
