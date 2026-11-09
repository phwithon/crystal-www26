"""추천기 인터페이스. 기본은 CROWN(crown_infer.CrownRecommender), CRYSTAL_RECSYS=itemknn 이면 ItemKNN."""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from crystal.recsys.dataset import distinct_history, iter_users, load_articles, user_item_matrix  # noqa: E402


class Recommender:
    """rank: 히스토리(시간순 article_id, 오래된 것 먼저)와 후보 article_id -> 후보별 점수 (높을수록 추천)."""

    def rank(self, history_article_ids, candidate_article_ids):
        return self.rank_batch([history_article_ids], candidate_article_ids)[0].tolist()

    def rank_batch(self, histories, candidate_article_ids):  # -> np.ndarray [B, C]
        raise NotImplementedError


class ItemKNNRecommender(Recommender):
    """학습 없음. 유저-기사 이진 행렬에서 기사 간 코사인 유사도를 구하고, 점수 = 히스토리 기사와 후보 기사 유사도 합.
    유사도가 전부 0인 유저(또는 동점)는 인기도(distinct 클릭 수)로 채운다."""

    def __init__(self, articles=None, users=None):
        self.articles = articles if articles is not None else load_articles()
        self.idx_of = {a["article_id"]: i for i, a in enumerate(self.articles)}
        X, self.user_ids, self.histories = user_item_matrix(users if users is not None else iter_users(), self.idx_of, len(self.articles))
        self.X = X
        self.pop = np.asarray(X.sum(0)).ravel()
        self.norm = np.sqrt(self.pop)
        self._cand_cache = (None, None)

    def _cand_sim(self, candidate_article_ids):  # [N, C] 코사인, 자기 자신은 0
        key = tuple(candidate_article_ids)
        if self._cand_cache[0] != key:
            c = np.array([self.idx_of[a] for a in key])
            S = (self.X.T @ self.X[:, c]).toarray()
            S /= np.maximum(self.norm[:, None] * self.norm[c][None, :], 1e-12)
            S[c, np.arange(len(c))] = 0.0
            self._cand_cache = (key, S.astype(np.float32))
        return self._cand_cache[1]

    def pop_of(self, candidate_article_ids):
        return self.pop[[self.idx_of[a] for a in candidate_article_ids]]

    def rank_batch(self, histories, candidate_article_ids):
        S = self._cand_sim(candidate_article_ids)
        rows, cols = [], []
        for b, h in enumerate(histories):
            idx = [self.idx_of[a] for a in h if a in self.idx_of]
            rows.extend([b] * len(idx)); cols.extend(idx)
        H = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(histories), len(self.articles)))
        return np.asarray(H @ S)

    def rank_rows(self, row_slice, candidate_article_ids):  # 생성 시 읽은 유저 행렬의 행 범위로 바로 점수 [B, C]
        return np.asarray(self.X[row_slice] @ self._cand_sim(candidate_article_ids))


def load_recommender(name=None, **kw):
    """기본 crown. 환경변수 CRYSTAL_RECSYS=itemknn 으로 기존 ItemKNN을 고른다."""
    name = name or os.environ.get("CRYSTAL_RECSYS", "crown")
    if name == "itemknn":
        return ItemKNNRecommender(**kw)
    if name == "crown":
        from crystal.recsys.crown_infer import CrownRecommender  # 지연 import (torch·CROWN 로드)
        return CrownRecommender(**kw)
    raise ValueError(name)
