"""CROWN(crown-www25) 학습 체크포인트로 추론하는 래퍼. 코드 수정 없이 crown-www25 모듈을 그대로 import해 조립한다.

- 체크포인트: crown/safetensors/fnn_gossip_desc/CROWN-CROWN/#6/CROWN-CROWN (키 "CROWN-CROWN", 153텐서), strict=True 로드.
- 토큰화: crown-www25 corpus.py의 MIND 토크나이저(pat, is_number)를 그대로 import. 제목 32·본문 64, <NUM>·<UNK>(1) 처리.
- category_dict 재구성: corpus.py 50행 `category_dict = {}` + 74-75행 "if category not in category_dict:
  category_dict[category] = len(category_dict)" — news.tsv 등장 순서대로 번호. 친구 news.tsv의 category 열은
  기사 진위 라벨 "label1"/"label0"이고 gossip.csv 첫 행 label=1이므로 {"label1": 0, "label0": 1}.
- userEncoders.CROWN은 user_history_graph(52x52 self-normalize-symmetric)를 인자로 받지만 forward에서 쓰지 않는다
  (SUE 전용). 실제로 쓰는 그래프는 create_bipartite_graph가 만드는 고정 edge_index 하나뿐이며, 이는 학습 배치
  크기(32)에 의존한다. MessagePassing node_dim=-2라 그래프는 각 유저의 노드 축(50 히스토리 + 32 user node)에만
  적용되고 배치 축과는 독립 → 학습과 동일한 edge_index(소스 0..31, 타깃 0..49)를 쓰면 유저 배치 크기는 자유.
  이 동등성은 --selftest가 user_encoder.forward 원본 호출과 대조해 검증한다.
- 효율: 뉴스 인코딩은 전 기사(18,124) + PAD 1건을 1회 사전계산. user_encoder가 내부에서 news_encoder를 다시
  돌리는 구조라서, forward를 그대로 쓰지 않고 같은 서브모듈(graph_sage, K, Q)을 같은 수식으로 호출해 사전계산
  임베딩을 주입한다 (점수 동일성은 --selftest로 확인).

python -X utf8 crystal/recsys/crown_infer.py --selftest  (KMP_DUPLICATE_LIB_OK=TRUE 필요)
"""
import json
import os
import pickle
import shutil
import sys
import tempfile
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from crystal.recsys.dataset import ROOT, load_articles  # noqa: E402
from crystal.recsys.recommender import Recommender  # noqa: E402

CROWN_SRC = os.path.join(ROOT, "crown", "crown-www25")
SAFE = os.path.join(ROOT, "crown", "safetensors")
CKPT = os.path.join(SAFE, "fnn_gossip_desc", "CROWN-CROWN", "#6", "CROWN-CROWN")
VOCAB = os.path.join(SAFE, "vocabulary-3-MIND-32-64-fnn_gossip_desc.json")
CATEGORY_DICT = {"label1": 0, "label0": 1}  # corpus.py 등장순 재구성 (위 docstring)
SUBCATEGORY = 0  # subCategory-fnn_gossip_desc.json = {"": 0}
TRAIN_BATCH = 32  # 체크포인트 user_node_embedding (32, 900) = 학습 batch_size


def _shim(name, **attrs):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


def _import_crown():
    """crown-www25 모듈 import. CROWN 경로에서 안 쓰는 외부 의존(fastformer, torch_scatter, torchtext)은
    import만 통과하는 가짜 모듈로 채운다. 진짜 torch_geometric을 먼저 올려 가짜 torch_scatter가 PyG에 안 잡히게 한다."""
    import torch_geometric  # noqa: F401

    def _unused(*a, **k):
        raise NotImplementedError("CROWN 추론 경로에서는 쓰이지 않는 함수")

    _shim("torch_scatter", scatter_sum=_unused, scatter_softmax=_unused)
    _shim("fastformer", FastformerEncoder=_unused)
    if "torchtext" not in sys.modules:
        try:
            import torchtext  # noqa: F401
        except ImportError:
            _shim("torchtext")
            _shim("torchtext.vocab", GloVe=_unused)
            sys.modules["torchtext"].vocab = sys.modules["torchtext.vocab"]
    sys.path.insert(0, CROWN_SRC)
    try:
        import corpus as crown_corpus
        import model as crown_model
    finally:
        sys.path.remove(CROWN_SRC)
    return crown_model, crown_corpus


def _config(vocab_size):
    """crown-www25 config.py 기본값(mind 오버라이드 포함). 파일명·체크포인트에서 확정된 값만 다름:
    max_abstract_length 64(기본 128, pe 버퍼 (1,64,300)으로 확인), dataset 이름."""
    c = types.SimpleNamespace(
        news_encoder="CROWN", user_encoder="CROWN", click_predictor="dot_product",
        dataset="fnn_gossip_desc", tokenizer="MIND", word_threshold=3,
        max_title_length=32, max_abstract_length=64, max_history_num=50,
        batch_size=TRAIN_BATCH, dropout_rate=0.2,
        num_layers=1, feedforward_dim=512, head_num=10, head_dim=20,
        intent_embedding_dim=400, intent_num=3, alpha=0.3, attention_dim=400,
        word_embedding_dim=300, isab_num_inds=4, isab_num_heads=4,
        category_embedding_dim=50, subCategory_embedding_dim=50,
        vocabulary_size=vocab_size, category_num=len(CATEGORY_DICT), subCategory_num=1,
    )
    return c


def load_model(device):
    crown_model, crown_corpus = _import_crown()
    with open(VOCAB, encoding="utf-8") as f:
        word_dict = json.load(f)
    cfg = _config(len(word_dict))
    # newsEncoders.NewsEncoder.__init__이 cwd의 word_embedding pkl을 읽으므로 0 텐서로 임시 생성
    # (실제 가중치는 바로 아래 load_state_dict가 덮어씀)
    tmp = tempfile.mkdtemp(prefix="crown_we_")
    fname = ("word_embedding-%d-%d-%s-%d-%d-%s.pkl" %
             (cfg.word_threshold, cfg.word_embedding_dim, cfg.tokenizer,
              cfg.max_title_length, cfg.max_abstract_length, cfg.dataset))
    with open(os.path.join(tmp, fname), "wb") as f:
        pickle.dump(torch.zeros(cfg.vocabulary_size, cfg.word_embedding_dim), f)
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        model = crown_model.Model(cfg)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    sd = torch.load(CKPT, map_location="cpu")["CROWN-CROWN"]
    model.load_state_dict(sd, strict=True)  # 153텐서 전부 일치해야 함
    model.to(device).eval()
    return model, cfg, word_dict, crown_corpus


class CrownRecommender(Recommender):
    """rank(history_ids, candidate_ids): 히스토리 최근 50개(데이터 그대로, 중복 포함)로 CROWN 점수."""

    def __init__(self, articles=None, device=None, user_batch=256):
        # PyG scatter가 CUDA 원자합을 써서 run마다 1e-6 수준이 흔들림 → 결정론 모드 (QA: 같은 입력 → 같은 순서)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.user_batch = user_batch
        self.model, self.cfg, self.word_dict, self.corpus = load_model(self.device)
        self.articles = articles if articles is not None else load_articles()
        self.idx_of = {a["article_id"]: i + 1 for i, a in enumerate(self.articles)}  # 0 = <PAD> 뉴스
        self.news_repr = self._encode_all_news()  # [N+1, 900] (PAD 포함)
        ue = self.model.user_encoder
        # 학습과 동일한 bipartite edge_index (user_history_mask 내용은 쓰이지 않고 shape만 쓰임)
        self._edge_index = ue.create_bipartite_graph(torch.empty(TRAIN_BATCH, self.cfg.max_history_num), self.device)
        self._user_node = ue.user_node_embedding.detach()  # [32, 900]
        self._cand_cache = (None, None, None)

    # --- 뉴스 토큰화 (corpus.py 364-420행과 동일, entity 제외) ---
    def _tokenize(self, text, max_len):
        pat, is_number = self.corpus.pat, self.corpus.is_number
        ids = np.zeros(max_len, dtype=np.int64)
        mask = np.zeros(max_len, dtype=bool)
        for i, word in enumerate(pat.findall((text or "").lower())):
            if i == max_len:
                break
            if is_number(word):
                ids[i] = self.word_dict["<NUM>"]
            elif word in self.word_dict:
                ids[i] = self.word_dict[word]
            else:
                ids[i] = 1  # <UNK>
            mask[i] = True
        return ids, mask

    @torch.no_grad()
    def _encode_all_news(self, chunk=512):
        n = len(self.articles) + 1
        tl, al = self.cfg.max_title_length, self.cfg.max_abstract_length
        tt = np.zeros([n, tl], np.int64); tm = np.zeros([n, tl], bool)
        bt = np.zeros([n, al], np.int64); bm = np.zeros([n, al], bool)
        cat = np.zeros(n, np.int64)
        tm[0][0] = bm[0][0] = True  # <PAD> 뉴스 (corpus.py 419-420행)
        for i, a in enumerate(self.articles):
            tt[i + 1], tm[i + 1] = self._tokenize(a["title"], tl)
            bt[i + 1], bm[i + 1] = self._tokenize(a["body"], al)
            cat[i + 1] = CATEGORY_DICT["label%d" % int(a["label"])]
        sub = np.full(n, SUBCATEGORY, np.int64)
        out = torch.empty(n, self.model.news_embedding_dim, device=self.device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            args = [torch.from_numpy(x[s:e]).unsqueeze(1).to(self.device) for x in (tt, tm, bt, bm)]
            c = torch.from_numpy(cat[s:e]).unsqueeze(1).to(self.device)
            sc = torch.from_numpy(sub[s:e]).unsqueeze(1).to(self.device)
            rep = self.model.news_encoder(args[0], args[1], None, args[2], args[3], None, c, sc, None)
            out[s:e] = rep.squeeze(1)
        return out

    def _history_rows(self, histories):
        H = np.zeros([len(histories), self.cfg.max_history_num], np.int64)  # PAD=0, 뒤쪽 패딩 (corpus.py 436행)
        for b, hist in enumerate(histories):
            rows = [self.idx_of[a] for a in hist if a in self.idx_of][-self.cfg.max_history_num:]
            H[b, :len(rows)] = rows
        return torch.from_numpy(H).to(self.device)

    def _candidates(self, candidate_article_ids):
        key = tuple(candidate_article_ids)
        if self._cand_cache[0] != key:
            rows = torch.tensor([self.idx_of[a] for a in key], device=self.device)
            cand_repr = self.news_repr[rows]                                # [C, 900]
            q = self.model.user_encoder.Q(cand_repr)                        # [C, 400]
            self._cand_cache = (key, cand_repr, q)
        return self._cand_cache[1], self._cand_cache[2]

    @torch.no_grad()
    def rank_batch(self, histories, candidate_article_ids):
        ue = self.model.user_encoder
        cand_repr, q = self._candidates(candidate_article_ids)
        out = np.empty([len(histories), len(candidate_article_ids)], np.float32)
        for s in range(0, len(histories), self.user_batch):
            hist = self._history_rows(histories[s:s + self.user_batch])
            B = hist.size(0)
            x = torch.cat([self.news_repr[hist], self._user_node.unsqueeze(0).expand(B, -1, -1)], dim=1)
            gcn = ue.graph_sage(x, self._edge_index)[:, :self.cfg.max_history_num]   # [B, 50, 900]
            K = ue.K(gcn)                                                            # [B, 50, 400]
            a = torch.einsum("bhd,cd->bhc", K, q) / ue.attention_scalar              # [B, 50, C]
            alpha = torch.softmax(a, dim=1)                                          # softmax over history (원본 dim=1)
            g = torch.einsum("bhd,cd->bhc", gcn, cand_repr)                          # [B, 50, C] = gcn·cand
            out[s:s + B] = (alpha * g).sum(dim=1).float().cpu().numpy()              # dot_product click predictor
        return out

    # --- 검증: user_encoder.forward 원본 경로와 점수 대조 ---
    @torch.no_grad()
    def _rank_reference(self, histories, candidate_article_ids):
        """학습 forward 그대로: 배치 32 고정, user_encoder가 내부에서 news_encoder 재호출."""
        assert len(histories) == TRAIN_BATCH
        tl, al, hn = self.cfg.max_title_length, self.cfg.max_abstract_length, self.cfg.max_history_num
        tt = np.zeros([TRAIN_BATCH, hn, tl], np.int64); tm = np.zeros([TRAIN_BATCH, hn, tl], bool)
        bt = np.zeros([TRAIN_BATCH, hn, al], np.int64); bm = np.zeros([TRAIN_BATCH, hn, al], bool)
        cat = np.zeros([TRAIN_BATCH, hn], np.int64); tm[:, :, 0] = True; bm[:, :, 0] = True
        by_pos = {self.idx_of[a["article_id"]]: a for a in self.articles}
        for b, hist in enumerate(histories):
            rows = [self.idx_of[a] for a in hist if a in self.idx_of][-hn:]
            for i, r in enumerate(rows):
                art = by_pos[r]
                tt[b, i], tm[b, i] = self._tokenize(art["title"], tl)
                bt[b, i], bm[b, i] = self._tokenize(art["body"], al)
                cat[b, i] = CATEGORY_DICT["label%d" % int(art["label"])]
        sub = np.zeros([TRAIN_BATCH, hn], np.int64)
        dev = self.device
        t = lambda x: torch.from_numpy(x).to(dev)
        cand_repr, _ = self._candidates(candidate_article_ids)
        cand = cand_repr.unsqueeze(0).expand(TRAIN_BATCH, -1, -1)
        mask = t((tt.sum(2) > 0))
        user_repr = self.model.user_encoder(t(tt), t(tm), None, t(bt), t(bm), None, None, t(cat), t(sub),
                                            mask, None, None, None, None, cand)
        return (user_repr * cand).sum(dim=2).float().cpu().numpy()


if __name__ == "__main__":
    import argparse
    import random
    import time

    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    rec = CrownRecommender()
    print(f"load+encode {time.time() - t0:.0f}s  device {rec.device}  news_repr {tuple(rec.news_repr.shape)}", flush=True)
    r = rec.news_repr[1:]
    print(f"news_repr std {r.std().item():.4f}  nan {torch.isnan(r).any().item()}", flush=True)
    if a.selftest:
        from crystal.recsys.dataset import iter_users
        random.seed(0)
        users = []
        for u in iter_users():
            users.append(u)
            if len(users) == 2000:
                break
        sample = random.sample(users, TRAIN_BATCH)
        hists = [[h["article_id"] for h in u["history"]] for u in sample]
        cand_ids = [x["article_id"] for x in random.sample(rec.articles, 7)]
        s_wrap = rec.rank_batch(hists, cand_ids)
        s_ref = rec._rank_reference(hists, cand_ids)
        diff = np.abs(s_wrap - s_ref).max()
        rel = diff / (np.abs(s_ref).max() + 1e-9)
        print(f"wrapper vs reference forward: max abs diff {diff:.6g} (rel {rel:.3g})", flush=True)
        s2 = rec.rank_batch(hists, cand_ids)
        print(f"재현성(같은 입력 2회): identical {np.array_equal(s_wrap, s2)}", flush=True)
        print(f"score sample row0 {np.round(s_wrap[0], 3)}  all-equal {bool((s_wrap == s_wrap.flat[0]).all())}  nan {bool(np.isnan(s_wrap).any())}", flush=True)
        assert rel < 1e-3 and not np.isnan(s_wrap).any()
        print("selftest OK", flush=True)
