# 키워드·문장 색칠 파이프라인 (crystal/keyword)

데모 기사 화면(B-2)에 칠해지는 **단어**와 **문장**을 만드는 코드입니다. 2026-11-03 기준 데모는 아래 "현재 데모" 방식을 씁니다.

## 현재 데모: ContextCite 문장 + 단서 사전 단어

```
articles.jsonl ─┬─ mask_names.py ──► masked.csv ──► roberta_veracity/train_masked.py ──► fold0_masked/best
                │                                                              │
                └──────────────────────────── contextcite.py (문장 기여도, 상위 5개) ◄──┘
                                                      │
                                             cc_sentences_top5.jsonl
                                                      │
                                   build_highlights.py  +  lexicon.py (단서 사전)
                                                      │
                                      cc_keywords.jsonl  (+ 설명 캐시)
                                                      │
                                CRYSTAL_KEYWORDS=cc 로 웹 서버 실행 → 기사 화면 색칠
```

| 단계 | 파일 | 하는 일 |
| --- | --- | --- |
| 1 | `mask_names.py` | spaCy로 인물명(PERSON)을 `[NAME]`으로 가린 제목·본문 csv를 만든다. 원래 모델이 특정 인물명을 가짜 신호로 크게 쓰기 때문. |
| 2 | `../../roberta_veracity/train_masked.py` | 가린 텍스트로 fold0 RoBERTa를 다시 학습한다(기여도 계산 전용, test acc 0.8489). **판정은 원래 10-fold 모델을 그대로 쓴다.** |
| 3 | `contextcite.py` | ContextCite 방식을 분류기에 맞게 직접 구현. 문장을 무작위로 지운 입력 256개의 판정 log-odds를 LASSO로 맞춰 문장별 기여도를 구하고 상위 5개를 원문 좌표로 저장. |
| 4 | `lexicon.py` | 사람이 정한 단서 사전. FAKE 목록(익명 출처·추측·과장·비속어·감정 단어) / TRUE 목록(성명·대변인·확인·공식 기록·본인 게시·인터뷰·날짜·언론사)과 단어별 설명 템플릿. |
| 5 | `build_highlights.py` | 문장(본문 길이에 따라 1~5개, 붙어 있으면 다음 순위로 대체) + 단어(판정 라벨 쪽 목록만, 문장 안 단어 제외, 최대 15개) → `cc_keywords.jsonl`. 단어 설명은 템플릿으로 캐시에 넣는다. |

실행 순서(전부 프로젝트 루트에서, Windows PowerShell은 앞에 `$env:KMP_DUPLICATE_LIB_OK='TRUE'`):

```
python -X utf8 crystal/keyword/mask_names.py
python -X utf8 roberta_veracity/train_masked.py
python -X utf8 crystal/keyword/contextcite.py --model roberta_veracity/folds/fold0_masked/best --ids ids.json --masked crystal/data/keyword/masked.csv
python -X utf8 crystal/keyword/build_highlights.py
$env:CRYSTAL_KEYWORDS='cc'; python -X utf8 -m uvicorn crystal.web.app:app --port 8765
```

`ids.json`은 대상 article_id 목록(JSON 배열)입니다. 데모에서는 두 데모 사용자의 피드 10페이지 + 관련 기사 393건을 썼습니다.

## 단서 후보 보기: IG / LIME (참고용)

| 파일 | 하는 일 |
| --- | --- |
| `ig.py`, `expand.py`, `run.py` | Integrated Gradients 단어 기여도 + spaCy 구 확장(초기 버전, `CRYSTAL_KEYWORDS` 기본값 `ig`). |
| `lime_words.py` | LIME 단어 기여도 top-K를 본문 좌표로 저장(`lime_masked_keywords.jsonl` 형식). |

IG/LIME 결과는 어떤 단어가 판정을 끌고 가는지 보는 데 썼고, 최종 색칠 단어는 그 후보를 보고 사람이 정한 `lexicon.py` 사전으로 매칭합니다.

## 산출 파일 형식 (`cc_keywords.jsonl`, 한 줄 = 한 기사)

```json
{"article_id": "gossipcop-7455413235", "final_label": 1, "method": "contextcite+lexicon",
 "keywords": [{"phrase": "rumours", "field": "body", "start": 200, "end": 207, "score": 0.6, "sentence": "...", "kind": "lexicon"},
              {"phrase": "(sic),’ wrote one fan ...", "field": "body", "start": 1756, "end": 1839, "score": 1.0, "sentence": "...", "kind": "sentence"}]}
```

`kind`가 `sentence`면 문장 색칠, `lexicon`이면 단어 색칠입니다. 설명 문구는 `crystal/data/rationale_cache_lime_masked.jsonl`에 `(article_id, kind=keyword, key=phrase)`로 저장되고, 없으면 웹에서 Gemini가 생성합니다.
