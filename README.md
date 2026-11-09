# CRYSTAL

진위(veracity) 기반 설명형 뉴스 추천 데모. PRISM(CIKM'26 demo)의 정치 성향 축을 가짜뉴스 진위로 바꾼 것.

- 진위 판정: RoBERTa 10-fold 교차 예측 (out-of-fold, temperature scaling). GossipCop 17,527건 acc 0.8725 / F1(fake) 0.7319.
  구글 Fact Check Tools API는 시도만 했고(PolitiFact 597건 중 24.3%만 매칭, 판정 이유 없음) 데모에는 쓰지 않음.
- 판정 근거 색칠: ContextCite 방식 문장 기여도(직접 구현) + 사람이 정한 단서 사전 단어. IG/LIME은 단서 후보 확인용. → `crystal/keyword/README.md`
- 같은 사건 매칭: SBERT(all-MiniLM-L6-v2) + HDBSCAN, 군집 내 코사인 ≥ 0.7
- 추천: CROWN (crown-www25)
- 설명 문장: 캐시 우선, 없으면 Gemini(gemini-3.5-flash-lite, verbalizer 역할만)

## 폴더

| 폴더 | 내용 |
| --- | --- |
| `roberta_veracity/` | RoBERTa 학습 (`train.py` 단일, `train_kfold.py` 10-fold, `train_masked.py` 인물명 마스킹 모델) |
| `crystal/veracity/` | 모델 로드·전건 판정·온도 보정 |
| `crystal/keyword/` | 색칠 파이프라인 (문장·단어·설명) |
| `crystal/event/` | 같은 사건 군집·반대 진위 짝 |
| `crystal/recsys/` | CROWN 추천·피드·진위 분포 |
| `crystal/rationale/` | Gemini 설명 생성·캐시 |
| `crystal/web/` | FastAPI 웹 (피드 / 기사 / 비교 화면) |

## 실행

데이터 파일(`crystal/data/*.jsonl`)과 가중치는 저장소에 없음. 파이프라인 산출물을 받아 넣은 뒤:

```
set GEMINI_API_KEY=...        # 또는 .env (없으면 캐시에 있는 설명만 표시)
set CRYSTAL_KEYWORDS=cc       # ContextCite+사전 색칠 (기본값 ig = 초기 IG 색칠)
python -X utf8 -m uvicorn crystal.web.app:app --port 8765
```

## Docker

데이터·캐시가 들어 있는 데모 이미지: `docker pull yusuklee/crystal:latest`

```
docker run -e GEMINI_API_KEY=... -p 8765:8765 yusuklee/crystal:latest
```

직접 빌드: `docker build -t crystal .` (Dockerfile이 `CRYSTAL_KEYWORDS=cc`, flash-lite 기본값을 넣어 둠)
