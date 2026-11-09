# CRYSTAL 웹 (골격)

- 실행 (프로젝트 루트에서): `python -X utf8 -m uvicorn crystal.web.app:app --port 8765` → http://127.0.0.1:8765/
- 데이터: `crystal/data/`에 articles·users·feeds가 있고 test 기사에 `keywords`가 채워져 있으면 실데이터, 아니면 `crystal/web/sample/`. 환경 변수 `CRYSTAL_DATA=real|sample|auto`(기본 auto)로 강제. `CRYSTAL_SEED_USER`로 기본 유저, `CRYSTAL_SHOW_FEED_LABELS=0`으로 피드 라벨 칩 숨김.
- API: `GET /api/feed?user_id=&page=`, `GET /api/article/{id}`, `GET /api/compare?a=&b=`, `GET /api/search?q=`, `POST /api/rationale {article_id, kind, key}`, `POST /api/log-click`, `GET /api/settings`. 화면: `/`, `/article/{id}`, `/article/{id}?compare={b}`.
- 샘플 다시 만들기: `python -X utf8 crystal/web/sample/make_sample.py`. 실데이터 연결 후 `sample/`은 지워도 된다 (`data.py`의 SAMPLE_DIR 분기만 남음).
- `crystal/web/_cache/`는 users·feeds 줄 오프셋 색인(자동 생성, 지워도 됨).
