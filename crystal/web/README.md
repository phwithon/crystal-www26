# CRYSTAL Web (scaffold)

- Run (from the project root): `python -X utf8 -m uvicorn crystal.web.app:app --port 8765` → http://127.0.0.1:8765/
- Data: uses the real data if `crystal/data/` has articles/users/feeds and the test articles have `keywords` filled in; otherwise falls back to `crystal/web/sample/`. Force with the env var `CRYSTAL_DATA=real|sample|auto` (default `auto`).
- Env vars: `CRYSTAL_SEED_USER` sets the default user; `CRYSTAL_KEYWORDS` sets the highlighting mode (default `ig`; `cc` = ContextCite + cue lexicon); `CRYSTAL_SHOW_FEED_LABELS=1` shows the feed label chips; `CRYSTAL_SHOW_LEVEL_CAPTION=1` shows the goal caption at the top of the page (hidden by default).
- API: `GET /api/settings`, `GET /api/feed?user_id=&page=`, `GET /api/article/{id}`, `GET /api/compare?a=&b=`, `GET /api/search?q=&limit=`, `POST /api/rationale {article_id, kind, key}`, `POST /api/log-click {user_id, article_id}`, `GET /api/demo-users`, `POST /api/reset-user {user_id}`. Views: `/`, `/article/{id}`, `/article/{id}?compare={b}`.
- Regenerate the sample: `python -X utf8 crystal/web/sample/make_sample.py`. Once the real data is wired up, `sample/` can be deleted (only the `SAMPLE_DIR` branch in `data.py` remains).
- `crystal/web/_cache/` holds the users/feeds line-offset index (auto-generated, safe to delete).
