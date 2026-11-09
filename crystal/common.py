"""모듈 공통. is_displayable: 화면·추천 후보·사건 짝에 올릴 수 있는 기사(본문 150단어 이상). 데이터(학습·히스토리)에서는 빼지 않는다."""

MIN_WORDS = 150  # 2026-10-21: 50이었다가 80단어대 기사도 짧다는 피드백으로 150으로 올림 (분포 두 봉우리 ~40/~180)


def is_boilerplate(body):
    """본문이 CSS 보일러플레이트: `.css`로 시작하거나 앞 300자 안에 `{display:` 규칙이 있음."""
    b = body or ""
    return b.lstrip().startswith(".css") or "{display:" in b[:300]


def is_displayable(title, body):
    """빈 본문 / 본문==제목(strip) / 보일러플레이트 / MIN_WORDS 단어 미만이면 False."""
    b = (body or "").strip()
    return b != "" and b != (title or "").strip() and not is_boilerplate(body) and len(b.split()) >= MIN_WORDS
