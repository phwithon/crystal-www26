FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn google-genai
COPY crystal/ crystal/
ENV CRYSTAL_DATA=real PYTHONUTF8=1 CRYSTAL_KEYWORDS=cc CRYSTAL_GEMINI_MODEL=gemini-3.5-flash-lite
# 실행 시: docker run -e GEMINI_API_KEY=<키> -p 8765:8765 crystal  (.env는 이미지에 안 들어감)
EXPOSE 8765
CMD ["uvicorn", "crystal.web.app:app", "--host", "0.0.0.0", "--port", "8765"]
