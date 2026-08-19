# Hugging Face Spaces (SDK: Docker) — exposes port 7860
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/hf \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PORT=7860

WORKDIR /app

# System deps for faiss / onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The repo ships a pre-built real MSMARCO-XI index (data/index, ~9MB), so the
# Space boots with real data instantly. If it's ever missing, fall back to the
# tiny offline sample so the app still starts.
RUN test -f data/index/index.faiss || python ingest.py --sample

EXPOSE 7860
CMD ["python", "app.py"]
