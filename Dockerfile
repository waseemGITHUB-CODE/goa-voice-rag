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

# Build a starter index at image-build time so the Space boots ready.
# For the full dataset, swap for:  RUN python ingest.py --config en --max 5000
RUN python ingest.py --sample

EXPOSE 7860
CMD ["python", "app.py"]
