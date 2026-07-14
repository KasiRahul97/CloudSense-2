# ── CloudSense Inference Server ──────────────────────────────────────────────
# Build from the REPO ROOT so src/ and model_export/ are in the build context:
#
#   python src/main.py                      # train + export model_export/
#   docker build -f deploy/inference/Dockerfile -t cloudsense-api .
#   docker run -p 8000:8000 \
#       -e CLOUDSENSE_CORS_ORIGINS=http://localhost:8501 cloudsense-api
#
# (CPU-only; no GPU needed for inference.)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache).
COPY deploy/inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared training/serving code, the trained model, and the API.
COPY src/ ./src/
COPY model_export/ ./model_export/
COPY deploy/inference/app.py ./app.py

EXPOSE 8000

ENV CLOUDSENSE_SRC=/app/src \
    MODEL_DIR=/app/model_export \
    PORT=8000

# 1 worker keeps the model in a single process (fine for a t2.micro / free tier).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
