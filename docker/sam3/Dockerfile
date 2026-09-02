# SAM3 inference container for Salad GPU benchmark (fal-compatible /v1/sam3).
# Local/CI: MOCK_INFERENCE=1 (default) — no ~10GB HF download.
# Production Salad: MOCK_INFERENCE=0 SAM3_MODEL=facebook/sam3

FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MOCK_INFERENCE=1 \
    SAM3_MODEL=facebook/sam3 \
    SAM3_CONFIG_PATH=/app/configs/sam3-model.yaml \
    HF_HOME=/cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY docker/sam3/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY configs/sam3-model.yaml /app/configs/sam3-model.yaml
COPY docker/sam3/app /app/app
COPY docker/sam3/entrypoint.py /app/entrypoint.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:8000/health" || curl -fsS -g "http://[::1]:8000/health" || exit 1

CMD ["python", "/app/entrypoint.py"]
