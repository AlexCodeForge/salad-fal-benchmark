# Depth-Anything-V2 inference service for Salad GPU benchmark.
# Local/CI: MOCK_INFERENCE=1 (default) — no HF download.
# Production Salad: MOCK_INFERENCE=0 MODEL_ID=depth-anything/Depth-Anything-V2-Large-hf

FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MOCK_INFERENCE=1 \
    MODEL_ID=depth-anything/Depth-Anything-V2-Small-hf \
    TORCH_DTYPE=float16 \
    HOST=:: \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY docker/depth/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY docker/depth/app /app/app
COPY docker/depth/entrypoint.py /app/entrypoint.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:8000/health" || curl -fsS -g "http://[::1]:8000/health" || exit 1

CMD ["python", "/app/entrypoint.py"]
