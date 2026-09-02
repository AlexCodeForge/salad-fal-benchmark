# Docker images — Salad self-hosted inference

Container images for the **salad-fal-benchmark** harness. Each service exposes FastAPI on **`[::]:8000`** with a `GET /health` readiness probe and fal-shaped JSON inference routes.

| Image | Route | fal endpoint parity |
|-------|-------|---------------------|
| `docker/sam3/` | `POST /v1/sam3` | `fal-ai/sam-3/image` |
| `docker/depth/` | `POST /v1/depth` | `fal-ai/image-preprocessors/depth-anything/v2` |

---

## Depth (`docker/depth/`)

Depth-Anything V2 depth map service. Output is a **single-channel uint8 grayscale PNG** with per-image min–max normalization (official `run.py --grayscale` parity). **Higher pixel value = nearer camera.**

### Build

```bash
cd /home/code/salad-fal-benchmark
docker build -t salad-fal-depth:dev -f docker/depth/Dockerfile .
```

### Run (mock — no GPU, no HF weights)

```bash
docker run --rm -p 8000:8000 \
  -e MOCK_INFERENCE=1 \
  salad-fal-depth:dev
```

Returns a deterministic left→right gradient uint8 PNG for local smoke tests.

### Run (GPU inference)

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e MODEL_ID=depth-anything/Depth-Anything-V2-Large-hf \
  -e TORCH_DTYPE=float16 \
  salad-fal-depth:dev
```

| `MODEL_ID` | Use |
|------------|-----|
| `depth-anything/Depth-Anything-V2-Small-hf` | Fast dev (default in Dockerfile) |
| `depth-anything/Depth-Anything-V2-Large-hf` | fal parity / production benchmark |

Large-hf is **CC-BY-NC-4.0** — internal benchmark only.

### Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Depth-Anything-V2-Small-hf` | Hugging Face model id |
| `TORCH_DTYPE` | `float16` | `float16` on CUDA, auto `float32` on CPU |
| `MOCK_INFERENCE` | (unset) | `1` → skip model load; gradient fake depth |
| `RETURN_IMAGE_URL` | (unset) | `1` → response uses `image.url` (data-URL) instead of `content` base64 |
| `IMAGE_DOWNLOAD_TIMEOUT_S` | `60` | Timeout for fetching `image_url` |
| `HOST` / `PORT` | `::` / `8000` | Set in Dockerfile CMD |

### API

**Health**

```bash
curl -sS "http://127.0.0.1:8000/health"
```

**Depth (JSON + `image_url`)**

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/depth" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/room.jpg"}'
```

**Depth (JSON + base64)**

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/depth" \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\":\"$(base64 -w0 photo.jpg)\"}"
```

**Depth (multipart file)**

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/depth" \
  -F "image=@photo.jpg"
```

**Response (fal-shaped)**

```json
{
  "image": {
    "content_type": "image/png",
    "width": 1920,
    "height": 1080,
    "content": "<base64 png>"
  }
}
```

Wire dimensions match the **input photo** (bilinear resize via HF `post_process_depth_estimation`). Client benchmark code may still bilinear-resize to source H×W when fal returns a different size.

### Salad deploy notes

- Gateway port **8000**, readiness `GET /health` after model warm-load (allow ~120s startup on cold pull).
- Resources: GPU, CPU 4, RAM 8192–12288 MB, storage ~25 GB (HF cache).
- Set `SALAD_DEPTH_GATEWAY_URL` from `scripts/salad/print_gateways.py` after deploy (orchestrator-owned).

### Local dev without Docker

```bash
cd docker/depth
pip install -r requirements.txt
MOCK_INFERENCE=1 python -m uvicorn app.main:app --host :: --port 8000
```

---

## SAM3 (`docker/sam3/`)

See `docker/sam3/` package (parallel implementer). Section added when that image lands.
