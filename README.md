# salad-fal-benchmark

Standalone benchmark comparing **fal.ai** vs **self-hosted Salad GPU** for the Teselio Rust production analyze scope: one image upload, 4× SAM-3 (`wall`, `molding`, `mullion`, `floor`), Depth-Anything v2, and local CPU post-process.

Measures latency, cost, and quality (mask IoU vs fal baseline, depth correlation).

See `configs/stages.yaml` for the Rust-prod stage DAG.

## Quickstart

### Install

```bash
cd /home/code/salad-fal-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional — replay needs no keys
```

### Replay benchmark (Tier A — no API keys)

Runs the fal backend against checked-in `fixtures/*/fal_replay` manifests (VLM skipped):

```bash
bench run --backend fal --fixture terminados-02 --replay
pytest tests/test_fal_replay.py -q
```

### Local mock stack (unified analyze)

Starts a single Docker service on **8001** with `MOCK_INFERENCE=1`, waits for `/health`, then runs one Salad backend iteration:

```bash
chmod +x scripts/run-local-stack.sh
./scripts/run-local-stack.sh
```

Manual control:

```bash
docker compose up -d --build
curl -sS http://127.0.0.1:8001/health

export SALAD_ANALYZE_GATEWAY_URL=http://127.0.0.1:8001
bench run --backend salad --fixture terminados-02 --runs 1
```

Images build from the **repo root** (`docker build -f docker/analyze/Dockerfile .`).

### Salad Cloud deploy

Build and push the unified analyze image, then deploy one SCE container group (requires Salad + registry credentials):

```bash
export DOCKER_USER=your-dockerhub-user
export TAG=dev
docker build -f docker/analyze/Dockerfile -t "${DOCKER_USER}/salad-fal-analyze:${TAG}" .
docker push "${DOCKER_USER}/salad-fal-analyze:${TAG}"

export SALAD_API_KEY=...
export SALAD_ORGANIZATION_NAME=...
export SALAD_PROJECT_NAME=...
export SALAD_ANALYZE_IMAGE="${DOCKER_USER}/salad-fal-analyze:${TAG}"

python scripts/salad/preflight.py
python scripts/salad/deploy.py
python scripts/salad/wait_ready.py
python scripts/salad/print_gateways.py   # export SALAD_ANALYZE_GATEWAY_URL from output

bench run --backend salad --fixture terminados-02 --runs 3
```

Dry-run deploy payloads without API calls:

```bash
python scripts/salad/deploy.py --dry-run
```

Teardown (removes bench-analyze and legacy bench-sam3 / bench-depth groups):

```bash
python scripts/salad/destroy.py
```

## Docs

- `docker/README.md` — image build, API routes, env vars
- `configs/pricing.yaml` — fal list prices and RTX 3090 GPU-hour rate ($0.09/hr)
