#!/usr/bin/env bash
# Start mock SAM3 + depth containers and run one Salad backend benchmark against localhost gateways.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SAM3_HEALTH="http://127.0.0.1:8001/health"
DEPTH_HEALTH="http://127.0.0.1:8002/health"
export SALAD_SAM3_GATEWAY_URL="http://127.0.0.1:8001"
export SALAD_DEPTH_GATEWAY_URL="http://127.0.0.1:8002"

wait_health() {
  local url=$1
  local name=$2
  echo "Waiting for ${name} at ${url} ..."
  for _ in $(seq 1 90); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "${name} ready"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: timeout waiting for ${name}" >&2
  docker compose ps >&2 || true
  exit 1
}

bench_cmd() {
  if [[ -x "${ROOT}/.venv/bin/bench" ]]; then
    echo "${ROOT}/.venv/bin/bench"
  elif command -v bench >/dev/null 2>&1; then
    command -v bench
  else
    echo ""
  fi
}

run_bench() {
  local bin
  bin="$(bench_cmd)"
  if [[ -n "${bin}" ]]; then
    "${bin}" run --backend salad --fixture terminados-02 --runs 1 "$@"
  else
    python3 -m bench run --backend salad --fixture terminados-02 --runs 1 "$@"
  fi
}

echo "Starting local stack (sam3:8001, depth:8002, MOCK_INFERENCE=1) ..."
docker compose up -d --build

wait_health "${SAM3_HEALTH}" "sam3"
wait_health "${DEPTH_HEALTH}" "depth"

echo "Health checks:"
curl -fsS "${SAM3_HEALTH}"
echo ""
curl -fsS "${DEPTH_HEALTH}"
echo ""

echo "Running bench against localhost gateways ..."
run_bench
