#!/usr/bin/env bash
# Start mock analyze container and run one Salad backend benchmark against localhost gateway.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ANALYZE_HEALTH="http://127.0.0.1:8001/health"
export SALAD_ANALYZE_GATEWAY_URL="http://127.0.0.1:8001"
export SALAD_GATEWAY_URL="http://127.0.0.1:8001"

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

echo "Stopping any existing stack (including orphan containers) ..."
docker compose down --remove-orphans

echo "Starting local stack (analyze:8001, MOCK_INFERENCE=1) ..."
docker compose up -d --build

wait_health "${ANALYZE_HEALTH}" "analyze"

echo "Health check:"
curl -fsS "${ANALYZE_HEALTH}"
echo ""

echo "Running bench against localhost gateway ..."
run_bench
