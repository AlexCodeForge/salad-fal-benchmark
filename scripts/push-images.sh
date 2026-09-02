#!/usr/bin/env bash
# Build and tag SAM3 + depth Docker images for Salad deploy.
#
# Requires DOCKER_USER — your Docker Hub username (or registry namespace prefix).
# Example:
#   export DOCKER_USER=myuser
#   ./scripts/push-images.sh
#   docker push "${DOCKER_USER}/salad-fal-sam3:${TAG}"
#   docker push "${DOCKER_USER}/salad-fal-depth:${TAG}"
#
# Then set deploy env:
#   export SALAD_SAM3_IMAGE="${DOCKER_USER}/salad-fal-sam3:${TAG}"
#   export SALAD_DEPTH_IMAGE="${DOCKER_USER}/salad-fal-depth:${TAG}"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${TAG:-dev}"

if [[ -z "${DOCKER_USER:-}" ]]; then
  echo "ERROR: Set DOCKER_USER to your Docker Hub username (or registry namespace)." >&2
  echo "  export DOCKER_USER=myuser" >&2
  exit 1
fi

SAM3_IMAGE="${DOCKER_USER}/salad-fal-sam3:${TAG}"
DEPTH_IMAGE="${DOCKER_USER}/salad-fal-depth:${TAG}"

echo "Building SAM3 image: ${SAM3_IMAGE}"
docker build -t "${SAM3_IMAGE}" -f "${ROOT}/docker/sam3/Dockerfile" "${ROOT}"

echo "Building depth image: ${DEPTH_IMAGE}"
docker build -t "${DEPTH_IMAGE}" -f "${ROOT}/docker/depth/Dockerfile" "${ROOT}"

echo ""
echo "Built:"
echo "  ${SAM3_IMAGE}"
echo "  ${DEPTH_IMAGE}"
echo ""
echo "Push to registry:"
echo "  docker push ${SAM3_IMAGE}"
echo "  docker push ${DEPTH_IMAGE}"
echo ""
echo "Deploy env:"
echo "  export SALAD_SAM3_IMAGE=${SAM3_IMAGE}"
echo "  export SALAD_DEPTH_IMAGE=${DEPTH_IMAGE}"
