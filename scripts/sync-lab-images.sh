#!/usr/bin/env bash
# Sync Tier B fixture photos from teselio-engine-py lab inputs into fixtures/<id>/photo.jpg.
#
# Usage:
#   ./scripts/sync-lab-images.sh              # all Tier B fixtures missing photo.jpg
#   ./scripts/sync-lab-images.sh terminados-04  # single fixture id
#
# Source SSOT: /home/code/teselio-engine-py/lab/storage/assets/inputs/<lab_image>
# Target layout: fixtures/<id>/photo.jpg (see configs/fixtures.yaml)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAB_INPUTS="${LAB_INPUTS:-/home/code/teselio-engine-py/lab/storage/assets/inputs}"

if [[ ! -d "$LAB_INPUTS" ]]; then
  echo "Lab inputs directory not found: $LAB_INPUTS" >&2
  echo "Set LAB_INPUTS or run on the Teselio VPS with engine-py checked out." >&2
  exit 1
fi

# TODO: invoke bench/fixtures/loader.py or parse configs/fixtures.yaml tier_b rows.
# For each tier_b entry: copy "$LAB_INPUTS/<lab_image>" -> "$ROOT/fixtures/<id>/photo.jpg"
echo "sync-lab-images.sh: stub — implement Tier B copy loop (see configs/fixtures.yaml tier_b)." >&2
exit 0
