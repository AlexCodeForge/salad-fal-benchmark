#!/usr/bin/env bash
# Record fal_replay JSON (+ masks/depth) for a fixture via teselio-engine-rs analyze-cli.
#
# Usage:
#   FAL_KEY=... ./scripts/record-fal-fixture.sh terminados-04
#
# Writes to: fixtures/<id>/fal_replay/ (manifest.json, sam-3-*.json, depth-anything-v2.json, binaries)
#
# SSOT recorder (engine-rs):
#   FAL_RECORD=1 FAL_KEY=... teselio-analyze-cli fal segment \
#     --image fixtures/<id>/photo.jpg \
#     --replay-dir fixtures/<id>/fal_replay
#
# Requires photo.jpg present (run sync-lab-images.sh first for Tier B scenes).

set -euo pipefail

FIXTURE_ID="${1:-}"
if [[ -z "$FIXTURE_ID" ]]; then
  echo "usage: $0 <fixture-id>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PHOTO="$ROOT/fixtures/$FIXTURE_ID/photo.jpg"
REPLAY_DIR="$ROOT/fixtures/$FIXTURE_ID/fal_replay"

if [[ ! -f "$PHOTO" ]]; then
  echo "Missing photo: $PHOTO (run sync-lab-images.sh first)" >&2
  exit 1
fi

mkdir -p "$REPLAY_DIR"

# TODO: wire to teselio-engine-rs analyze-cli when recording Tier B baselines.
echo "record-fal-fixture.sh: stub for $FIXTURE_ID -> $REPLAY_DIR" >&2
echo "Set FAL_RECORD=1 and FAL_KEY, then run engine-rs fal segment (see script header)." >&2
exit 0
