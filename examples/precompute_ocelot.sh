#!/usr/bin/env bash
# Precompute precomputed_instants.npz for Ocelot (lowstate.csv per sequence).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_ROOT="${DATASET_ROOT:-/path/to/processed_ocelot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./out_precomputed_ocelot}"
ROBOT="${ROBOT:-go2}"

conda run -n leg-odometry python -m leg_odom.features.precompute_contact_instants \
  --dataset-root "$DATASET_ROOT" \
  --dataset-kind ocelot \
  --output-root "$OUTPUT_ROOT" \
  --robot "$ROBOT" \
  "$@"

echo "Manifest: $OUTPUT_ROOT/precompute_manifest.json"
