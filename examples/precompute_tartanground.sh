#!/usr/bin/env bash
# Precompute precomputed_instants.npz for tartanground (imu + one *_bag.csv per sequence).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# EDIT: directory tree containing sequences; EDIT: output tree for npz + manifest
DATASET_ROOT="${DATASET_ROOT:-/path/to/processed_tartanground}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./out_precomputed_tartanground}"
ROBOT="${ROBOT:-anymal}"

conda run -n leg-odometry python -m leg_odom.features.precompute_contact_instants \
  --dataset-root "$DATASET_ROOT" \
  --dataset-kind tartanground \
  --output-root "$OUTPUT_ROOT" \
  --robot "$ROBOT" \
  "$@"

echo "Manifest: $OUTPUT_ROOT/precompute_manifest.json"
