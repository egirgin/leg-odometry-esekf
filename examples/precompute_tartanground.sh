#!/usr/bin/env bash
# Precompute precomputed_instants.npz for tartanground (imu + one *_bag.csv per sequence).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# EDIT: YAML with dataset_root, output_root, dataset_kind: tartanground, robot, labels, overwrite, ...
PRECOMPUTE_CONFIG="${PRECOMPUTE_CONFIG:-$REPO_ROOT/leg_odom/features/default_precompute_config.yaml}"

conda run -n leg-odometry python -m leg_odom.features.precompute_contact_instants \
  --config "$PRECOMPUTE_CONFIG" \
  "$@"

echo "See precompute_manifest.json under output_root from your YAML"
