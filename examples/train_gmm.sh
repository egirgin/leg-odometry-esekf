#!/usr/bin/env bash
# Fit GMM contact weights from a precomputed_instants.npz tree.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PRECOMPUTED_ROOT="${PRECOMPUTED_ROOT:-./out_precomputed_tartanground}"
OUT_NPZ="${OUT_NPZ:-leg_odom/training/gmm/weights_example.npz}"
ROBOT="${ROBOT:-anymal}"

conda run -n leg-odometry python -m leg_odom.training.gmm.train_gmm \
  --precomputed-root "$PRECOMPUTED_ROOT" \
  --output "$OUT_NPZ" \
  --robot-kinematics "$ROBOT" \
  --history-length 1 \
  "$@"
