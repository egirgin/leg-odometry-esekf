#!/usr/bin/env bash
# GRF-threshold contact replay plot (standalone; no EKF).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SEQ_DIR="${SEQ_DIR:?Set SEQ_DIR to trajectory directory}"
DATASET_KIND="${DATASET_KIND:-tartanground}"
ROBOT="${ROBOT:-anymal}"
SAVE="${SAVE:-}"

args=(--sequence-dir "$SEQ_DIR" --dataset-kind "$DATASET_KIND" --robot-kinematics "$ROBOT")
[[ -n "$SAVE" ]] && args+=(--save "$SAVE")

conda run -n leg-odometry python -m leg_odom.contact.grf_threshold "${args[@]}" "$@"
