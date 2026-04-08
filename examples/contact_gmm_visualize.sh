#!/usr/bin/env bash
# GMM+HMM contact visualization (standalone; no EKF). Use --mode offline or online with --pretrained-path.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SEQ_DIR="${SEQ_DIR:?Set SEQ_DIR to trajectory directory}"
DATASET_KIND="${DATASET_KIND:-tartanground}"
ROBOT="${ROBOT:-anymal}"
MODE="${MODE:-offline}"
PRETRAINED="${PRETRAINED:-leg_odom/training/gmm/weights_example.npz}"
SAVE="${SAVE:-}"

args=(
  --sequence-dir "$SEQ_DIR"
  --dataset-kind "$DATASET_KIND"
  --robot-kinematics "$ROBOT"
  --mode "$MODE"
)
[[ "$MODE" == "online" ]] && args+=(--pretrained-path "$PRETRAINED")
[[ -n "$SAVE" ]] && args+=(--save "$SAVE")

conda run -n leg-odometry python -m leg_odom.contact.gmm_hmm.visualize "${args[@]}" "$@"
