#!/usr/bin/env bash
# Train neural contact model (requires precomputed_instants.npz under precomputed_root in YAML).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-leg_odom/training/nn/default_train_config.yaml}"

conda run -n leg-odometry python -m leg_odom.training.nn.train_contact_nn --config "$CONFIG" "$@"
