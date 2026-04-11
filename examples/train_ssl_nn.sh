#!/usr/bin/env bash
# Train self-supervised neural backbone (requires precomputed_instants.npz under precomputed_root in YAML).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-leg_odom/training/ssl_nn/default_ssl_config.yaml}"

conda run -n leg-odometry python -m leg_odom.training.ssl_nn.train_ssl_nn --config "$CONFIG" "$@"
