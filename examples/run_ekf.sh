#!/usr/bin/env bash
# Run full EKF experiment from YAML (creates run dir, writes history if configured).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/default_experiment.yaml}"

conda run -n leg-odometry python main.py --config "$CONFIG" "$@"
