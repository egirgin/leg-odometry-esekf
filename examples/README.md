# Example commands

Shell snippets for **conda env `leg-odometry`** and repo root **`async_ekf_workspace`**. Edit paths before running.

Order when using learned contact models:

1. **Precompute** → 2. **Train** (NN and/or GMM) → 3. **Point experiment YAML** at weights → 4. **`main.py`**.

For **GRF-threshold** contact only, skip steps 1–2.

| Script | Purpose |
| ------ | ------- |
| [`precompute_tartanground.sh`](precompute_tartanground.sh) | Build npz tree for imu+bag layouts |
| [`precompute_ocelot.sh`](precompute_ocelot.sh) | Build npz tree for Ocelot `lowstate.csv` |
| [`train_nn.sh`](train_nn.sh) | CNN/GRU training from npz |
| [`train_gmm.sh`](train_gmm.sh) | GMM fit from npz |
| [`contact_grf_plot.sh`](contact_grf_plot.sh) | GRF-threshold replay plot (no EKF) |
| [`contact_gmm_visualize.sh`](contact_gmm_visualize.sh) | GMM+HMM replay plot (no EKF) |
| [`run_ekf.sh`](run_ekf.sh) | Full experiment via `main.py` |

More detail: [Features README](../leg_odom/features/README.md), [Training README](../leg_odom/training/README.md), [Contact README](../leg_odom/contact/README.md), [root README](../README.md).
