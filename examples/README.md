# Leg Odometry Framework Examples

This folder contains shell examples for common workflows.

Assumptions:
- conda env: `leg-odometry`
- working directory: repository root

## Recommended Order

1. Preprocess (if training is needed)
2. Train detector models
3. Run detector-only visualization (optional)
4. Run full EKF experiment

For pure GRF-threshold EKF runs, preprocessing and training are optional.

## Scripts

| Script | Purpose |
| ------ | ------- |
| [`precompute_tartanground.sh`](precompute_tartanground.sh) | Preprocessing example for Tartanground-style sequences |
| [`precompute_ocelot.sh`](precompute_ocelot.sh) | Preprocessing example for Ocelot-style sequences |
| [`train_nn.sh`](train_nn.sh) | Neural detector training from precomputed bundles |
| [`train_gmm.sh`](train_gmm.sh) | GMM detector training from precomputed bundles |
| [`contact_grf_plot.sh`](contact_grf_plot.sh) | Detector-only GRF threshold replay plot |
| [`contact_gmm_visualize.sh`](contact_gmm_visualize.sh) | Detector-only GMM replay plot |
| [`run_ekf.sh`](run_ekf.sh) | Run full EKF pipeline using experiment YAML |

## Documentation Links

- [`../README.md`](../README.md)
- [`../leg_odom/features/README.md`](../leg_odom/features/README.md)
- [`../leg_odom/training/README.md`](../leg_odom/training/README.md)
- [`../leg_odom/contact/README.md`](../leg_odom/contact/README.md)
