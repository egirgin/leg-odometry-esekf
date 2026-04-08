# Training: neural and GMM contact models

This package fits **contact classifiers** and **GMM+HMM** weights from **precomputed** per-sequence features. It does **not** require importing the EKF core; it uses the same **dataset kinds**, **kinematics**, and **merged frames** as the main pipeline.

## Layout

| Path | Role |
| ---- | ---- |
| [`nn/`](nn/) | CNN/GRU training, precomputed I/O, labels via detector **replay**, configs. |
| [`gmm/`](gmm/) | Offline fit of 2-component GMM (+ optional post-train replay plot). |

## Dependency: precompute first

Both tracks expect a directory tree of **`precomputed_instants.npz`** files (see [`leg_odom/features/README.md`](../features/README.md)).

```text
precompute_contact_instants  →  precomputed_instants.npz (tree)
                                      ↓
                          train_contact_nn  /  train_gmm
                                      ↓
                    .pt + meta + scaler  |  .npz weights
                                      ↓
                    experiment YAML (contact.neural / contact.gmm)
```

## Script: neural contact training

**Entry point:** `python -m leg_odom.training.nn.train_contact_nn`

**Purpose:** Load npz bundles from `dataset.precomputed_root` in YAML; build stance labels (e.g. GRF replay or GMM+HMM replay); train CNN or GRU; write checkpoint + `_meta.json` + `_scaler.npz`.

### Config

- Default template: [`nn/default_train_config.yaml`](nn/default_train_config.yaml)
- Ocelot + Go2 example: [`nn/default_train_config_ocelot_go2.yaml`](nn/default_train_config_ocelot_go2.yaml)
- **`--config`** path to your YAML (see [`nn/config.py`](nn/config.py) for validation rules).

Key sections: `dataset.kind`, `dataset.precomputed_root`, `robot.kinematics`, `architecture` (`cnn`/`gru`), `features.fields`, `labels.method`, `training.*`, `model.window_size`.

### Supporting modules (library, not CLIs)

| Module | Role |
| ------ | ---- |
| [`nn/discovery.py`](nn/discovery.py) | Discover sequence dirs for precompute/train routing. |
| [`nn/io_labels.py`](nn/io_labels.py) | Dispatch frame load / discovery by `dataset.kind`. |
| [`nn/sequence_frames.py`](nn/sequence_frames.py) | Tartanground vs Ocelot merged-frame loaders for training. |
| [`nn/label_timelines.py`](nn/label_timelines.py) | Pseudo-labels via `build_leg_odometry_dataset` + contact replay. |
| [`nn/precomputed_io.py`](nn/precomputed_io.py) | Load/save contract for npz bundles. |
| [`nn/data.py`](nn/data.py) | PyTorch `Dataset` assembly, sliding windows. |
| [`nn/models.py`](nn/models.py) | `ContactCNN`, `ContactGRU`. |

## Script: GMM training

**Entry point:** `python -m leg_odom.training.gmm.train_gmm`

**Purpose:** Stack sliding-window features from all `precomputed_instants.npz` under `--precomputed-root`, fit ordered 2-GMM, save **`.npz`** for online HMM (`train_gmm.py` docstring lists full CLI).

### Common arguments

| Argument | Description |
| -------- | ----------- |
| `--precomputed-root` | Tree containing `precomputed_instants.npz`. |
| `--output` | Output `.npz` path (default under `leg_odom/training/gmm/`). |
| `--robot-kinematics` | `anymal` or `go2` (must match precompute). |
| `--feature-fields` | Comma-separated instant field names. |
| `--history-length` | Default `1` (required for offline GMM mode in EKF). |
| `--max-sequences` | Optional cap for quick runs. |
| `--skip-train-plot` | Skip replay figure after training. |

## Outputs and EKF wiring

- **NN:** Point `contact.detector: neural` and `contact.neural.checkpoint` (+ scaler/meta paths) in experiment YAML — see [`leg_odom/run/experiment_config.py`](../run/experiment_config.py) and [`leg_odom/run/contact_factory.py`](../run/contact_factory.py).
- **GMM:** Point `contact.detector: gmm` and pretrained npz / mode (`offline` vs `online`) per project conventions — see ARCHITECTURE / config reference.

## Eval CLIs (post-EKF, optional)

Not part of training, but useful after a full run:

```bash
python -m leg_odom.eval.trajectory_eval --help
python -m leg_odom.eval.analysis_plots --help
```

## Related documentation

- [Features README](../features/README.md) — precompute CLI.
- [Contact README](../contact/README.md) — how detectors consume weights at runtime.
- [Repository README](../../README.md) — main EKF entrypoint.
