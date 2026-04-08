# Features: instant specification and precompute

This package defines the **per-timestep feature vector** used by **GMM+HMM**, **neural contact training**, and the **EKF contact stack** (same field names, compatible layouts). It also hosts the **offline precompute** CLI that writes NumPy bundles for training.

## Modules (no CLI)

| Module | Role |
| ------ | ---- |
| [`instant_spec.py`](instant_spec.py) | Canonical field names, `parse_instant_feature_fields`, `build_timeline_features_for_leg`, helpers shared with `leg_odom.contact.gmm_hmm`. |
| [`__init__.py`](__init__.py) | Re-exports the public names from `instant_spec` for convenience. |

**Consumers:** `leg_odom.contact` (GMM+HMM, neural runtime), `leg_odom.training` (NN/GMM training), `leg_odom.features.precompute_contact_instants`.

## Script: precompute for training

**Entry point:** `python -m leg_odom.features.precompute_contact_instants`

**Purpose:** Walk a tree of raw sequences, load merged timelines per `dataset.kind`, compute full offline instants + foot forces, and write one **`precomputed_instants.npz`** per sequence under `--output-root` (mirrored relative paths). Writes **`precompute_manifest.json`** under the output root.

**Requires:** A conda env with project deps (and training discovery imports). **Does not** import the EKF filter loop.

### Arguments

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `--dataset-root` | Yes | Root containing sequence directories (recursive discovery). |
| `--output-root` | Yes | Root for mirrored `.npz` tree + manifest. |
| `--dataset-kind` | No (default `tartanground`) | `tartanground` or `ocelot` — must match layout under `--dataset-root`. |
| `--robot` | Yes | `anymal` or `go2` (kinematics for feature computation). |
| `--overwrite` | Flag | Replace existing `precomputed_instants.npz`. |
| `--no-validate` | Flag | Skip merged-frame validation (debug only). |
| `--max-sequences` | Optional | Process only first N sequences after discovery order (1–240); prints sample path for smoke tests. |

### Layout discovery (training-side mirror)

- **`tartanground`:** each sequence dir has `imu.csv` and exactly one `*_bag.csv`.
- **`ocelot`:** each sequence dir has `lowstate.csv`.

Same rules as [`leg_odom.training.nn.discovery`](../training/nn/discovery.py).

### Outputs

- **`precomputed_instants.npz`** per sequence: `instants_leg*`, `foot_forces`, `sequence_dir`, `robot_kinematics`, format/spec version fields (see [`leg_odom.training.nn.precomputed_io`](../training/nn/precomputed_io.py)).
- **`precompute_manifest.json`**: roots, counts, per-sequence npz paths and UIDs.

## Downstream dependencies

| Step | Needs precompute? |
| ---- | ----------------- |
| **NN contact training** (`train_contact_nn`) | **Yes** — training reads only npz under `dataset.precomputed_root`. |
| **GMM fit** (`train_gmm`) | **Yes** — same npz discovery. |
| **EKF with `contact.detector: neural` or `gmm` (online)** | Indirect — needs weights produced **after** training/fit (not the npz at EKF runtime unless you use offline GMM inside the detector). |
| **EKF with `grf_threshold`** | **No** — raw logs only. |

## Related documentation

- [Training README](../training/README.md) — how npz is consumed.
- [Repository README](../../README.md) — full pipeline overview.
