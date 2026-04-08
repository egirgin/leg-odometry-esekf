# Architecture snapshot (living document)

**Purpose**: Quick map of layout, classes, and data flow. **Update when structure, APIs, or schemas change**—see [TARTANGROUND_EKF_REFACTOR_PLAN.md](TARTANGROUND_EKF_REFACTOR_PLAN.md) for the full roadmap. Mermaid class/dependency diagrams: [docs/CLASS_DIAGRAM.md](docs/CLASS_DIAGRAM.md).

**Audience**: **Research** codebase—clarity for people who know the methods beats generic “industrial” extensibility.

**Agents**: Prefer **atomic** edits; **wait for user review** before large multi-file refactors unless the user scoped the full task in one request. See the plan’s *Instructions for agentic AI*.

**Last updated**: 2026-04-08 (**Unified dataset / kinematics / training I/O**: `dataset.kind` `tartanground` \| `ocelot`; `TartangroundDataset` / `OcelotDataset`; shared `CachedSingleSequenceDataset`; `io/ocelot_recording`; `precompute_contact_instants`; `build_kinematics_by_name`; factory-only label replay.)

---

## Repository layout (current)

| Area | Role |
| ---- | ---- |
| **`leg_odom/`** | **New** implementation target: ABCs, IO, **`thresholds.py`** (implementation constants + usage docstrings — not YAML experiment params). **Do not import `legacy/`.** Optional PyTorch for `training/nn` — see [requirements-nn.txt](requirements-nn.txt). |
| **`legacy/`** | **Frozen reference**: full prior pipeline—`run_ekf_learning.py`, `run_ekf.py`, loaders, EKF, contact stack, `parameters.py`, `preprocessing/`, `pretrained_models/`, batch shell. For comparison and porting only. |
| **Workspace root** | Docs, **`main.py`**, **`config/`** (experiment YAML templates), `leg_odom/`, `legacy/`, `scripts/`, `.vscode/`. |
| **`scripts/`** | Placeholder for future **new** CLIs wired to `leg_odom` only (see `scripts/README.md`). |

**Import the new package** (from repo root):

```bash
cd /home/girgine/Documents/leg-odometry/iros/async_ekf_workspace
python -c "import leg_odom.contact; import leg_odom.kinematics; import leg_odom.datasets"
```

**New stack entrypoint** (YAML experiment required; repo root = directory containing `main.py`):

```bash
conda activate leg-odometry
cd /home/girgine/Documents/leg-odometry/iros/async_ekf_workspace
python main.py --config config/default_experiment.yaml
# ESEKF runs; outputs `ekf_history_<slug>.csv` + `ekf_process_summary.json` under the run directory.
```

Experiment YAML files must set **`run.name`** and **`dataset.sequence_dir`** explicitly (validated on load). **`dataset.kind`** selects the loader and a path segment under `output_<run>/` (`tartanground` or `ocelot`).

**Run the reference EKF** (must `cd` to legacy for sibling imports, or rely on path hacks inside sub-scripts):

```bash
cd /home/girgine/Documents/leg-odometry/iros/async_ekf_workspace/legacy
python run_ekf_learning.py <sequence_dir> <run_name>
```

---

## Configuration

- **Reference**: [legacy/parameters.py](legacy/parameters.py) holds tunables for the legacy pipeline; `PRETRAINED_MODEL_PATHS` are absolute under `legacy/pretrained_models/`.
- **New experiment YAML**: [config/default_experiment.yaml](config/default_experiment.yaml) — `robot.kinematics`, **`dataset.kind`** (output path segment + factory key: `tartanground` or `ocelot`), **`dataset.sequence_dir`** (required **absolute** path to one trajectory folder; `~` allowed; **`dataset.data_root` removed**), `contact.detector`, **`ekf.noise_config`** (path to [config/ekf_noise/](config/ekf_noise/) YAML for `imu_noise` + `P0_diagonal`; optional inline `ekf` keys still override in code/tests), **`run.debug`** as a **mapping** (`enabled`, **`generate_analysis_plots`** — when `enabled` is false only; if `enabled` is true, analysis figures always run, nested **`live_visualizer`** incl. `buffer_length` / `video_path` / **`hz`** (matplotlib refresh cap vs `median_rate_hz`), `output.*`. No legacy key migration on load. Loaded by [leg_odom/run/experiment_config.py](leg_odom/run/experiment_config.py); validated before [leg_odom/run/output_layout.py](leg_odom/run/output_layout.py). **Not** [leg_odom/thresholds.py](leg_odom/thresholds.py).
- **CLI**: `python main.py --config config/default_experiment.yaml` → writes `experiment_resolved.yaml` under **`output.base_dir / output_{run.name} / {dataset.kind} / {parent(sequence_dir).name} / {traj_slug[_timestamp]}/`** (relative `output.base_dir` is resolved vs workspace root). Orchestration lives in repo-root [main.py](main.py) (`_run_experiment`, `_touch_subpackages`).

---

## Data flow (high level)

1. **Dataset / IO** → merged timeline (**tartanground**: `imu.csv` + `*_bag.csv`; **ocelot**: `lowstate.csv`).
2. **Kinematics model** → foot positions, Jacobians, velocities (world or body per config).
3. **ESEKF** → IMU predict; ZUPT when contact detectors assert stance.
4. **Contact detectors** → `ContactDetectorStepInput` per foot → `(stance, p_stance, zupt_meas_var)`.
5. **Eval / logs** → `ekf_history_<sequence>.csv` (per-step state + contact + ZUPT; wall time as **`sec`** + **`nanosec`**), plots, metrics. Trajectory eval prefers **`t_abs`** on history CSV; fallback **`sec`/`nanosec`** so overlap matches GT **`t_abs`** from merged logs.

**`leg_odom` today**: [leg_odom/run/ekf_process.py](leg_odom/run/ekf_process.py) runs **IMU propagation** via [leg_odom/filters/esekf.py](leg_odom/filters/esekf.py) `ErrorStateEkf` (discrete error-state predict + Joseph-form `update_zupt`). With `contact.detector: grf_threshold`, each step uses per-foot GRF vs threshold ([grf_threshold.py](leg_odom/contact/grf_threshold.py)). With `contact.detector: gmm`, per-foot [contact/gmm_hmm/](leg_odom/contact/gmm_hmm/) consumes `ContactDetectorStepInput`, emission dim `history_length × instant_dim`; `contact.gmm.mode: offline` prefits from the loaded recording with **`history_length` fixed to 1** (instant emissions), `online` may use `N>1` with a pretrained `.npz` plus sliding refit with fallback. With `contact.detector: neural`, [contact/neural.py](leg_odom/contact/neural.py) runs the trained CNN/GRU on sliding windows (same instant fields as training) and maps `p_stance` to ZUPT covariance like GMM. World gravity is fixed **FLU** `g_w = [0,0,9.81]`; kinematic acceleration uses `R @ f - g_w` unless `accel_gravity_compensated` (from [imu_sanitize.py](leg_odom/io/imu_sanitize.py)) is true. IMU load checks include **positive mean accel_z** for specific-force mode. `build_error_state_ekf` loads optional **`ekf.noise_config`** file then optional inline `ekf.imu_noise` / `ekf.P0_diagonal` overrides. [dataset_factory.py](leg_odom/run/dataset_factory.py) builds **`TartangroundDataset`** or **`OcelotDataset`** (shared [datasets/single_sequence.py](leg_odom/datasets/single_sequence.py) base; default `preload=True`). For **tartanground**, `dataset.sequence_dir` must be the **trajectory folder** that contains `imu.csv` (not a parent of multiple runs); **`__len__` is always 1**. [run_ekf_pipeline](leg_odom/run/ekf_process.py) runs **`dataset[0]`** over the **full** merged timeline (no `max_timesteps` cap).

---

## `leg_odom/` map _(placeholders vs live)_

| Subpackage | Status | Notes |
| ---------- | ------ | ----- |
| `contact/base.py` | **ABCs** | `ContactDetectorStepInput` (per-foot GRF + `p_foot_body` + body kin + gyro), `BaseContactDetector.update(step)`, `ContactEstimate` (`stance`, `p_stance`, `zupt_meas_var`; NaN only when undefined, e.g. empty buffer). |
| `contact/grf_threshold.py` | **Live** | `GrfThresholdContactDetector` (per-foot GRF ≥ threshold; constant isotropic ZUPT variance); `build_grf_threshold_detectors_from_cfg`; `python -m leg_odom.contact.grf_threshold` for GRF vs stance plots. |
| `contact/grf_stance_plot.py`, `contact/replay_timeline.py` | **Live** | Shared matplotlib GRF overview + bag replay for any `BaseContactDetector` (used by GMM visualize, GRF CLI, `train_gmm`). |
| `features/` | **Live** | `instant_spec`: scalar instant vector for `ContactDetectorStepInput` (GMM+HMM, NN training, precompute). NN precompute CLI: `--dataset-kind tartanground|ocelot` → `python -m leg_odom.features.precompute_contact_instants`. |
| `contact/gmm_hmm/` | **Live** | `detector`, `fitting`, `hmm_gaussian`, `visualize` (`python -m leg_odom.contact.gmm_hmm.visualize`); instant layout from `leg_odom.features` (`INSTANT_FEATURE_SPEC_VERSION`, re-exported names). |
| `contact/neural.py` | **Live** | `NeuralContactDetector` + `NeuralSharedRuntime`: loads `train_contact_nn` `.pt` / `_meta.json` / `_scaler.npz`, `instant_vector_from_step` + training-style window padding, ZUPT `R` from `p_stance` like GMM. `build_neural_detectors_from_cfg`; YAML `contact.detector: neural`. |
| `contact/dual_hmm_fusion.py`, `contact/ocelot.py` | Stub | Port from `legacy/` (Phases 5+). |
| `kinematics/base.py` | **ABC** | `fk`, `J_analytical`, `jacobian_numerical` (shared finite differences). |
| `kinematics/anymal.py` | **Live** | `AnymalKinematics`: URDF chain, analytic J, `leg_chain_points` (legacy port). |
| `kinematics/go2.py` | **Live** | `Go2Kinematics`: serial abad–hip–knee FK; J via numerical (legacy `J_num`). |
| `datasets/base.py` | **ABC** | `__len__` / `__getitem__` → `LegOdometrySequence`. |
| `datasets/single_sequence.py` | **Live** | `CachedSingleSequenceDataset`: shared cache/validation/`LegOdometrySequence` build for one directory per trajectory. |
| `datasets/tartanground.py` | **Live** | `TartangroundDataset`: imu+bag layout; **`__len__==1`**; **`preload=True` by default**. |
| `datasets/types.py` | **Live** | `LegOdometrySequence` dataclass returned by `__getitem__`. |
| `datasets/anymal.py`, `go2.py` | Stub | Doc-only placeholders; **robot** models are under `kinematics/`, not here. |
| `datasets/grandtour.py` | Stub | Future data product layout placeholder. |
| `datasets/ocelot.py` | **Live** | `OcelotDataset` (`dataset.kind: ocelot`): requires `lowstate.csv`; optional `groundtruth.csv` and `frames/` are detected as metadata in V1. |
| `filters/esekf.py` | **Live** | `ErrorStateEkf`: `predict` / `imu_predict`, `foot_velocity_world`, `update_zupt`; `build_error_state_ekf`. |
| `run/ekf_process.py` | **Live** | `run_ekf_pipeline` / `run_ekf_on_recording`: **`ContactStack`** carries `detector_id` + optional per-foot detectors; timestep loop; optional **`ekf.initialize_nominal_from_data`** seeds nominal state via `run/ekf_nominal_init.py`; streaming `ekf_history_*.csv` when `run_dir` set; **`EkfProcessSummary`** + flat `ekf_process_summary.json`; **live matplotlib** when `live_visualizer=True`. |
| `run/dataset_factory.py`, `run/kinematics_factory.py`, `run/contact_factory.py` | **Live** | `build_leg_odometry_dataset` (`dataset.kind`: `tartanground` \| `ocelot`; optional ctor kwargs `validate`, …), `build_kinematics_backend` / `build_kinematics_by_name`, `build_contact_stack` (`grf_threshold` \| `gmm` \| `neural`; `workspace_root` for relative neural checkpoint). |
| `thresholds.py` (package root) | **Live** | IMU / timebase / **kinematics numerical Jacobian step** — each constant documents **where used**; not experiment YAML. |
| `io/split_imu_bag.py`, `io/ocelot_recording.py`, `io/timebase.py`, `io/imu_sanitize.py`, `io/ground_truth.py`, `io/validation.py` | **Live** | Tartanground split merge + Ocelot recording load (`load_prepared_ocelot`); split requires `imu.csv` + `*_bag.csv`, Ocelot requires `lowstate.csv`; both use `sec`/`nanosec`, FLU IMU checks, and shared timebase/validation. |
| `io/anymal_split.py` | Thin re-export | Documents ANYmal/Tartanground split layout; same API as `split_imu_bag`. |
| `eval/ekf_step_log.py` | **Live** | Per-step CSV: time, `p,v`, euler (deg), biases, `P` diagonal, contact + `zupt_meas_var`, batch NIS, per-foot ZUPT diagnostics + `leg{i}_v_w{axis}` world foot velocity. |
| `eval/metrics.py` | **Live** | :class:`~leg_odom.eval.metrics.TrajectoryEvaluator` + `evaluate_trajectory`; `time_alignment_report` (overlap diagnostic); **`evaluation_metrics.csv`**: **`ate_m`** = standard Euclidean RMSE over time (2D or 3D); **`ate_x/y/z_m`** per-axis RMSE when available. |
| `eval/analysis_plots.py` | **Live** | :class:`~leg_odom.eval.analysis_plots.EkfRunAnalysis` (legacy-style OOP); **`evaluation_metrics.png`** (scalar metric subplots); CLI `python -m leg_odom.eval.analysis_plots`. |
| `eval/trajectory_eval.py` | **Live** | CLI → `evaluation_metrics.csv`; **`--check-only`** prints time overlap JSON (no full metric run). |
| `eval/live_visualizer.py` | **Live** | Debug 2×2: trajectory, camera N/A, **p_z / v vs time** with sliding x-window, dashed **GT z + GT v**, velocity legend **vx/vy/vz** (est then GT per axis), GT heading from **quats** when merged into viz DF else velocity atan2, **`np.unwrap`** before `interp`, optional **`hz`**, progress bar. |
| `run/post_ekf.py` | **Live** | After EKF: debug → `plots/*.png`; analysis enabled (YAML or debug) → `analysis/*.png` + `evaluation_metrics.png` (no nested sequence subfolder); each tree has `evaluation_metrics.csv` at its root. |
| `training/nn/` | **Live** | Dataset routing `dataset.kind: tartanground|ocelot` ([discovery.py](leg_odom/training/nn/discovery.py), [sequence_frames.py](leg_odom/training/nn/sequence_frames.py), [io_labels.py](leg_odom/training/nn/io_labels.py)). Label replay uses `build_leg_odometry_dataset` only. `dual_hmm`/`ocelot` label methods remain `NotImplementedError`. default `output.dir` = `pretrained_{cnn,gru}`. Artifacts: `contact_{cnn,gru}.pt`, `_*_meta.json`, `_*_scaler.npz`. |
| `training/gmm/` | **Live** | `train_gmm.py`: fit from `precomputed_instants.npz` under `--precomputed-root` (subset columns + `sliding_windows_flat`); optional `--max-sequences`; `save_pretrained_gmm_npz`; post-train replay plot infers dataset kind from stored `sequence_dir` (`lowstate.csv` => `ocelot`, else => `tartanground`). |
| `run/experiment_config.py`, `run/output_layout.py` | **Live** | YAML deep-merge + validate; `resolve_ekf_noise_config_path`; `resolve_contact_neural_paths`; `generate_analysis_plots_enabled(cfg, cli_debug=…)` (true if effective debug **or** YAML flag); `prepare_run_output_dir` uses **`dataset.kind`** in the path; repo [main.py](main.py) wires `--config` → EKF + post-EKF |
| `run/__init__.py` | **Live** | Re-exports `experiment_config` helpers; **lazy** `run_ekf_pipeline` / `prepare_run_output_dir` so importing `leg_odom.run.contact_factory` does not load matplotlib. |

---

## GMM + HMM modes _(summary)_

| Mode | Meaning |
| ---- | ------- |
| `offline` (`contact.gmm`) | Whole-sequence 2-GMM per leg before EKF; no `.npz` required at runtime. |
| `online` | Initial + fallback emissions from pretrained `.npz` under `leg_odom/training/gmm/` (or absolute path); periodic sliding refit. |
| Stationary guard | Degenerate window refit → revert to last good or pretrained means (**plan §3a**). |
| Single vs dual HMM | Kin-only, GRF-only, or fused ([legacy/dual_hmm.py](legacy/dual_hmm.py)). |
| `USE_ENERGY` | Optional transition scaling from foot “energy”; `gamma` update subject to revision—keep toggle. |

---

## Detectors _(names)_

| Name | Note |
| ---- | ---- |
| GMM+HMM (kin / GRF / dual) | Primary statistical track; configurable modalities. |
| CNN / GRU | Neural; same `(N, D)` feature contract, no sliding GMM. |
| **Ocelot** | FSM + GMM thresholds + GLRT ([legacy/contact_detector.py](legacy/contact_detector.py)); plug under `BaseContactDetector` in new code. |

---

## Key abstractions _(target in `leg_odom`)_

| Concept | Location | Intent |
| ------- | -------- | ------ |
| `BaseContactDetector` | `leg_odom/contact/base.py` | `update` / `reset`; fixed `N`, `D`; standardized output. |
| `BaseKinematics` | `leg_odom/kinematics/base.py` | `fk`, `J_analytical`, optional `jacobian_numerical` (finite differences). |
| `BaseLegOdometryDataset` | `leg_odom/datasets/base.py` | PyTorch-style sequences; implementations keyed by **data product** (`dataset.kind`), not robot name. |

---

## Legacy module index _(reference only)_

| File / folder | Role |
| ------------- | ---- |
| [legacy/run_ekf_learning.py](legacy/run_ekf_learning.py) | Main ANYmal EKF + learning entry. |
| [legacy/data_loader.py](legacy/data_loader.py), [legacy/helpers.py](legacy/helpers.py) | Load, timebase, alignment. |
| [legacy/estimator.py](legacy/estimator.py) | ESEKF + ZUPT. |
| [legacy/kinematics.py](legacy/kinematics.py) | Go2 + ANYmal kinematics. |
| [legacy/contact_detector_learning.py](legacy/contact_detector_learning.py), [legacy/dual_hmm.py](legacy/dual_hmm.py), [legacy/contact_detector.py](legacy/contact_detector.py) | Contact stack. |
| [legacy/parameters.py](legacy/parameters.py) | Config. |
| [legacy/preprocessing/](legacy/preprocessing/) | Reference preprocessing. |
| [legacy/pretrained_models/](legacy/pretrained_models/) | Training scripts + weights. |

---

## Changelog _(brief)_

| Date | Change |
| ---- | ------ |
| 2026-04-08 | **Unified dataset / kinematics / training I/O**: `dataset.kind` → `tartanground` \| `ocelot` (removed `tartanground_split`). Classes `TartangroundDataset`, `OcelotDataset` share `CachedSingleSequenceDataset`. IO: `io/ocelot_recording.py` (`load_prepared_ocelot`). Precompute CLI: `python -m leg_odom.features.precompute_contact_instants`. `build_kinematics_by_name`; NN discovery `discover_tartanground_sequence_dirs`; `sequence_frames.py`; label replay via `build_leg_odometry_dataset` only. |
| 2026-04-08 | **Training Ocelot routing**: preprocessing now accepts `--dataset-kind ocelot` with recursive `lowstate.csv` discovery; NN training routing/label replay supports Ocelot+Go2 for `grf_threshold` and `gmm_hmm`; GMM post-train replay infers dataset kind from `sequence_dir_stored` instead of hard-coded Tartanground. |
| 2026-04-08 | **Ocelot lowstate dataset V1**: added `dataset.kind: ocelot`, `io/ocelot_lowstate.py`, and `OcelotLowstateDataset`; strict path validation now checks `lowstate.csv` for Ocelot while keeping split checks for Tartanground. Optional `groundtruth.csv` and `frames/` are presence-checked only (not wired to eval/viz yet). |
| 2026-04-08 | **CSV time**: `imu.csv`, `*_bag.csv`, and `ekf_history_*.csv` require **`sec` + `nanosec`** (no `time` / `ros_*` / `timestamp_*` / single-column aliases); `build_timebase` + eval/GT fallbacks match. |
| 2026-04-08 | **Eval + outputs**: `t_abs` eval alignment; per-axis ATE + standard **`ate_m`** only; `evaluation_metrics.png`; run dir `…/output_<run>/<dataset.kind>/…`; flat `analysis/`; debug forces analysis. |
| 2026-04-03 | **NN labels.method `grf_threshold`**: matches `contact.detector` id; `labels.grf_threshold` mirrors `contact.grf_threshold`. |
| 2026-04-03 | **NN training labels**: `gmm_hmm` (offline per-sequence, `history_length: 1`); `dual_hmm`/`ocelot` not implemented; `label_timelines.py`, `stance_by_seq_leg`. |
| 2026-04-03 | **Live visualizer GT heading**: EKF loop copies `ori_q*` onto viz GT frame (same length as merged timeline); `zyx` yaw + `np.unwrap` before `np.interp` (fixes wrap artifacts); still falls back to velocity atan2 if no quats. |
| 2026-04-03 | **Live visualizer**: `run.debug.live_visualizer.hz` throttles matplotlib redraws (effective ``min(hz, recording.median_rate_hz)``); base velocity legend order vx est/GT, vy est/GT, vz est/GT. |
| 2026-04-03 | **GMM offline `N=1`**: `mode: offline` requires `history_length: 1` (instant emissions only); no `_flat_window` writes offline; removed offline left-padding; factory + `GmmHmmContactDetector` raise on mismatch. |
| 2026-04-03 | **GMM features v3** (`p_foot_body_*`, `est_tau_*`, 12 joint indices); **`ContactDetectorStepInput.p_foot_body`**; ZUPT variance from `p_stance` for swing too; online warmup; **`contact/replay_timeline.py`**, **`contact/grf_stance_plot.py`**; **`train_gmm`** `tqdm` + `plots/`; **`python -m leg_odom.contact.grf_threshold`**. |
| 2026-04-02 | **GMM+HMM**: package `contact/gmm_hmm/`; modes `offline` / `online`; ZUPT `R=(1/p_stance)I` (floored); `save_pretrained_gmm_npz` in `training/gmm/train_gmm.py`; `python -m leg_odom.contact.gmm_hmm.visualize`; expanded feature fields vs `ContactDetectorStepInput`. |
| 2026-04-02 | **Contact + EKF init**: `ContactDetectorStepInput` + `BaseContactDetector.update(step)`; `ContactEstimate.zupt_meas_var` always `float` (NaN when not applicable); `ContactStack.detector_id`; `run_ekf_on_recording(..., contact_stack=)` only; **`ekf.initialize_nominal_from_data`** + `ErrorStateEkf.seed_nominal_state` / `run/ekf_nominal_init.py` (column contract in `config/experiment_parameters_reference.yaml`). |
| 2026-04-02 | **`sequence_name` rename** (was `sequence_id`): **`LegOdometrySequence`**, **`EkfProcessSummary`** / JSON key, **`evaluation_metrics.csv`** first column, **`TrajectoryEvaluator.evaluate`**, **`python -m leg_odom.eval.trajectory_eval --sequence-name`**. |
| 2026-04-02 | **Flat EKF process summary**: removed **`EkfSequenceRunStats`** and **`sequence_stats`**; **`EkfProcessSummary`** includes sequence label, **`median_rate_hz`**, **`ekf_history_csv`**; JSON drops the **`sequences`** wrapper (breaking change for parsers of `ekf_process_summary.json`). **`run_ekf_on_recording`** returns a **`(sequence_name, hz, history_path)`** tuple. |
| 2026-04-02 | **Strict dataset paths + full EKF**: removed **`dataset.data_root`**; **`dataset.sequence_dir`** must be **absolute** (after `~` expansion); reject legacy `data_root` key; `resolve_dataset_paths` / `experiment_resolved.yaml` drop `data_root`; removed **`max_timesteps`** from `run_ekf_on_recording` / `run_ekf_pipeline` / `main.py`. |
| 2026-04-02 | **Single-sequence Tartanground**: `TartangroundSplitDataset` requires `imu.csv` on `sequence_dir` (no batch parent); `__len__==1`; removed `discover_split_sequence_directories` export; `run_ekf_pipeline` uses `dataset[0]`; `post_ekf` uses `dataset[0]` for merged/GT; `analysis_plots.plot_states` uses `.to_numpy()` for matplotlib/pandas compatibility. |
| 2026-04-01 | **`ekf.noise_config`** sidecar YAML; **`run.debug.generate_analysis_plots`** (replaces `output.run_analysis_after_ekf`); live viz **`buffer_length` / `video_path`**; remove **`ekf.enabled`** (strip on load). |
| 2026-04-01 | **`main.py`** CLI reduced to **`--config`** only; removed `--run-name`, `--sequence-dir`, `--skip-output`, `--max-ekf-steps`, `--debug`, `--no-ekf`, `--skip-import-check`; dropped **`apply_cli_overrides`**. |
| 2026-04-01 | Removed **`leg_odom/run/cli_main.py`**; CLI orchestration lives in repo-root **`main.py`** (`_run_experiment`, `_touch_subpackages`). |
| 2026-04-01 | Single run config snapshot: **`experiment_resolved.yaml`** with comment header (path rewrites + optional `input_yaml`); removed **`experiment_input.yaml`** copy. |
| 2026-04-01 | **`experiment_config`**: removed legacy merge helpers (`_normalize_run_debug_section`, `_migrate_legacy_output_analysis_flag`, `_strip_deprecated_ekf_keys`); `run.debug` must be a mapping in merged config. |
| 2026-04-01 | **`run.debug` nested**: `enabled` + `live_visualizer.enabled` / `sliding_window_s`; legacy bool + top-level `run.live_visualizer` normalized on merge; `run_ekf_*(..., live_visualizer=)`. |
| 2026-04-01 | **`run.live_visualizer.sliding_window_s`** in YAML; `live_visualizer_sliding_window_s(cfg)`. |
| 2026-04-01 | **Live visualizer**: 60 s sliding window on z/v panels; dashed GT z + GT v (`np.gradient` on position). |
| 2026-04-01 | **`debug` and `run_analysis_after_ekf` independent** (each triggers post-EKF to its folder; both allowed). |
| 2026-04-01 | **`EkfRunAnalysis` / `TrajectoryEvaluator`**, `post_ekf.py`, `evaluation_metrics.csv`, `output.run_analysis_after_ekf`, **`run.debug`** + live viz in EKF loop. |
| 2026-04-01 | **`eval`**: `metrics.py`, `analysis_plots.py`, `trajectory_eval.py`, `live_visualizer.py` (camera panel N/A for Tartanground); `tests/test_eval_metrics.py`. |
| 2026-04-01 | **EKF summary**: dropped `n_timesteps` / `n_imu_predictions` / `n_timesteps_total` from stats + JSON (infer from history CSV or `len(frames)`). |
| 2026-04-01 | **`eval/ekf_step_log`**: streaming `ekf_history_*.csv` from `run_ekf_pipeline`; `ErrorStateEkf.foot_velocity_world`; removed terminal velocity plot / `print_velocity`. |
| 2026-04-01 | **GRF threshold contact**: `leg_odom/contact/grf_threshold.py`, `ContactEstimate.zupt_meas_var`, `run_ekf_on_recording` + `contact.detector: grf_threshold`, optional YAML `contact.grf_threshold`, `tests/test_grf_threshold_contact.py`. |
| 2026-04-01 | **ESEKF / IMU defaults**: removed `gravity_sign` / `static_velocity_std`; fixed FLU gravity `R@f - [0,0,g]`; `predict(..., accel_gravity_compensated=...)`; IMU sanitize **mean accel_z > 0** check; `ekf.enabled` default **true**; dataset **`preload` default true**; `n_imu_predictions` stat from loop count only. |
| 2026-04-01 | **ESEKF**: `leg_odom/filters/esekf.py` full port from `legacy/estimator.py` (`ErrorStateEkf`, `build_error_state_ekf`); process loop uses `accel_gravity_compensated` meta; `tests/test_esekf.py`. |
| 2026-04-01 | **CLI + dataset memo**: `main.py` requires `--config` only; `TartangroundSplitDataset` memoizes `__getitem__`; `prepare_run_output_dir` returns resolved dict; `load_experiment_yaml` requires explicit `run.name` / `dataset.sequence_dir`. |
| 2026-04-01 | **EKF process loop**: `run/ekf_process.py`, dataset + kinematics factories, `ErrorStateEkf`, `main.py` + `ekf.enabled`, `tests/test_ekf_process_loop.py`. |
| 2026-04-01 | **Experiment YAML** + `leg_odom/run/` + `main.py --config`; `config/default_experiment.yaml`; output dirs with `experiment_resolved.yaml`. |
| 2026-04-01 | **Kinematics**: `BaseKinematics`, `AnymalKinematics`, `Go2Kinematics` (ports from `legacy/kinematics.py`); `tests/test_kinematics.py`. |
| 2026-04-01 | **`leg_odom/thresholds.py`**: documented implementation constants; foot forces **0-indexed only** (removed legacy 1..4 copies). |
| 2026-04-01 | **IMU**: FLU-only validation in `imu_sanitize.py`; flag `accel_gravity_compensated`; no FRD axis remap. |
| 2026-04-01 | **`leg_odom.io`**: split IMU+bag load/merge, timebase, GT extract, validation; **`TartangroundSplitDataset`**; `tests/` + `scripts/plot_tartanground_checks.py` → `plots/`. |
| 2026-04-01 | Added root **`main.py`**: CLI for Tartanground/ANYmal sequence validation + `run_name`; no `legacy` imports. |
| 2026-04-01 | Consolidated full reference tree under `legacy/`; `leg_odom` documented as independent of `legacy`. |
| 2026-04-01 | Added `leg_odom/` package with ABCs and placeholder modules; initial ARCHITECTURE table. |
| _(prior)_ | Initial stub. |
