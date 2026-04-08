# Leg odometry (async EKF workspace)

Research codebase for **quadruped leg odometry** using an **error-state extended Kalman filter (ESEKF)** with IMU propagation and **zero-velocity updates (ZUPT)** during stance, driven by **contact estimates** (GRF threshold, GMM+HMM, or neural networks).

The **new** implementation lives in [`leg_odom/`](leg_odom/). A frozen reference pipeline is under [`legacy/`](legacy/) (do not import `legacy` from `leg_odom`).

## Environment

```bash
conda activate leg-odometry
cd /path/to/async_ekf_workspace
```

Python import check (from repo root):

```bash
python -c "import leg_odom.contact; import leg_odom.kinematics; import leg_odom.datasets"
```

Optional NN stack: see [`requirements-nn.txt`](requirements-nn.txt).

## Running the main EKF experiment

Experiments are defined by **YAML** under [`config/`](config/). Required keys include `run.name` and `dataset.sequence_dir` (absolute path to one trajectory folder).

```bash
python main.py --config config/default_experiment.yaml
```

[`main.py`](main.py) will:

1. Validate paths and merge defaults ([`leg_odom/run/experiment_config.py`](leg_odom/run/experiment_config.py)).
2. Create a timestamped **run directory** ([`leg_odom/run/output_layout.py`](leg_odom/run/output_layout.py)):
   - `output.base_dir` / `output_{run.name}` / `{dataset.kind}` / `{parent(sequence_dir).name}` / `{trajectory_folder}[_timestamp]/`
3. Write **`experiment_resolved.yaml`** (canonical snapshot with absolute paths).
4. Run the EKF pipeline on **`dataset[0]`** (one full merged recording).
5. If debug or analysis flags apply, run post-EKF evaluation/plots ([`leg_odom/run/post_ekf.py`](leg_odom/run/post_ekf.py)).

`dataset.kind` is **`tartanground`** (imu + one bag CSV) or **`ocelot`** (`lowstate.csv`). Robot model is `robot.kinematics`: **`anymal`** or **`go2`**.

## Run outputs (what to expect)

Inside the run directory you typically get:

| Artifact | Meaning |
| -------- | ------- |
| `experiment_resolved.yaml` | Validated experiment config used for this run |
| `ekf_process_summary.json` | Summary: sequence name, rate, path to history CSV if written |
| `ekf_history_<sequence>.csv` | Per-step state, contact, ZUPT diagnostics (`sec` + `nanosec` timebase) |
| `plots/` | When **debug** is on: quick figures + `evaluation_metrics.csv` |
| `analysis/` | When **run.debug.generate_analysis_plots** (or equivalent) is on: analysis figures + metrics |

Trajectory metrics and figures can also be produced via eval CLIs (see below).

## Pipelines that do **not** require the EKF loop

These can be run as standalone CLIs; they share **datasets**, **kinematics**, and **instant features** with the EKF but do not import the timestep loop unless you run `main.py`:

| Area | README |
| ---- | ------ |
| Precompute (features for training) | [`leg_odom/features/README.md`](leg_odom/features/README.md) |
| Training (NN + GMM) | [`leg_odom/training/README.md`](leg_odom/training/README.md) |
| Contact inference & visualization | [`leg_odom/contact/README.md`](leg_odom/contact/README.md) |

**Dependency chain (high level):** precompute produces `precomputed_instants.npz` → NN and/or GMM training → checkpoints in experiment YAML → EKF uses `contact.detector`. **GRF-threshold** contact needs **no** precompute. Details are in the submodule READMEs.

## Eval (after a run)

```bash
python -m leg_odom.eval.trajectory_eval --help
python -m leg_odom.eval.analysis_plots --help
```

These consume run outputs (e.g. `ekf_history_*.csv`, GT when available) and write metrics/plots.

## Example shell workflows

See [`examples/README.md`](examples/README.md) for copy-paste commands and placeholder paths.

## Deeper reference

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Layout, factories, data flow, changelog table.
- **[docs/CLASS_DIAGRAM.md](docs/CLASS_DIAGRAM.md)** — Mermaid UML-style class and dependency diagrams.
