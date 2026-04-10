"""
Train CNN or GRU contact classifiers. Hyperparameters come from YAML (see ``default_train_config.yaml``).

Requires precomputed ``precomputed_instants.npz`` under ``dataset.precomputed_root`` (stance timelines
are stored in the bundle; run ``python -m leg_odom.features.precompute_contact_instants --config <yaml>``).

Example::

    python -m leg_odom.training.nn.train_contact_nn --config leg_odom/training/nn/default_train_config.yaml

Sequences are discovered as ``precomputed_instants.npz`` files under ``dataset.precomputed_root`` only.

When ``output.dir`` is null or omitted in YAML, weights are written under
``leg_odom/training/nn/pretrained_{cnn,gru}/`` (see :func:`~leg_odom.training.nn.config.load_nn_train_config`).

If ``visualization.enabled`` and a test split exists, each time the best checkpoint improves the run
writes ``output.dir/plots/samples/<stem>_epoch_<k>.png`` (random ``[train]`` and ``[test]`` windows).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from leg_odom.features.instant_spec import (
    FULL_OFFLINE_INSTANT_FIELDS,
    INSTANT_FEATURE_SPEC_VERSION,
    parse_instant_feature_fields,
)
from leg_odom.run.kinematics_factory import build_kinematics_by_name
from leg_odom.training.nn.config import default_train_config_path, load_nn_train_config
from leg_odom.training.nn.data import (
    build_sliding_window_datasets,
    collect_train_instant_matrix,
    load_precomputed_subset_by_npz_paths,
)
from leg_odom.training.nn.models import ContactCNN, ContactGRU
from leg_odom.training.nn.precomputed_io import discover_precomputed_instants_npz, load_precomputed_sequence_npz
from leg_odom.training.nn.visualize_sections import plot_random_train_test_sections


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split_sequence_paths(
    paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1, got {s}")
    rng = np.random.default_rng(seed)
    p = np.array(paths, dtype=object)
    rng.shuffle(p)
    n = len(p)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(
            "Train split is empty: use more sequences or lower val/test ratios "
            f"(n={n}, n_train={n_train}, n_val={n_val}, n_test={n_test})."
        )
    train = p[:n_train].tolist()
    val = p[n_train : n_train + n_val].tolist()
    test = p[n_train + n_val :].tolist()
    return train, val, test


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CNN/GRU contact models (YAML config; dataset.kind + precomputed npz bundles)"
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Training YAML (default: {default_train_config_path()})",
    )
    return p.parse_args()


def _eval_loader_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    desc: str,
) -> tuple[float, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Average BCE loss and boolean labels (swing=False, stance=True) vs predictions at 0.5 threshold.

    Iteration uses ``tqdm`` (``leave=False``) for progress during eval.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    all_y: list[float] = []
    all_p: list[float] = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc=desc, leave=False, unit="batch"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            n_batches += 1
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            yy = batch_y.cpu().numpy().flatten()
            all_p.extend(prob.tolist())
            all_y.extend(yy.tolist())
    avg_loss = total_loss / max(n_batches, 1)
    y_true = np.array(all_y) > 0.5
    y_pred = np.array(all_p) > 0.5
    return avg_loss, y_true, y_pred


def _eval_loader_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device, *, desc: str
) -> tuple[float, dict[str, float]]:
    avg_loss, y_true, y_pred = _eval_loader_predictions(model, loader, device, desc=desc)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return avg_loss, metrics


def _binary_eval_details(
    y_true: npt.NDArray[np.bool_], y_pred: npt.NDArray[np.bool_]
) -> dict[str, Any]:
    """
    Confusion matrix and per-class metrics with labels swing=0, stance=1.

    Sklearn ``confusion_matrix(..., labels=[0,1])`` rows/cols are true/pred class order.
    """
    yt = y_true.astype(np.int32)
    yp = y_pred.astype(np.int32)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    prec, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    return {
        "confusion_true_swing_pred_swing": int(cm[0, 0]),
        "confusion_true_swing_pred_stance": int(cm[0, 1]),
        "confusion_true_stance_pred_swing": int(cm[1, 0]),
        "confusion_true_stance_pred_stance": int(cm[1, 1]),
        "overall_accuracy": float(accuracy_score(yt, yp)),
        "precision_swing": float(prec[0]),
        "precision_stance": float(prec[1]),
        "recall_swing": float(rec[0]),
        "recall_stance": float(rec[1]),
        "f1_swing": float(f1[0]),
        "f1_stance": float(f1[1]),
        "support_swing": int(sup[0]),
        "support_stance": int(sup[1]),
    }


def _save_eval_metrics_csv(path: Path, eval_split: str, details: dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"eval_split": eval_split, **details}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def _save_training_curves(
    path: Path,
    train_losses: list[float],
    val_losses: list[float] | None,
    train_accs: list[float],
    val_accs: list[float] | None,
) -> None:
    """Write train/val BCE loss and accuracy vs epoch (losses comparable: eval mode, no dropout)."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(train_losses)
    if n == 0:
        return
    x = np.arange(1, n + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_loss.plot(x, train_losses, label="train", color="C0")
    if val_losses is not None and len(val_losses) == n:
        ax_loss.plot(x, val_losses, label="val", color="C1")
    ax_loss.set_ylabel("BCE loss")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_title("Training curves")

    ax_acc.plot(x, train_accs, label="train", color="C0")
    if val_accs is not None and len(val_accs) == n:
        ax_acc.plot(x, val_accs, label="val", color="C1")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve() if args.config else default_train_config_path()
    print(f"[train_contact_nn] loading YAML: {cfg_path}")
    cfg: dict[str, Any] = load_nn_train_config(cfg_path)

    device = _pick_device()
    print(f"[train_contact_nn] device={device}")
    print(f"[train_contact_nn] config={cfg_path}")

    dataset_kind = str(cfg["dataset"]["kind"]).strip()
    precomputed_root = Path(cfg["dataset"]["precomputed_root"]).expanduser().resolve()
    dl = cfg["data_loading"]
    prep_verbose = bool(dl["verbose"])
    arch = str(cfg["architecture"])

    all_npz = discover_precomputed_instants_npz(precomputed_root, verbose=prep_verbose)
    tr = cfg["training"]
    train_paths, val_paths, test_paths = _split_sequence_paths(
        all_npz,
        float(tr["train_ratio"]),
        float(tr["val_ratio"]),
        float(tr["test_ratio"]),
        int(tr["seed"]),
    )
    print(
        f"[train_contact_nn] dataset.kind={dataset_kind!r} architecture: {arch} "
        f"precomputed_instants.npz: total={len(all_npz)} train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}"
    )

    print("[train_contact_nn] loading precomputed npz and building DataLoaders…")

    fields = tuple(str(x).strip() for x in cfg["features"]["fields"])
    for f in fields:
        if f not in FULL_OFFLINE_INSTANT_FIELDS:
            raise ValueError(
                f"features.fields contains {f!r} which is not in FULL_OFFLINE_INSTANT_FIELDS "
                f"(re-run preprocess if you need a new column in the npz)."
            )
    spec = parse_instant_feature_fields(fields)
    kin = build_kinematics_by_name(str(cfg["robot"]["kinematics"]))
    n_legs = int(kin.n_legs)
    window = int(cfg["model"]["window_size"])
    robot_kin_name = str(cfg["robot"]["kinematics"])

    instants_by_seq_leg, foot_forces_by_seq, sequence_uid_by_seq, stance_by_seq_leg = (
        load_precomputed_subset_by_npz_paths(
            (train_paths, val_paths, test_paths),
            robot_kin_name,
            n_legs,
            fields,
            show_progress=prep_verbose,
        )
    )

    ref_bundle = load_precomputed_sequence_npz(
        train_paths[0],
        expected_robot_kinematics=robot_kin_name,
        n_legs=n_legs,
    )

    scaler = StandardScaler()
    train_mat = collect_train_instant_matrix(train_paths, n_legs, instants_by_seq_leg)
    scaler.fit(train_mat)

    for_cnn = arch == "cnn"
    train_ds = build_sliding_window_datasets(
        train_paths,
        n_legs,
        scaler,
        window,
        for_cnn=for_cnn,
        foot_forces_by_seq=foot_forces_by_seq,
        instants_by_seq_leg=instants_by_seq_leg,
        sequence_uid_by_seq=sequence_uid_by_seq,
        stance_by_seq_leg=stance_by_seq_leg,
    )
    val_ds = None
    if val_paths:
        val_ds = build_sliding_window_datasets(
            val_paths,
            n_legs,
            scaler,
            window,
            for_cnn=for_cnn,
            foot_forces_by_seq=foot_forces_by_seq,
            instants_by_seq_leg=instants_by_seq_leg,
            sequence_uid_by_seq=sequence_uid_by_seq,
            stance_by_seq_leg=stance_by_seq_leg,
        )

    bs = int(tr["batch_size"])
    nw = int(tr["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = (
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw) if val_ds is not None else None
    )

    n_train_samples = len(train_ds)
    n_val_samples = len(val_ds) if val_ds is not None else 0
    print(
        f"[train_contact_nn] tensors ready: train_samples={n_train_samples} "
        f"val_samples={n_val_samples} batch_size={bs} window={window} n_legs={n_legs}"
    )

    d_in = spec.instant_dim
    if for_cnn:
        model = ContactCNN(d_in, window_size=window).to(device)
    else:
        model = ContactGRU(d_in).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train_contact_nn] model: {arch}  d_in={d_in}  parameters={n_params:,}  "
        f"optimizer=Adam lr={float(tr['learning_rate']):.2e}"
    )

    out_dir = Path(str(cfg["output"]["dir"])).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"contact_{arch}"
    pt_path = out_dir / f"{stem}.pt"
    meta_path = out_dir / f"{stem}_meta.json"
    scaler_path = out_dir / f"{stem}_scaler.npz"
    plots_dir = out_dir / "plots"
    samples_dir = plots_dir / "samples"
    curves_path = plots_dir / f"{stem}_training_curves.png"
    eval_csv_path = out_dir / f"{stem}_eval_metrics.csv"

    meta: dict[str, Any] = {
        "config_path": str(cfg_path),
        "dataset_kind": dataset_kind,
        "contact_label_method": ref_bundle.contact_label_method,
        "contact_labels_config": dict(ref_bundle.contact_labels_config),
        "reference_train_npz": str(Path(train_paths[0]).resolve()),
        "architecture": arch,
        "instant_feature_spec_version": int(INSTANT_FEATURE_SPEC_VERSION),
        "feature_fields": list(spec.fields),
        "history_length": window,
        "instant_dim": int(spec.instant_dim),
        "robot_kinematics": str(cfg["robot"]["kinematics"]),
        "train_ratio": float(tr["train_ratio"]),
        "val_ratio": float(tr["val_ratio"]),
        "test_ratio": float(tr["test_ratio"]),
        "seed": int(tr["seed"]),
        "train_precomputed_instants_npz": [str(p.resolve()) for p in train_paths],
        "val_precomputed_instants_npz": [str(p.resolve()) for p in val_paths],
        "test_precomputed_instants_npz": [str(p.resolve()) for p in test_paths],
        "precomputed_root": str(precomputed_root),
    }

    epochs = int(tr["epochs"])
    best_metric = float("inf")
    best_epoch = 0
    use_val = val_loader is not None and len(val_paths) > 0

    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    train_acc_hist: list[float] = []
    val_acc_hist: list[float] = []

    selection = "lowest val BCE" if use_val else "lowest train BCE (no val split)"
    print(f"[train_contact_nn] epochs={epochs}  best checkpoint: {selection} → {pt_path.name}")

    epoch_pbar = tqdm(range(epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        for batch_x, batch_y in tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{epochs}",
            leave=False,
            unit="batch",
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        train_loss, train_m = _eval_loader_metrics(
            model, train_loader, device, desc=f"Eval train {epoch + 1}/{epochs}"
        )
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_m["accuracy"])

        if use_val:
            val_loss, val_m = _eval_loader_metrics(
                model, val_loader, device, desc=f"Eval val {epoch + 1}/{epochs}"
            )
            val_loss_hist.append(val_loss)
            val_acc_hist.append(val_m["accuracy"])
            metric = val_loss
            epoch_pbar.set_postfix(
                tl=f"{train_loss:.4f}",
                vl=f"{val_loss:.4f}",
                ta=f"{train_m['accuracy']:.3f}",
                va=f"{val_m['accuracy']:.3f}",
            )
            tqdm.write(
                f"[train_contact_nn] epoch {epoch + 1}/{epochs}  "
                f"train loss={train_loss:.5f} acc={train_m['accuracy']:.4f}  "
                f"val loss={val_loss:.5f} acc={val_m['accuracy']:.4f}"
            )
        else:
            metric = train_loss
            epoch_pbar.set_postfix(tl=f"{train_loss:.4f}", ta=f"{train_m['accuracy']:.3f}")
            tqdm.write(
                f"[train_contact_nn] epoch {epoch + 1}/{epochs}  "
                f"train loss={train_loss:.5f} acc={train_m['accuracy']:.4f}  (no val)"
            )

        _save_training_curves(
            curves_path,
            train_loss_hist,
            val_loss_hist if use_val else None,
            train_acc_hist,
            val_acc_hist if use_val else None,
        )

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            torch.save({"state_dict": model.state_dict(), "meta": meta}, pt_path)
            meta["best_epoch"] = best_epoch
            meta["best_metric"] = best_metric
            meta["best_metric_name"] = "val_bce" if use_val else "train_bce"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)

            viz = cfg["visualization"]
            if test_paths and bool(viz["enabled"]):
                sample_plot_path = samples_dir / f"{stem}_epoch_{epoch + 1}.png"
                plot_random_train_test_sections(
                    train_paths=train_paths,
                    test_paths=test_paths,
                    model=model,
                    device=device,
                    scaler=scaler,
                    window_size=window,
                    n_legs=n_legs,
                    for_cnn=for_cnn,
                    num_train_sections=int(viz["num_train_sections"]),
                    num_test_sections=int(viz["num_test_sections"]),
                    dpi=int(viz["dpi"]),
                    save_path=sample_plot_path,
                    rng_seed=int(tr["seed"]) + 100_003,
                    foot_forces_by_seq=foot_forces_by_seq,
                    instants_by_seq_leg=instants_by_seq_leg,
                    sequence_uid_by_seq=sequence_uid_by_seq,
                    stance_by_seq_leg=stance_by_seq_leg,
                )
                meta["test_sample_plots_dir"] = str(samples_dir.resolve())
                meta["test_sample_plot_latest"] = str(sample_plot_path.resolve())
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                print(f"[train_contact_nn] wrote sample plot {sample_plot_path}")

    meta["training_history"] = {
        "train_loss": train_loss_hist,
        "train_accuracy": train_acc_hist,
        "val_loss": val_loss_hist if use_val else None,
        "val_accuracy": val_acc_hist if use_val else None,
    }
    meta["training_curves_plot"] = str(curves_path.resolve())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[train_contact_nn] best checkpoint epoch={best_epoch} metric={best_metric:.5f}")
    print(f"[train_contact_nn] wrote {pt_path}")
    print(f"[train_contact_nn] wrote {meta_path}")
    print(f"[train_contact_nn] wrote {scaler_path}")
    print(f"[train_contact_nn] wrote {curves_path}")

    if test_paths:
        test_ds = build_sliding_window_datasets(
            test_paths,
            n_legs,
            scaler,
            window,
            for_cnn=for_cnn,
            foot_forces_by_seq=foot_forces_by_seq,
            instants_by_seq_leg=instants_by_seq_leg,
            sequence_uid_by_seq=sequence_uid_by_seq,
            stance_by_seq_leg=stance_by_seq_leg,
        )
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
        try:
            ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        _, y_true_t, y_pred_t = _eval_loader_predictions(model, test_loader, device, desc="Eval test")
        test_metrics = {
            "accuracy": float(accuracy_score(y_true_t, y_pred_t)),
            "precision": float(precision_score(y_true_t, y_pred_t, zero_division=0)),
            "recall": float(recall_score(y_true_t, y_pred_t, zero_division=0)),
            "f1": float(f1_score(y_true_t, y_pred_t, zero_division=0)),
        }
        cls_details = _binary_eval_details(y_true_t, y_pred_t)
        _save_eval_metrics_csv(eval_csv_path, "test", cls_details)
        print(f"[train_contact_nn] test metrics: {test_metrics}")
        meta["test_metrics"] = test_metrics
        meta["eval_classification_details"] = cls_details
        meta["eval_metrics_csv"] = str(eval_csv_path.resolve())
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[train_contact_nn] wrote {eval_csv_path}")
    elif pt_path.is_file():
        try:
            ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        split_name = "val" if use_val and val_loader is not None else "train"
        loader = val_loader if split_name == "val" else train_loader
        _, y_te, y_pr = _eval_loader_predictions(model, loader, device, desc=f"Eval {split_name}")
        cls_details = _binary_eval_details(y_te, y_pr)
        _save_eval_metrics_csv(eval_csv_path, split_name, cls_details)
        meta["eval_classification_details"] = cls_details
        meta["eval_metrics_csv"] = str(eval_csv_path.resolve())
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[train_contact_nn] wrote {eval_csv_path} (split={split_name}, no test set)")


if __name__ == "__main__":
    main()
