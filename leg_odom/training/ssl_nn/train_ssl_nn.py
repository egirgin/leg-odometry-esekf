"""Train self-supervised CNN/GRU backbones with an initial contrastive placeholder objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from leg_odom.features.instant_spec import FULL_OFFLINE_INSTANT_FIELDS, INSTANT_FEATURE_SPEC_VERSION, parse_instant_feature_fields
from leg_odom.run.kinematics_factory import build_kinematics_by_name
from leg_odom.training.nn.data import collect_train_instant_matrix, load_precomputed_subset_by_npz_paths
from leg_odom.training.nn.precomputed_io import discover_precomputed_instants_npz
from leg_odom.training.ssl_nn.config import default_ssl_train_config_path, load_ssl_train_config
from leg_odom.training.ssl_nn.data import build_ssl_window_dataset
from leg_odom.training.ssl_nn.loss_functions import nt_xent_loss
from leg_odom.training.ssl_nn.models import ProjectionHead, build_backbone
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
        description="Train SSL CNN/GRU backbones (YAML config + precomputed npz bundles)"
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Training YAML (default: {default_ssl_train_config_path()})",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve() if args.config else default_ssl_train_config_path()
    print(f"[train_ssl_nn] loading YAML: {cfg_path}")
    cfg: dict[str, Any] = load_ssl_train_config(cfg_path)

    device = _pick_device()
    print(f"[train_ssl_nn] device={device}")

    dataset_kind = str(cfg["dataset"]["kind"]).strip()
    precomputed_root = Path(cfg["dataset"]["precomputed_root"]).expanduser().resolve()
    dl = cfg["data_loading"]
    prep_verbose = bool(dl["verbose"])
    arch = str(cfg["architecture"]).strip().lower()

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
        f"[train_ssl_nn] dataset.kind={dataset_kind!r} architecture={arch} "
        f"precomputed_instants.npz: total={len(all_npz)} train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}"
    )

    fields = tuple(str(x).strip() for x in cfg["features"]["fields"])
    for f in fields:
        if f not in FULL_OFFLINE_INSTANT_FIELDS:
            raise ValueError(
                f"features.fields contains {f!r} which is not in FULL_OFFLINE_INSTANT_FIELDS"
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

    scaler = StandardScaler()
    train_mat = collect_train_instant_matrix(train_paths, n_legs, instants_by_seq_leg)
    scaler.fit(train_mat)

    ssl_cfg = cfg["ssl"]
    aug_cfg = ssl_cfg["augmentation"]

    train_ds = build_ssl_window_dataset(
        train_paths,
        n_legs,
        scaler,
        window,
        for_cnn=(arch == "cnn"),
        instants_by_seq_leg=instants_by_seq_leg,
        sequence_uid_by_seq=sequence_uid_by_seq,
        gaussian_noise_std=float(aug_cfg["gaussian_noise_std"]),
        feature_dropout_prob=float(aug_cfg["feature_dropout_prob"]),
        scale_jitter_std=float(aug_cfg["scale_jitter_std"]),
    )

    bs = int(tr["batch_size"])
    nw = int(tr["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)

    backbone = build_backbone(arch, spec.instant_dim, window).to(device)
    projector = ProjectionHead(in_dim=1, out_dim=int(ssl_cfg["projection_dim"])).to(device)
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=float(tr["learning_rate"]),
    )

    out_dir = Path(str(cfg["output"]["dir"])).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"contact_{arch}_ssl"
    pt_path = out_dir / f"{stem}.pt"
    meta_path = out_dir / f"{stem}_meta.json"
    scaler_path = out_dir / f"{stem}_scaler.npz"
    samples_dir = out_dir / "plots" / "samples"

    meta: dict[str, Any] = {
        "config_path": str(cfg_path),
        "dataset_kind": dataset_kind,
        "architecture": arch,
        "training_method": "self_supervised",
        "ssl_method": str(ssl_cfg["method"]),
        "instant_feature_spec_version": int(INSTANT_FEATURE_SPEC_VERSION),
        "feature_fields": list(spec.fields),
        "history_length": window,
        "instant_dim": int(spec.instant_dim),
        "robot_kinematics": str(cfg["robot"]["kinematics"]),
        "train_ratio": float(tr["train_ratio"]),
        "val_ratio": float(tr["val_ratio"]),
        "test_ratio": float(tr["test_ratio"]),
        "seed": int(tr["seed"]),
        "precomputed_root": str(precomputed_root),
        "train_precomputed_instants_npz": [str(p.resolve()) for p in train_paths],
        "val_precomputed_instants_npz": [str(p.resolve()) for p in val_paths],
        "test_precomputed_instants_npz": [str(p.resolve()) for p in test_paths],
        "ssl_temperature": float(ssl_cfg["temperature"]),
        "ssl_projection_dim": int(ssl_cfg["projection_dim"]),
        "ssl_augmentation": {
            "gaussian_noise_std": float(aug_cfg["gaussian_noise_std"]),
            "feature_dropout_prob": float(aug_cfg["feature_dropout_prob"]),
            "scale_jitter_std": float(aug_cfg["scale_jitter_std"]),
        },
    }

    epochs = int(tr["epochs"])
    best_loss = float("inf")
    best_epoch = 0
    train_loss_hist: list[float] = []

    print(f"[train_ssl_nn] epochs={epochs} best checkpoint: lowest train SSL loss")
    epoch_pbar = tqdm(range(epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        backbone.train()
        projector.train()
        running_loss = 0.0
        n_batches = 0
        for x1, x2 in tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}", leave=False, unit="batch"):
            x1 = x1.to(device)
            x2 = x2.to(device)
            optimizer.zero_grad()
            z1 = projector(backbone(x1))
            z2 = projector(backbone(x2))
            loss = nt_xent_loss(z1, z2, temperature=float(ssl_cfg["temperature"]))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        epoch_loss = running_loss / max(n_batches, 1)
        train_loss_hist.append(epoch_loss)
        epoch_pbar.set_postfix(train_ssl_loss=f"{epoch_loss:.5f}")
        tqdm.write(f"[train_ssl_nn] epoch {epoch + 1}/{epochs} train_ssl_loss={epoch_loss:.6f}")

        viz = cfg["visualization"]
        if bool(viz["enabled"]):
            sample_plot_path = samples_dir / f"{stem}_epoch_{epoch + 1}.png"
            plot_random_train_test_sections(
                train_paths=train_paths,
                test_paths=test_paths,
                model=backbone,
                device=device,
                scaler=scaler,
                window_size=window,
                n_legs=n_legs,
                for_cnn=(arch == "cnn"),
                num_train_sections=int(viz["num_train_sections"]),
                num_test_sections=int(viz["num_test_sections"]),
                dpi=int(viz["dpi"]),
                save_path=sample_plot_path,
                rng_seed=int(tr["seed"]) + 100_003 + epoch,
                foot_forces_by_seq=foot_forces_by_seq,
                instants_by_seq_leg=instants_by_seq_leg,
                sequence_uid_by_seq=sequence_uid_by_seq,
                stance_by_seq_leg=stance_by_seq_leg,
            )
            meta["test_sample_plots_dir"] = str(samples_dir.resolve())
            meta["test_sample_plot_latest"] = str(sample_plot_path.resolve())
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            tqdm.write(f"[train_ssl_nn] wrote sample plot {sample_plot_path}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "state_dict": backbone.state_dict(),
                    "ssl_projector_state_dict": projector.state_dict(),
                    "meta": meta,
                },
                pt_path,
            )
            meta["best_epoch"] = best_epoch
            meta["best_metric"] = best_loss
            meta["best_metric_name"] = "train_ssl_loss"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)

    meta["training_history"] = {"train_ssl_loss": train_loss_hist}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[train_ssl_nn] best checkpoint epoch={best_epoch} metric={best_loss:.6f}")
    print(f"[train_ssl_nn] wrote {pt_path}")
    print(f"[train_ssl_nn] wrote {meta_path}")
    print(f"[train_ssl_nn] wrote {scaler_path}")


if __name__ == "__main__":
    main()
