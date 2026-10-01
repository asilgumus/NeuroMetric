"""Validation-locked V3 transfer learning. Never opens test/external splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from brainage import bootstrap, load_model, sha256, write_json
from v3_model import (SFCN, expected_age, gaussian_targets, load_pretrained,
                      precision_loss, rounded_interval_loss, rounded_interval_nll)


def shift_volume(array: np.ndarray, offsets: tuple[int, int, int]) -> np.ndarray:
    """Translate with zero padding, never wrap anatomy across volume edges."""
    output = np.zeros_like(array)
    source, target = [], []
    for size, offset in zip(array.shape, offsets):
        if abs(offset) >= size:
            return output
        source.append(slice(max(0, -offset), min(size, size - offset)))
        target.append(slice(max(0, offset), min(size, size + offset)))
    output[tuple(target)] = array[tuple(source)]
    return output


def sfcn_refine_groups(model, lr_scale=1., deeper=False):
    """Unfreeze late blocks and head, optionally adding one spatial block."""
    if not np.isfinite(lr_scale) or not 0 < lr_scale <= 1:
        raise ValueError("Refinement learning-rate scale must be in (0, 1]")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = []
    if deeper:
        for parameter in model.feature_extractor.conv_3.parameters():
            parameter.requires_grad_(True)
        groups.append({"params": list(model.feature_extractor.conv_3.parameters()),
                       "lr": 1e-6 * lr_scale})
    for module, lr in ((model.feature_extractor.conv_4, 3e-6),
                       (model.feature_extractor.conv_5, 1e-5),
                       (model.classifier.conv_6, 1e-4)):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        groups.append({"params": list(module.parameters()), "lr": lr * lr_scale})
    return groups


def age_sampling_weights(rows: pd.DataFrame) -> torch.Tensor:
    """Train-only, square-root inverse decade frequency; cap ratio at two."""
    if len(rows) == 0 or not (rows.split == "train").all():
        raise ValueError("Age sampling requires nonempty training rows only")
    ages = rows.age.to_numpy(dtype=float)
    if not np.isfinite(ages).all() or (ages < 18).any() or (ages > 100).any():
        raise ValueError("Invalid training ages")
    bands = np.floor(ages / 10).astype(int)
    _, inverse, counts = np.unique(bands, return_inverse=True, return_counts=True)
    weights = np.minimum(np.sqrt(counts.max() / counts[inverse]), 2.)
    return torch.tensor(weights, dtype=torch.double)


def gamma_contrast(array: np.ndarray, gamma: float) -> np.ndarray:
    """Change positive tissue contrast while preserving background and mean scale."""
    if not np.isfinite(gamma) or not .8 <= gamma <= 1.2:
        raise ValueError("Gamma must be finite and in [0.8, 1.2]")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Contrast augmentation requires finite input")
    positive = array > 0
    if not positive.any():
        raise ValueError("Contrast augmentation requires positive tissue")
    output = array.copy()
    tissue = array[positive]
    scale = float(tissue.mean())
    transformed = np.power(tissue / scale, gamma)
    output[positive] = transformed * (scale / float(transformed.mean()))
    return output


class Volumes(Dataset):
    def __init__(self, rows: pd.DataFrame, roots: dict[str, Path], model: str,
                 augment: bool = False, spatial_augment: bool = False,
                 contrast_augment: bool = False):
        self.rows = rows.reset_index(drop=True)
        self.roots = roots
        self.column = f"{model}_array"
        self.model = model
        self.augment = augment
        self.spatial_augment = spatial_augment
        self.contrast_augment = contrast_augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        loaded = np.load(self.roots[row.dataset] / row[self.column], allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                array = loaded["x"]
        else:
            array = loaded
        expected = (160, 192, 160) if self.model == "sfcn" else (80, 96, 80)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"Invalid {self.model} input: {row.subject_id} {array.shape}")
        array = np.asarray(array, dtype=np.float32)
        if self.model == "sfcn":
            array = normalize_sfcn_array(array)
        if self.augment:
            if self.contrast_augment and np.random.random() < .5:
                array = gamma_contrast(array, float(np.random.uniform(.8, 1.2)))
            if self.spatial_augment:
                # Stored SFCN arrays retain MNI NIfTI axes: axis 0 is L/R.
                array = shift_volume(array, tuple(np.random.randint(-2, 3, size=3)))
                if np.random.random() < .5:
                    array = np.flip(array, axis=0)
            array = array * np.random.uniform(.95, 1.05)
        if self.model == "sfcn":
            array = array[None]
        return torch.from_numpy(array.copy()), torch.tensor(float(row.age), dtype=torch.float32)


def normalize_sfcn_array(array: np.ndarray) -> np.ndarray:
    """Convert stored positive-voxel scale to the released UKB SFCN scale."""
    array = np.asarray(array, dtype=np.float32)
    if array.shape != (160, 192, 160) or not np.isfinite(array).all():
        raise ValueError("Invalid SFCN volume")
    # The UKB example divides registered 182x218x182 data by whole-volume
    # mean before center cropping. The prepared brain is inside that crop.
    mean = float(array.mean())
    if mean <= 0:
        raise ValueError("Empty SFCN input")
    full_to_crop = (182 * 218 * 182) / array.size
    return array * (full_to_crop / mean)


def read_ready(catalog: Path, prepared_root: Path, sources=("IXI", "SALD", "NIMH")):
    locked = pd.read_csv(catalog)
    if locked.subject_id.duplicated().any():
        raise ValueError("Duplicate subject in locked catalogue")
    roots = {source: prepared_root / source.lower() for source in sources}
    records = []
    for source, root in roots.items():
        path = root / "manifest.csv"
        summary = root / "prepare_summary.json"
        if not path.exists() or not summary.exists() or not json.loads(summary.read_text())["complete"]:
            raise FileNotFoundError(f"Complete V3 preparation required: {root}")
        frame = pd.read_csv(path)
        if frame.subject_id.duplicated().any():
            raise ValueError(f"Duplicate {source} MRI in prepared manifest")
        records.append(frame)
    ready = pd.concat(records, ignore_index=True)
    locked = locked.loc[locked.dataset.isin(sources)]
    joined = locked[["subject_id", "dataset", "split", "age", "site"]].merge(
        ready[["subject_id", "qc_status", "resnet_array", "sfcn_array"]],
        on="subject_id", validate="one_to_one", how="left", indicator=True)
    if not (joined._merge == "both").all():
        raise ValueError("Prepared manifest does not cover locked catalogue")
    joined = joined.loc[joined.qc_status == "automatic_checks_passed"].drop(columns="_merge")
    if len(joined.loc[joined.split == "train"]) < 700:
        raise ValueError("Too few training MRIs survived QC")
    return joined, roots


def hit_selection_key(score):
    """Predeclared exact rounded hit priority, then +/-1, then lower MAE."""
    return (score["rounded_year_match"], score["within_1_year"], -score["mae"])


def metric(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.ndim != 1 or pred.shape != y.shape or len(y) == 0:
        raise ValueError("Age and prediction must be non-empty aligned vectors")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Age and prediction metrics require finite values")
    errors = pred - y
    denominator = np.square(y - y.mean()).sum()
    return {"n": len(y), "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
            "r2": float(1 - np.square(errors).sum() / denominator) if denominator > 0 else None,
            "within_1_year": float((np.abs(errors) <= 1).mean()),
            "within_5_years": float((np.abs(errors) <= 5).mean()),
            "within_10_years": float((np.abs(errors) <= 10).mean()),
            "rounded_year_match": float((np.rint(y) == np.rint(pred)).mean()),
            "mean_gap": float(errors.mean())}


def inference(model, loader, architecture, device):
    model.eval()
    predictions, ages = [], []
    with torch.inference_mode():
        for x, age in loader:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                output = model(x.to(device))
                pred = expected_age(output) if architecture == "sfcn" else output[0].flatten()
            predictions.extend(pred.float().cpu().tolist())
            ages.extend(age.tolist())
    return np.asarray(ages), np.asarray(predictions)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("V3 full-volume training requires a CUDA GPU")
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    rows, roots = read_ready(args.catalog, args.prepared_root)
    train_rows = rows.loc[rows.split == "train"].reset_index(drop=True)
    val_rows = rows.loc[rows.split == "val"].reset_index(drop=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.model == "sfcn":
        if sha256(args.sfcn_weights) != args.sfcn_sha256:
            raise ValueError("SFCN pretrained checkpoint checksum mismatch")
        model = load_pretrained(args.sfcn_weights, device)
        base_hash = args.sfcn_sha256
    else:
        weights, provenance = bootstrap(args.cache, member=args.member)
        model = load_model(weights, device)
        base_hash = sha256(weights)
    parent_hash = None
    if args.warm_start:
        parent_hash = sha256(args.warm_start)
        if parent_hash != args.warm_start_sha256:
            raise ValueError("Warm-start checkpoint checksum mismatch")
        parent = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        if (parent.get("model") != args.model or
                parent.get("config", {}).get("catalog_sha256") != sha256(args.catalog) or
                parent.get("config", {}).get("base_weights_sha256") != base_hash):
            raise ValueError("Warm-start architecture, catalogue or base weights differ")
        model.load_state_dict(parent["model_state"], strict=True)
    head_lr = 1e-3 if args.model == "sfcn" else 1e-4
    config = {"model": args.model, "seed": 42, "catalog_sha256": sha256(args.catalog),
              "base_weights_sha256": base_hash, "train_n": len(train_rows),
              "val_n": len(val_rows), "test_used_for_selection": False,
              "epochs_head": args.head_epochs, "epochs_block": args.block_epochs,
              "patience": args.patience, "warm_start_sha256": parent_hash,
              "optimizer_reinitialized": bool(args.warm_start),
              "learning_rates": {"head": head_lr, "block": 1e-5},
              "sfcn_normalization": "whole_MNI_volume_mean_before_center_crop"
              if args.model == "sfcn" else None}
    config["sfcn_refine"] = args.sfcn_refine
    config["sfcn_deeper_refine"] = args.sfcn_deeper_refine
    config["age_balanced_sampling"] = args.age_balanced_sampling
    config["sfcn_contrast_augment"] = args.sfcn_contrast_augment
    config["sfcn_target_sigma"] = args.sfcn_target_sigma
    config["sfcn_precision_loss"] = args.sfcn_precision_loss
    config["sfcn_rounded_interval_loss"] = args.sfcn_rounded_interval_loss
    config["sfcn_rounded_interval_nll"] = args.sfcn_rounded_interval_nll
    config["loss"] = {"kind": "gaussian_KL_plus_MAE", "target_sigma": args.sfcn_target_sigma,
                      "mae_weight": .05, "train_only": True, "decoder": "expected_age"}
    if args.sfcn_precision_loss:
        config["loss"].update({"kind": "gaussian_KL_plus_MAE_plus_Welsch",
                              "precision_weight": .25, "precision_scale_years": .5})
    if args.sfcn_rounded_interval_loss:
        config["loss"].update({"kind": "gaussian_KL_plus_MAE_plus_rounded_interval",
                              "interval_weight": .25, "interval_temperature_years": .25,
                              "interval_center": "ties_to_even_rounded_chronological_age"})
    if args.sfcn_rounded_interval_nll:
        config["loss"].update({"kind": "gaussian_KL_plus_MAE_plus_rounded_interval_NLL",
                              "interval_weight": .25, "interval_temperature_years": .25,
                              "nll_scale_years": .25,
                              "interval_center": "ties_to_even_rounded_chronological_age"})
    sampler = None
    if args.age_balanced_sampling:
        weights = age_sampling_weights(train_rows)
        sampler = WeightedRandomSampler(weights, len(train_rows), replacement=True,
                                        generator=torch.Generator().manual_seed(42))
        config["sampling"] = {"kind": "capped_sqrt_inverse_decade", "max_ratio": 2.,
                              "replacement": True, "draws_per_epoch": len(train_rows),
                              "weight_min": float(weights.min()), "weight_max": float(weights.max())}
    if args.sfcn_refine:
        config.update({"refine_lr_scale": args.refine_lr_scale,
                       "learning_rates": {"conv_4": 3e-6 * args.refine_lr_scale,
                                          "conv_5": 1e-5 * args.refine_lr_scale,
                                          "head": 1e-4 * args.refine_lr_scale},
                       "scheduler": {"kind": "ReduceLROnPlateau", "patience": 5,
                                     "factor": .5, "threshold": 0., "min_lr": 1e-7},
                       "augmentation": {"shift_voxels": 2, "mirror_axis": 0,
                                        "mirror_probability": .5, "intensity_range": [.95, 1.05]}})
        if args.sfcn_deeper_refine:
            config["learning_rates"]["conv_3"] = 1e-6 * args.refine_lr_scale
        if args.sfcn_contrast_augment:
            config["augmentation"]["gamma_contrast"] = {
                "range": [.8, 1.2], "probability": .5,
                "preserve_positive_mean": True, "train_only": True}
    write_json(output / "config.json", config)
    train_loader = DataLoader(Volumes(train_rows, roots, args.model, True,
                                     args.sfcn_refine, args.sfcn_contrast_augment), batch_size=args.batch,
                              shuffle=sampler is None, sampler=sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(Volumes(val_rows, roots, args.model), batch_size=args.batch,
                            num_workers=args.workers, pin_memory=True)
    # Measure the exact transfer-learning starting point before any updates.
    # SFCN's enlarged age head is an initialization, not the untouched UKB head.
    actual, predicted = inference(model, val_loader, args.model, device)
    baseline = {"model": args.model, "kind": "transfer_initialization",
                "base_weights_sha256": base_hash, "split": "val",
                "sfcn_head_adapted": args.model == "sfcn", **metric(actual, predicted)}
    write_json(output / "baseline_metrics.json", baseline)
    pd.DataFrame({"subject_id": val_rows.subject_id, "age": actual,
                  "prediction": predicted}).to_csv(
                      output / "baseline_validation_predictions.csv", index=False)
    print("V3_BASELINE", baseline, flush=True)
    history = []
    best = baseline["mae"]
    best_hits = hit_selection_key(baseline)
    hit_metrics = dict(baseline)
    # Retain the starting checkpoint if further training never improves it.
    torch.save({"model_state": model.state_dict(), "model": args.model,
                "stage": "initialization", "epoch": 0,
                "validation_mae": best, "config": config}, output / "best.pt")
    torch.save({"model_state": model.state_dict(), "model": args.model,
                "stage": "initialization", "epoch": 0,
                "validation_mae": best, "config": config}, output / "best_hits.pt")
    pd.DataFrame({"subject_id": val_rows.subject_id, "age": actual,
                  "prediction": predicted}).to_csv(output / "best_hits_validation_predictions.csv", index=False)
    pd.DataFrame({"subject_id": val_rows.subject_id, "age": actual,
                  "prediction": predicted}).to_csv(
                      output / "best_validation_predictions.csv", index=False)
    for stage, epochs in (("head", args.head_epochs), ("block", args.block_epochs)):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if args.model == "sfcn":
            head = model.classifier.conv_6
            block = model.feature_extractor.conv_5
        else:
            head = model.fc1
            block = model.features.features[7]
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        if stage == "block":
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        groups = [{"params": head.parameters(), "lr": head_lr}]
        if stage == "block":
            groups.append({"params": block.parameters(), "lr": 1e-5})
        if args.sfcn_refine:
            groups = sfcn_refine_groups(model, args.refine_lr_scale, args.sfcn_deeper_refine)
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=.5, patience=5, threshold=0.,
            threshold_mode="abs", min_lr=1e-7) if args.sfcn_refine else None)
        scaler = torch.amp.GradScaler("cuda")
        stale = 0
        for epoch in range(1, epochs + 1):
            model.eval()  # preserve pretrained batch-normalization statistics
            if args.model == "sfcn":
                model.classifier.dropout.train()
            optimizer.zero_grad(set_to_none=True)
            running = 0.; observed = 0
            for index, (x, age) in enumerate(train_loader):
                x, age = x.to(device), age.to(device)
                with torch.autocast("cuda"):
                    output_tensor = model(x)
                    if args.model == "sfcn":
                        kl = nn.functional.kl_div(nn.functional.log_softmax(output_tensor.float(), 1),
                                                  gaussian_targets(age, args.sfcn_target_sigma), reduction="batchmean")
                        loss = kl + .05 * nn.functional.l1_loss(expected_age(output_tensor.float()), age)
                        if args.sfcn_precision_loss:
                            loss = loss + .25 * precision_loss(expected_age(output_tensor.float()), age)
                        if args.sfcn_rounded_interval_loss:
                            loss = loss + .25 * rounded_interval_loss(expected_age(output_tensor.float()), age)
                        if args.sfcn_rounded_interval_nll:
                            loss = loss + .25 * rounded_interval_nll(expected_age(output_tensor.float()), age)
                    else:
                        loss = nn.functional.l1_loss(output_tensor[0].flatten(), age)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite V3 training loss")
                window_start = (index // args.accumulation) * args.accumulation
                window_size = min(args.accumulation, len(train_loader) - window_start)
                scaler.scale(loss / window_size).backward()
                if (index + 1) % args.accumulation == 0 or index + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running += float(loss.detach()) * len(age); observed += len(age)
            actual, predicted = inference(model, val_loader, args.model, device)
            score = metric(actual, predicted)
            item = {"stage": stage, "epoch": epoch, "train_loss": running / observed,
                    **{f"val_{key}": value for key, value in score.items()}}
            if scheduler is not None:
                item.update({f"lr_group_{index}": group["lr"]
                             for index, group in enumerate(optimizer.param_groups)})
                scheduler.step(score["mae"])
            history.append(item)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            pd.DataFrame({"subject_id": val_rows.subject_id, "age": actual,
                          "prediction": predicted}).to_csv(output / "validation_predictions.csv", index=False)
            state = {"model_state": model.state_dict(), "model": args.model,
                     "stage": stage, "epoch": epoch, "validation_mae": score["mae"],
                     "config": config}
            torch.save(state, output / "last.pt")
            print("V3_EPOCH", item, flush=True)
            if hit_selection_key(score) > best_hits:
                best_hits = hit_selection_key(score)
                hit_metrics = dict(score)
                torch.save(state, output / "best_hits.pt")
                pd.DataFrame({"subject_id": val_rows.subject_id, "age": actual,
                              "prediction": predicted}).to_csv(
                                  output / "best_hits_validation_predictions.csv", index=False)
            if score["mae"] < best:
                best = score["mae"]; stale = 0
                torch.save(state, output / "best.pt")
                pd.DataFrame({"subject_id": val_rows.subject_id, "age": actual,
                              "prediction": predicted}).to_csv(output / "best_validation_predictions.csv", index=False)
            else:
                stale += 1
            if stale >= args.patience:
                break
        if stage == "head":
            model.load_state_dict(torch.load(output / "best.pt", map_location="cpu",
                                             weights_only=True)["model_state"])
    write_json(output / "selection.json", {"validation_mae": best,
               "checkpoint": "best.pt", "test_used_for_selection": False})
    write_json(output / "hit_selection.json", {
        "priority": ["rounded_year_match", "within_1_year", "lower_mae"],
        "checkpoint": "best_hits.pt", "validation_metrics": hit_metrics,
        "test_used_for_selection": False,
        "legacy_mae_checkpoint_preserved": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("sfcn", "resnet"), required=True)
    parser.add_argument("--sfcn-weights", type=Path)
    parser.add_argument("--sfcn-sha256")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/brainage"))
    parser.add_argument("--member", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--block-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--warm-start-sha256")
    parser.add_argument("--sfcn-refine", action="store_true")
    parser.add_argument("--sfcn-deeper-refine", action="store_true")
    parser.add_argument("--age-balanced-sampling", action="store_true")
    parser.add_argument("--sfcn-contrast-augment", action="store_true")
    parser.add_argument("--sfcn-target-sigma", type=float, default=2.)
    parser.add_argument("--sfcn-precision-loss", action="store_true")
    parser.add_argument("--sfcn-rounded-interval-loss", action="store_true")
    parser.add_argument("--sfcn-rounded-interval-nll", action="store_true")
    parser.add_argument("--refine-lr-scale", type=float, default=1.)
    args = parser.parse_args()
    if args.sfcn_rounded_interval_nll and (not args.sfcn_refine or args.sfcn_precision_loss
                                         or args.sfcn_rounded_interval_loss):
        parser.error("Interval NLL requires refinement and replaces other precision auxiliaries")
    if args.sfcn_rounded_interval_loss and (not args.sfcn_refine or args.sfcn_precision_loss):
        parser.error("Rounded interval loss requires refinement and replaces precision loss")
    if args.sfcn_precision_loss and not args.sfcn_refine:
        parser.error("Precision loss requires SFCN refinement")
    if (not np.isfinite(args.sfcn_target_sigma) or not .5 <= args.sfcn_target_sigma <= 2.
            or (args.sfcn_target_sigma != 2. and not args.sfcn_refine)):
        parser.error("Target sigma must be in [0.5, 2]; nondefault requires SFCN refinement")
    if args.sfcn_contrast_augment and not args.sfcn_refine:
        parser.error("Contrast augmentation requires SFCN refinement")
    if args.sfcn_deeper_refine and not args.sfcn_refine:
        parser.error("Deeper refinement requires SFCN refinement")
    if args.age_balanced_sampling and not args.sfcn_refine:
        parser.error("Age-balanced sampling requires SFCN refinement")
    if (not np.isfinite(args.refine_lr_scale) or not 0 < args.refine_lr_scale <= 1
            or (args.refine_lr_scale != 1. and not args.sfcn_refine)):
        parser.error("Learning-rate scale requires refinement and must be in (0, 1]")
    if bool(args.warm_start) != bool(args.warm_start_sha256):
        parser.error("Warm-start path and SHA-256 must be supplied together")
    if args.sfcn_refine and (args.model != "sfcn" or not args.warm_start or args.head_epochs != 0):
        parser.error("SFCN refinement requires SFCN, a warm-start checkpoint and zero head epochs")
    if min(args.head_epochs, args.block_epochs) < 0 or args.head_epochs + args.block_epochs < 1 or args.patience < 1:
        parser.error("Epoch counts must be nonnegative with positive total and patience")
    if args.model == "sfcn" and (not args.sfcn_weights or not args.sfcn_sha256):
        parser.error("SFCN weights and SHA-256 are required")
    train(args)


if __name__ == "__main__":
    main()
