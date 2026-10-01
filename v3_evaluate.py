"""Select V3 on locked IXI validation, then open IXI test and Dallas external."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brainage import bootstrap, load_model, sha256, write_json
from v3_model import SFCN
from v3_train import Volumes, inference, metric, read_ready


def load_v3(path: Path, kind: str, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("model") != kind:
        raise ValueError(f"Wrong V3 architecture in {path}")
    if kind == "sfcn":
        model = SFCN().to(device)
    else:
        base, _ = bootstrap(Path("/tmp/brainage"), member=1)
        model = load_model(base, device)
    model.load_state_dict(state["model_state"], strict=True)
    return model.eval()


def predict_split(rows, roots, sfcn, resnet, v2_paths, device, batch):
    frame = rows.copy().reset_index(drop=True)
    for kind, model in (("sfcn", sfcn), ("resnet", resnet)):
        loader = DataLoader(Volumes(frame, roots, kind), batch_size=batch,
                            num_workers=2, pin_memory=True)
        _, pred = inference(model, loader, kind, device)
        frame[kind] = pred
    loader = DataLoader(Volumes(frame, roots, "resnet"), batch_size=batch,
                        num_workers=2, pin_memory=True)
    members = []
    for path in v2_paths:
        model = load_model(path, device)
        _, values = inference(model, loader, "resnet", device)
        members.append(values)
        del model
    frame["v2"] = np.stack(members).mean(axis=0)
    return frame


def ci_mae(y, pred, seed=42, draws=2000):
    rng = np.random.default_rng(seed)
    losses = np.abs(np.asarray(pred) - np.asarray(y))
    samples = rng.integers(0, len(losses), (draws, len(losses)))
    return [float(x) for x in np.quantile(losses[samples].mean(axis=1), [.025, .975])]


def paired_mae_difference(y, candidate, baseline, seed=42, draws=2000):
    """Subject-paired ΔMAE; negative values favour the selected candidate."""
    y = np.asarray(y, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    metric(y, candidate)
    metric(y, baseline)
    difference = np.abs(candidate - y) - np.abs(baseline - y)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(y), (draws, len(y)))
    ci = np.quantile(difference[samples].mean(axis=1), [.025, .975])
    return {"delta_mae_years": float(difference.mean()),
            "delta_mae_ci95": [float(value) for value in ci],
            "negative_favours": "selected"}


def assess(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Full-volume V3 assessment requires a CUDA GPU")
    device = torch.device("cuda")
    rows, roots = read_ready(args.catalog, args.prepared_root,
                             sources=("IXI", "SALD", "NIMH", "DLBS"))
    training_ages = rows.loc[rows.split == "train", "age"]
    train_age_range = [float(training_ages.min()), float(training_ages.max())]
    args.output.mkdir(parents=True, exist_ok=True)
    sfcn = load_v3(args.sfcn_checkpoint, "sfcn", device)
    resnet = load_v3(args.resnet_checkpoint, "resnet", device)
    v2_paths = sorted(args.v2_dir.glob("member_*.pt"))
    if len(v2_paths) != 4:
        raise ValueError("Expected the four validation-selected V2 members")
    val = rows.loc[(rows.dataset == "IXI") & (rows.split == "val")]
    if len(val) < 70:
        raise ValueError("Too few IXI validation cases survived QC")
    if len(rows.loc[(rows.dataset == "IXI") & (rows.split == "test")]) < 70:
        raise ValueError("Too few IXI held-out test cases survived QC")
    if len(rows.loc[rows.dataset == "DLBS"]) < 350:
        raise ValueError("Too few Dallas external cases survived QC")
    val_predictions = predict_split(val, roots, sfcn, resnet, v2_paths, device, args.batch)
    choices = {}
    for count in (1, 2, 3):
        for names in combinations(("v2", "sfcn", "resnet"), count):
            candidate = "+".join(names)
            values = val_predictions[list(names)].mean(axis=1)
            choices[candidate] = metric(val_predictions.age, values)
    # Explicit epsilon tie handling: MAE primary, ±1 year secondary.
    best_mae = min(row["mae"] for row in choices.values())
    eligible = [name for name, row in choices.items() if row["mae"] <= best_mae + 1e-8]
    winner = sorted(eligible, key=lambda key: (-choices[key]["within_1_year"], key))[0]
    selection = {"candidate": winner, "validation": choices,
                 "selection_rule": "lowest IXI validation MAE, ±1y tie-break",
                 "test_used_for_selection": False,
                 "train_age_range": train_age_range,
                 "catalog_sha256": sha256(args.catalog),
                 "checkpoints": {"sfcn": sha256(args.sfcn_checkpoint),
                                 "resnet": sha256(args.resnet_checkpoint),
                                 "v2": [sha256(path) for path in v2_paths]}}
    write_json(args.output / "selection.json", selection)
    val_predictions["selected"] = val_predictions[winner.split("+")].mean(axis=1)
    val_predictions.to_csv(args.output / "ixi_validation_predictions.csv", index=False)
    report = {"selected": winner, "validation": choices,
              "train_age_range": train_age_range, "holdout": {}}
    for name, subset in (("ixi_test", rows.loc[(rows.dataset == "IXI") & (rows.split == "test")]),
                         ("dlbs_external", rows.loc[rows.dataset == "DLBS"])):
        predictions = predict_split(subset, roots, sfcn, resnet, v2_paths, device, args.batch)
        predictions["selected"] = predictions[winner.split("+")].mean(axis=1)
        predictions["outside_train_age_range"] = ~predictions.age.between(*train_age_range)
        predictions.to_csv(args.output / f"{name}_predictions.csv", index=False)
        report["holdout"][name] = {}
        for key in ("v2", "sfcn", "resnet", "selected"):
            result = metric(predictions.age, predictions[key])
            result["mae_ci95"] = ci_mae(predictions.age, predictions[key])
            report["holdout"][name][key] = result
        report["holdout"][name]["selected_vs_v2"] = paired_mae_difference(
            predictions.age, predictions.selected, predictions.v2)
        report["holdout"][name]["age_ranges"] = {}
        predictions["age_band"] = pd.cut(predictions.age, [18, 30, 40, 50, 60, 70, 80, 101],
                                          right=False).astype(str)
        for band, group in predictions.groupby("age_band"):
            if len(group) > 1:
                report["holdout"][name]["age_ranges"][band] = metric(group.age, group.selected)
        outside = predictions.loc[predictions.outside_train_age_range]
        report["holdout"][name]["outside_train_age_range_n"] = len(outside)
        if len(outside) > 1:
            report["holdout"][name]["outside_train_age_range"] = metric(
                outside.age, outside.selected)
        report["holdout"][name]["sites"] = {}
        for site, group in predictions.groupby("site"):
            if len(group) >= 5:
                report["holdout"][name]["sites"][site] = metric(group.age, group.selected)
    write_json(args.output / "metrics.json", report)
    print("V3_EVALUATION_COMPLETE", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--sfcn-checkpoint", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    assess(parser.parse_args())


if __name__ == "__main__":
    main()
