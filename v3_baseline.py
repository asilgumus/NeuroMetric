"""Provisional zero-shot validation inference, with no test-set access or fitting."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brainage import bootstrap, load_model, sha256, write_json
from v3_model import SFCN
from v3_train import Volumes, inference, metric, read_ready

SFCN_SHA256 = "f9197c92486a8dba7fd87bb9f2ecc4f42f7c9628b9150d1b95c8a5ac10721ad3"


def select_validation(rows):
    selected = rows.loc[rows.split == "val"].copy().reset_index(drop=True)
    if selected.empty or not set(selected.dataset).issubset({"IXI", "SALD", "NIMH"}):
        raise ValueError("Expected nonempty training-cohort validation only")
    return selected


def released_sfcn_age(logits):
    if logits.ndim != 2 or logits.shape[1] != 40:
        raise ValueError("The released UKB head must have 40 bins")
    centers = torch.arange(40, device=logits.device, dtype=torch.float32) + 42.5
    return (logits.float().softmax(1) * centers).sum(1)


def run(args):
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, roots = read_ready(args.catalog, args.prepared_root)
    rows = select_validation(rows)
    result = rows[["subject_id", "dataset", "site", "age", "split"]].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    members = []
    loader = DataLoader(Volumes(rows, roots, "resnet"), batch_size=1, num_workers=0)
    for member in range(5):
        weights, _ = bootstrap(args.cache, member=member)
        model = load_model(weights, device)
        actual, values = inference(model, loader, "resnet", device)
        if not np.allclose(actual, result.age.to_numpy()):
            raise ValueError("Validation prediction order changed")
        name = f"resnet_member_{member}"
        result[name] = values
        members.append(values)
        provenance[name] = sha256(weights)
        result.to_csv(args.output / "validation_predictions.csv", index=False)
        print("V3_ZERO_SHOT", name, metric(actual, values), flush=True)
        del model
    result["resnet_ensemble_5"] = np.stack(members).mean(axis=0)
    if sha256(args.sfcn_weights) != SFCN_SHA256:
        raise ValueError("SFCN released checkpoint checksum mismatch")
    # The original UKB bins are 42.5..81.5, as in dp_utils.num2vect.
    # Preserve the entire released classifier rather than expanding its head.
    model = SFCN(40).to(device)
    state = torch.load(args.sfcn_weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    model.eval()
    sfcn_values = []
    loader = DataLoader(Volumes(rows, roots, "sfcn"), batch_size=1, num_workers=0)
    with torch.inference_mode():
        for index, (x, age) in enumerate(loader):
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x.to(device))
            value = released_sfcn_age(logits)
            sfcn_values.extend(value.cpu().tolist())
            if (index + 1) % 25 == 0:
                print(f"V3_ZERO_SHOT sfcn {index + 1}/{len(rows)}", flush=True)
    result["sfcn_released"] = sfcn_values
    result.to_csv(args.output / "validation_predictions.csv", index=False)
    provenance["sfcn_released"] = SFCN_SHA256
    names = [*provenance, "resnet_ensemble_5"]
    write_json(args.output / "metrics.json", {
        "qc_status": "human_review_pending", "provisional": True,
        "weights_fitted": False, "model_selected": False,
        "split": "val", "held_out_predictions_computed": False,
        "catalog_sha256": sha256(args.catalog), "weights_sha256": provenance,
        "sfcn_age_bin_centers": [42.5, 81.5],
        "sfcn_output_range_limited": True,
        "overall": {name: metric(result.age, result[name]) for name in names},
        "by_dataset": {source: {name: metric(group.age, group[name]) for name in names}
                       for source, group in result.groupby("dataset")}})
    print("V3_ZERO_SHOT COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--sfcn-weights", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/brainage"))
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
