"""Locked single-checkpoint held-out evaluation. Never selects or trains models."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from brainage import sha256, write_json
from v3_model import SFCN
from v3_train import Volumes, inference, metric, read_ready


def heldout_groups(rows):
    groups = {"ixi_test": rows[(rows.dataset == "IXI") & (rows.split == "test")],
              "dlbs_external": rows[(rows.dataset == "DLBS") & (rows.split == "external")]}
    if len(groups["ixi_test"]) < 70 or len(groups["dlbs_external"]) < 350:
        raise ValueError("Incomplete held-out cohorts")
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.checkpoint) != args.sha256:
        raise ValueError("Locked checkpoint mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for full-volume evaluation")
    rows, roots = read_ready(args.catalog, args.prepared_root, sources=("IXI", "SALD", "NIMH", "DLBS"))
    groups = heldout_groups(rows)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("model") != "sfcn":
        raise ValueError("Expected SFCN")
    device = torch.device("cuda")
    model = SFCN().to(device).eval()
    model.load_state_dict(state["model_state"], strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint_sha256": args.sha256, "checkpoint_epoch": state["epoch"],
              "test_used_for_selection": False, "research_only": True,
              "restriction": "Held-out results must not guide subsequent tuning", "holdout": {}}
    for name, subset in groups.items():
        loader = DataLoader(Volumes(subset, roots, "sfcn"), batch_size=2,
                            num_workers=2, pin_memory=True)
        actual, predicted = inference(model, loader, "sfcn", device)
        frame = subset.copy()
        frame["prediction"] = predicted
        frame.to_csv(args.output / (name + "_predictions.csv"), index=False)
        score = metric(actual, predicted)
        score["within_1_count"] = int((np.abs(actual-predicted) <= 1).sum())
        score["rounded_match_count"] = int((np.rint(actual) == np.rint(predicted)).sum())
        report["holdout"][name] = score
        write_json(args.output / "metrics.json", report)
        print("HELDOUT", name, score, flush=True)
    write_json(args.output / "complete.json", {"complete": True, "checkpoint_sha256": args.sha256})


if __name__ == "__main__":
    main()
