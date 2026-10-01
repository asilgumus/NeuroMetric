"""Exploratory OASIS-1 AD-cohort sensitivity test, never training or diagnosis."""
import argparse
import json
import os
from pathlib import Path
import re
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from brainage import bootstrap, check_volume, sha256, write_json
from v3_model import SFCN, expected_age
from v3_prepare import ensure_fast, ensure_synthstrip, sfcn_array, save_qc
from v3_train import normalize_sfcn_array

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "artifacts/alzheimer/model_lock.json"


def eligible_clinical(frame):
    frame = frame.copy()
    required = {"ID", "Age", "M/F", "CDR"}
    if not required <= set(frame.columns):
        raise ValueError(f"Missing clinical columns: {required - set(frame.columns)}")
    for name in ("Age", "CDR"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame[frame.ID.str.fullmatch(r"OAS1_\d{4}_MR1", na=False) &
                  frame.Age.between(60, 100) & frame.CDR.isin([0., .5, 1., 2.]) &
                  frame["M/F"].isin(["M", "F"])].copy()
    if frame.ID.duplicated().any():
        raise ValueError("Duplicate baseline clinical subjects")
    frame["group"] = np.where(frame.CDR == 0, "control", "dementia_AD_cohort")
    return frame


def matched_manifest(clinical, images, caliper=3):
    frame = eligible_clinical(clinical)
    frame["source_path"] = frame.ID.map(images)
    missing = frame[frame.source_path.isna()].ID.tolist()
    frame = frame.dropna(subset=["source_path"])
    controls = frame[frame.group == "control"].copy()
    cases = frame[frame.group != "control"].sort_values(["Age", "ID"])
    selected = []
    for _, case in cases.iterrows():
        pool = controls[controls["M/F"] == case["M/F"]].copy()
        pool["difference"] = abs(pool.Age - case.Age)
        pool = pool[pool.difference <= caliper].sort_values(["difference", "ID"])
        if pool.empty:
            continue
        control = pool.iloc[0]
        for row in (case, control):
            selected.append({"subject_id": row.ID, "age": row.Age, "sex": row["M/F"],
                             "cdr": row.CDR, "group": row.group, "pair_id": case.ID,
                             "source_path": str(row.source_path)})
        controls = controls[controls.ID != control.ID]
    return pd.DataFrame(selected, columns=["subject_id", "age", "sex", "cdr", "group",
                                          "pair_id", "source_path"]), missing


def image_index(root):
    images = {}
    for p in sorted(root.rglob("*")):
        # Use one raw T1 acquisition per baseline person, never 2D PNGs or
        # OASIS's t88 processed volumes (not our MNI preprocessing).
        if not p.is_file() or "RAW" not in p.parts:
            continue
        match = re.fullmatch(r"(OAS1_\d{4}_MR1)_mpr-1_anon\.(?:img|img.gz|nii|nii.gz)", p.name)
        if match:
            key = match.group(1)
            if key in images:
                raise ValueError(f"Ambiguous raw MRI for {key}")
            images[key] = p.resolve()
    return images


def matched_statistics(predictions, replicates=5000):
    if predictions.subject_id.duplicated().any():
        raise ValueError("Repeated participants in inference results")
    paired = predictions.pivot(index="pair_id", columns="group", values="brain_age_gap")
    if not {"control", "dementia_AD_cohort"} <= set(paired.columns):
        raise ValueError("Both groups are required")
    complete = paired.dropna()
    if len(complete) < 5:
        raise ValueError("At least five complete pairs required; not clinical validation")
    delta = (complete.dementia_AD_cohort - complete.control).to_numpy()
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(delta, size=(replicates, len(delta)), replace=True).mean(1)
    return {"complete_pairs": len(delta), "incomplete_pairs_excluded": len(paired) - len(delta),
            "mean_paired_gap_difference_years": float(delta.mean()),
            "bootstrap_95_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
            "bootstrap_replicates": replicates, "seed": 42,
            "interpretation": "Exploratory AD-cohort vs control comparison; not diagnosis",
            "no_true_brain_age_labels": True}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--clinical", type=Path, required=True)
    manifest.add_argument("--raw-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, default=ROOT / "artifacts/alzheimer/cohort")
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, default=ROOT / "artifacts/alzheimer/test")
    run.add_argument("--cache", type=Path, default=ROOT / ".cache/brainage")
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--allow-exploratory-overlap", action="store_true")
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "manifest":
        cohort, missing = matched_manifest(pd.read_csv(args.clinical), image_index(args.raw_root))
        if cohort.empty:
            raise ValueError("No matched raw baseline T1 pairs; check RAW files and clinical CSV")
        cohort.to_csv(args.output / "manifest.csv", index=False)
        write_json(args.output / "cohort.json", {"clinical_sha256": sha256(args.clinical),
                   "age_caliper_years": 3, "minimum_age": 60, "pairs": len(cohort) // 2,
                   "missing_images": missing, "matcher": "deterministic greedy age/sex without replacement",
                   "group_label": "CDR-defined dementia in OASIS-1 AD cohort, not biomarker-confirmed AD"})
        return
    lock = json.loads(LOCK.read_text())
    if lock["pretraining_overlap_status"] != "verified_disjoint" and not args.allow_exploratory_overlap:
        raise ValueError("Pretraining overlap not verified; explicitly allow exploratory-only evaluation")
    checkpoint = ROOT / lock["checkpoint"]
    if sha256(checkpoint) != lock["sha256"]:
        raise ValueError("Locked checkpoint changed")
    cohort = pd.read_csv(args.manifest)
    if cohort.subject_id.duplicated().any() or not cohort.group.isin(["control", "dementia_AD_cohort"]).all():
        raise ValueError("Invalid cohort")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("model") != "sfcn":
        raise ValueError("Expected SFCN checkpoint")
    model = SFCN().to(device).eval()
    model.load_state_dict(state["model_state"], strict=True)
    bootstrap(args.cache, need_fsl=True, member=1)
    ensure_fast(args.cache)
    ensure_synthstrip(args.cache)
    predictions, failures = [], []
    if args.resume and (args.output / "report.json").exists():
        previous = json.loads((args.output / "report.json").read_text())
        if previous["model_lock"]["sha256"] != lock["sha256"]:
            raise ValueError("Cannot resume with different model")
        if (args.output / "predictions.csv").stat().st_size > 1:
            predictions = pd.read_csv(args.output / "predictions.csv").to_dict("records")
        if (args.output / "failures.json").exists():
            failures = json.loads((args.output / "failures.json").read_text())
    completed = {row["subject_id"] for row in predictions + failures}
    for row in cohort.to_dict("records"):
        if row["subject_id"] in completed:
            continue
        try:
            source = Path(row["source_path"])
            with tempfile.TemporaryDirectory(prefix="oasis-test-") as folder:
                folder = Path(folder)
                raw = folder / "raw.nii.gz"
                volume = nib.load(str(source))
                data = volume.get_fdata(dtype=np.float32)
                if data.ndim == 4 and data.shape[-1] == 1:
                    data = data[..., 0]
                if data.ndim != 3:
                    raise ValueError("Expected one 3D T1 volume, not a time series")
                nib.save(nib.Nifti1Image(data, volume.affine), raw)
                check_volume(raw)
                array = sfcn_array(raw, folder / "prepared", args.cache / "fsl")
                qc = args.output / "qc" / f"{row['subject_id']}.png"
                qc.parent.mkdir(parents=True, exist_ok=True)
                save_qc(array, qc, row["subject_id"] + " | registration QC, not attention")
                x = torch.from_numpy(normalize_sfcn_array(array)[None, None].copy()).to(device)
                with torch.inference_mode():
                    prediction = float(expected_age(model(x)).cpu()[0])
                if not np.isfinite(prediction):
                    raise ValueError("Non-finite prediction")
                predictions.append({**row, "prediction": prediction,
                                    "brain_age_gap": prediction - row["age"],
                                    "source_sha256": sha256(source), "automatic_qc_passed": True})
        except Exception as exc:
            failures.append({"subject_id": row["subject_id"], "error": str(exc)})
        pd.DataFrame(predictions).to_csv(args.output / "predictions.csv", index=False)
        write_json(args.output / "failures.json", failures)
        print(f"OASIS_TEST passed={len(predictions)} failed={len(failures)}", flush=True)
    summary = {"research_only": True, "clinical_diagnosis": False, "model_lock": lock,
               "manifest_sha256": sha256(args.manifest), "attempted": len(cohort),
               "passed": len(predictions), "failed": len(failures),
               "visual_qc_status": "human_review_required", "analysis_status": "pending"}
    # Registration panels require visual review before interpreting group results.
    if predictions:
        try:
            summary["provisional_statistics"] = matched_statistics(pd.DataFrame(predictions))
        except ValueError as exc:
            summary["analysis_error"] = str(exc)
    write_json(args.output / "report.json", summary)


if __name__ == "__main__":
    main()
