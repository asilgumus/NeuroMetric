"""Predict every currently prepared OASIS MRI with locked contrast-cont4."""
import json
from pathlib import Path
import sys
import fcntl

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from v3_model import SFCN, expected_age
from v3_train import normalize_sfcn_array, metric

CHECKPOINT = ROOT / "artifacts/v3/autopilot/brainage-v3-train-sfcn-contrast-cont4/outputs/training/best.pt"
HASH = "6646617b6ac64877b03fbdc94f42395ef22974240237121e0d9dd9e3d53abc40"


def rank_predictions(frame):
    ranked = frame.copy()
    ranked["absolute_chronological_error"] = abs(ranked.prediction - ranked.age)
    ranked["within_1_year"] = ranked.absolute_chronological_error <= 1
    ranked["rounded_year_match"] = np.rint(ranked.prediction) == np.rint(ranked.age)
    return ranked.sort_values(["absolute_chronological_error", "subject_id"])


def main():
    torch.set_num_threads(2)
    output = ROOT / "artifacts/alzheimer/predictions_contrast_cont4"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sha256(CHECKPOINT) != HASH:
            raise ValueError("Checkpoint mismatch")
        clinical = pd.read_csv(ROOT / "data/oasis1/oasis_cross-sectional.csv").set_index("ID")
        ready = []
        for status in sorted((ROOT / "artifacts/alzheimer/prepared").glob("*/status.json")):
            if json.loads(status.read_text()).get("state") == "prepared":
                ready.append(status.parent)
        write_json(output / "input_snapshot.json", {"subjects": [p.name for p in ready],
                   "n": len(ready), "checkpoint_sha256": HASH,
                   "visual_qc": "human_review_required", "pretraining_overlap": "not_verified",
                   "selection_warning": "Best examples are cherry-picked by chronological error, not general performance or verified biological age"})
        if not ready:
            raise ValueError("No fully prepared MRI available")
        model = SFCN().eval()
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)
        predictions, failures = [], []
        for folder in ready:
            try:
                person = clinical.loc[folder.name]
                array_path = folder / "sfcn.npy"
                array = normalize_sfcn_array(np.load(array_path, allow_pickle=False))
                with torch.inference_mode():
                    age = float(expected_age(model(torch.from_numpy(array[None,None].copy())))[0])
                cdr = float(person.CDR) if pd.notna(person.CDR) else None
                group = "control" if cdr == 0 else "dementia_AD_cohort" if cdr in (.5,1,2) else "CDR_unknown"
                predictions.append({"subject_id": folder.name, "age": float(person.Age),
                                    "cdr": cdr, "group": group, "prediction": age,
                                    "brain_age_gap": age - float(person.Age),
                                    "prepared_sha256": sha256(array_path)})
                rank_predictions(pd.DataFrame(predictions)).to_csv(output / "all_predictions.csv", index=False)
                print(f"PREDICTED {folder.name}: actual={person.Age} predicted={age:.2f} group={group}", flush=True)
            except Exception as exc:
                failures.append({"subject_id": folder.name, "error": str(exc)})
            write_json(output / "failures.json", failures)
            write_json(output / "status.json", {"state": "running", "target": len(ready),
                       "predicted": len(predictions), "failed": len(failures)})
        if predictions:
            ranked = rank_predictions(pd.DataFrame(predictions))
            ranked.head(10).to_csv(output / "best_examples.csv", index=False)
            ranked[ranked.group == "dementia_AD_cohort"].head(10).to_csv(output / "best_dementia_examples.csv", index=False)
            report = {"research_only": True, "visual_qc": "human_review_required", "groups": {},
                      "warning": "Chronological error in dementia is not true biological brain-age accuracy; selected examples are not an unbiased performance estimate"}
            for group, frame in ranked.groupby("group"):
                report["groups"][group] = metric(frame.age.to_numpy(), frame.prediction.to_numpy()) if len(frame)>1 else {"n":len(frame)}
            write_json(output / "report.json", report)
        write_json(output / "status.json", {"state": "finished", "target": len(ready),
                   "predicted": len(predictions), "failed": len(failures)})


if __name__ == "__main__":
    main()
