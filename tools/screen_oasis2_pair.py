"""Measure chronological prediction error for both visits before choosing a demo.

Candidate selection is explicitly post-hoc, never a generalization metric.
Visual QC is required before publication; predictions remain diagnostic only.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from tools.predict_oasis_prepared import CHECKPOINT, HASH
from v3_model import SFCN, expected_age
from v3_train import normalize_sfcn_array
from tools.oasis_visit_selection import visit_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--wait-for-download", action="store_true")
    parser.add_argument("--maximum-error", type=float, default=3.0)
    args = parser.parse_args()
    if not re.fullmatch(r"OAS2_\d{4}", args.subject):
        raise ValueError("Expected subject identifier")
    selected = visit_ids(ROOT / "data/oasis2/demographics.xlsx", args.subject)
    if args.wait_for_download:
        deadline = time.monotonic() + 7200
        while True:
            ready = True
            for identifier in selected:
                files = list((ROOT / "data/oasis2/raw" / identifier).glob("*.img"))
                if len(files) != 1 or not files[0].with_suffix(".hdr").exists() or files[0].stat().st_size != 33554432:
                    ready = False
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Paired download incomplete after two hours")
            time.sleep(10)
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_oasis2_demo.py"), "--subject", args.subject], check=True, cwd=ROOT)
    if sha256(CHECKPOINT) != HASH:
        raise ValueError("Checkpoint mismatch")
    model = SFCN().eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state"])
    torch.set_num_threads(2)
    metadata = pd.read_excel(ROOT / "data/oasis2/demographics.xlsx").set_index("MRI ID")
    visits = []
    for identifier in selected:
        folder = ROOT / "artifacts/longitudinal" / identifier
        array = normalize_sfcn_array(np.load(folder / "sfcn.npy", allow_pickle=False))
        with torch.inference_mode():
            prediction = float(expected_age(model(torch.from_numpy(array[None, None].copy())))[0])
        age = float(metadata.loc[identifier].Age)
        visits.append({"scan_id": identifier, "age": age, "prediction": prediction,
                       "absolute_error": abs(prediction - age), "mr_delay_days": int(metadata.loc[identifier, "MR Delay"]),
                       "input_sha256": sha256(folder / "sfcn.npy"), "visual_qc": "pending"})
        print(f"{identifier}: age={age:.0f}, predicted={prediction:.2f}, error={abs(prediction-age):.2f}", flush=True)
    result = {"patient_id": args.subject, "checkpoint_sha256": HASH, "visits": visits,
              "candidate_passes_error_threshold": all(v["absolute_error"] <= args.maximum_error for v in visits),
              "threshold_years": args.maximum_error, "publication_approved": False,
              "selection_warning": "Post-hoc demo candidate screening by chronological error, not unbiased accuracy or biological-age validation."}
    write_json(ROOT / "artifacts/longitudinal" / (args.subject + "_screening.json"), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
