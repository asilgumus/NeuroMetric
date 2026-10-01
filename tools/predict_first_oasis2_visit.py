"""Predict the first complete arriving visit; continue only if error <=2 years."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import nibabel as nib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from tools.predict_oasis_prepared import CHECKPOINT, HASH
from v3_model import SFCN, expected_age
from v3_train import normalize_sfcn_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--continue-if-good", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"OAS2_\d{4}", args.subject):
        raise ValueError("Expected subject identifier")
    status = ROOT / "artifacts/longitudinal" / (args.subject + "_first_prediction.json")
    write_json(status, {"state": "waiting_for_first_mri", "patient_id": args.subject})
    deadline = time.monotonic() + 3600
    source = None
    while source is None:
        for path in sorted((ROOT / "data/oasis2/raw").glob(args.subject + "_MR[12]/*.img")):
            if not path.with_suffix(".hdr").exists():
                continue
            image = nib.load(path)
            if path.stat().st_size == int(np.prod(image.shape)) * image.get_data_dtype().itemsize:
                source = path
                break
        if source is not None:
            break
        if time.monotonic() > deadline:
            write_json(status, {"state": "download_timeout", "patient_id": args.subject})
            raise TimeoutError("No complete MRI arrived in one hour")
        time.sleep(5)
    visit = source.parent.name.rsplit("_", 1)[1]
    write_json(status, {"state": "preparing_first_mri", "scan_id": source.parent.name})
    subprocess.run([sys.executable, "tools/prepare_oasis2_demo.py", "--subject", args.subject, "--visit", visit], cwd=ROOT, check=True)
    if sha256(CHECKPOINT) != HASH:
        raise ValueError("Wrong checkpoint")
    torch.set_num_threads(2)
    model = SFCN().eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state"])
    array_path = ROOT / "artifacts/longitudinal" / source.parent.name / "sfcn.npy"
    array = normalize_sfcn_array(np.load(array_path, allow_pickle=False))
    with torch.inference_mode():
        prediction = float(expected_age(model(torch.from_numpy(array[None, None].copy())))[0])
    metadata = pd.read_excel(ROOT / "data/oasis2/demographics.xlsx").set_index("MRI ID")
    age = float(metadata.loc[source.parent.name].Age)
    error = abs(prediction - age)
    result = {"state": "predicted", "scan_id": source.parent.name, "age": age, "prediction": prediction,
              "absolute_error": error, "passes": error <= 2, "criterion": "absolute chronological error <=2 years",
              "model": "SFCN contrast-cont4 best.pt", "checkpoint_sha256": HASH,
              "input_sha256": sha256(array_path), "visual_qc": "pending", "publication_approved": False}
    write_json(status, result)
    print(json.dumps(result, indent=2), flush=True)
    if args.continue_if_good and result["passes"]:
        subprocess.run([sys.executable, "tools/screen_oasis2_pair.py", "--subject", args.subject,
                        "--wait-for-download", "--maximum-error", "2"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
