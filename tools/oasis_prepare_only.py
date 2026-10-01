"""Prepare downloaded OASIS raw MRI; never load an age model or predict."""
import json
import os
from pathlib import Path
import sys
import time
import fcntl

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import check_volume, write_json
from oasis_test import image_index
from v3_prepare import sfcn_array, save_qc


def main():
    os.environ["FSLDIR"] = str(ROOT / ".cache/brainage/fsl")
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    output = ROOT / "artifacts/alzheimer/prepared"
    output.mkdir(parents=True, exist_ok=True)
    # One baseline acquisition per person; exclude repeat visits and unknown IDs.
    clinical = pd.read_csv(ROOT / "data/oasis1/oasis_cross-sectional.csv")
    target_ids = set(clinical.loc[clinical.ID.str.fullmatch(r"OAS1_\d{4}_MR1", na=False), "ID"])
    with (output / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            images = image_index(ROOT / "data/oasis1/raw")
            for subject, source in images.items():
                if subject not in target_ids:
                    continue
                if source.suffix != ".img" or not source.with_suffix(".hdr").exists():
                    continue
                # Extraction may expose a filename before all voxels are written.
                try:
                    with source.with_suffix(".hdr").open("rb") as header_file:
                        header = nib.AnalyzeHeader.from_fileobj(header_file)
                    expected_bytes = int(np.prod(header.get_data_shape())) * header.get_data_dtype().itemsize
                    if source.stat().st_size < expected_bytes:
                        continue
                except (OSError, ValueError):
                    continue
                folder = output / subject
                status = folder / "status.json"
                if status.exists():
                    previous = json.loads(status.read_text())
                    if previous.get("state") == "prepared":
                        continue
                    if previous.get("state") == "failed" and "SIGABRT" not in previous.get("error", ""):
                        continue
                folder.mkdir(exist_ok=True)
                write_json(status, {"state": "running", "source": str(source)})
                print(f"PREPARE {subject}", flush=True)
                try:
                    volume = nib.load(str(source))
                    data = volume.get_fdata(dtype=np.float32)
                    if data.ndim == 4 and data.shape[-1] == 1:
                        data = data[..., 0]
                    if data.ndim != 3:
                        raise ValueError("Expected a single 3D MRI")
                    raw = folder / "raw.nii.gz"
                    nib.save(nib.Nifti1Image(data, volume.affine), raw)
                    check_volume(raw)
                    array = sfcn_array(raw, folder / "registration", ROOT / ".cache/brainage/fsl")
                    np.save(folder / "sfcn.npy", array, allow_pickle=False)
                    save_qc(array, folder / "qc.png", subject + " | registration QC")
                    write_json(status, {"state": "prepared", "source": str(source),
                               "visual_qc": "human_review_required", "prediction_performed": False})
                    print(f"PREPARED {subject}", flush=True)
                except Exception as exc:
                    write_json(status, {"state": "failed", "error": str(exc), "prediction_performed": False})
                    print(f"FAILED {subject}: {exc}", flush=True)
            states = {p.parent.name: json.loads(p.read_text()).get("state")
                      for p in output.glob("*/status.json")}
            write_json(output / "worker_status.json", {
                "target_people": len(target_ids), "prepared": sum(v == "prepared" for v in states.values()),
                "failed": sum(v == "failed" for v in states.values()),
                "prediction_enabled": False})
            if all(states.get(sid) in ("prepared", "failed") for sid in target_ids):
                print("All baseline participants processed; review failures and visual QC", flush=True)
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
