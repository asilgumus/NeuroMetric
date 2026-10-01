"""Segment original, hash-matched MRI for selected cases; never warped volumes.

Serial CPU execution limits RAM usage. Outputs remain pending visual QC.
"""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import nibabel as nib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from tools.export_fastsurfer_candidate_regions import grouped_volumes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["dlbs3898", "ixi361", "ixi170", "ixi566"])
    args = parser.parse_args()
    states = {}
    assets = ROOT / "assets/selected-regional-cases.js"
    if args.case and assets.exists():
        prefix = "window.NEUROMETRIC_SELECTED_REGIONS = "
        states = json.loads(assets.read_text().removeprefix(prefix).strip().removesuffix(";"))
    targets = [("dlbs3898", "DLBS:sub-3898", "dlbs"), ("ixi361", "IXI:IXI361", "ixi"),
               ("ixi170", "IXI:IXI170", "ixi"), ("ixi566", "IXI:IXI566", "ixi")]
    if args.case:
        targets = [target for target in targets if target[0] == args.case]
    def publish():
        assets.write_text("window.NEUROMETRIC_SELECTED_REGIONS = " + json.dumps(states).replace("<", "\\u003c") + ";\n")
        write_json(ROOT / "assets/selected-regional-cases.json", states)
    for key, subject, cohort in targets:
        states[subject] = {"patient_id": subject, "state": "queued", "regions": []}
    publish()
    env = dict(os.environ, PYTHONPATH=str(ROOT / ".cache/FastSurfer"), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2")
    for key, subject, cohort in targets:
        manifest = pd.read_csv(ROOT / ("artifacts/v3/qc_review_" + cohort + "_current/prepared/" + cohort + "/manifest.csv")).set_index("subject_id")
        row = manifest.loc[subject]
        if cohort == "dlbs":
            native = ROOT / "artifacts/demo_exact/dlbs/sub-3898-native.nii.gz"
        else:
            deadline = time.monotonic() + 1800
            while True:
                files = list((ROOT / "artifacts/demo_exact/native_downloads").rglob(row.source_name))
                if len(files) == 1 and sha256(files[0]) == row.source_sha256:
                    native = files[0]
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Original {subject} MRI not available")
                time.sleep(5)
        if sha256(native) != row.source_sha256:
            raise ValueError(f"Original MRI hash mismatch for {subject}")
        folder = ROOT / "artifacts/demo_exact/regional" / key
        folder.mkdir(parents=True, exist_ok=True)
        states[subject].update(state="segmenting", native_sha256=sha256(native))
        publish()
        print(f"SEGMENTING {subject}", flush=True)
        try:
            with (folder / "console.log").open("w") as log:
                subprocess.run([str(ROOT / ".cache/fastsurfer-env/bin/python"), str(ROOT / ".cache/FastSurfer/FastSurferCNN/run_prediction.py"),
                                "--t1", str(native.resolve()), "--sid", key, "--sd", str(folder), "--device", "cpu",
                                "--viewagg_device", "cpu", "--batch_size", "1", "--threads", "4", "--seg_log", str(folder / "run.log")],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            seg = folder / key / "mri/aparc.DKTatlas+aseg.deep.mgz"
            mri = folder / key / "mri/orig.mgz"
            rows = grouped_volumes(nib.load(mri), nib.load(seg), json.loads((ROOT / "data/dkt_lobe_groups.json").read_text()))
            if any(r["status"] != "measured" for r in rows):
                raise ValueError("Incomplete anatomical labels")
            image = ROOT / "assets" / (key + "-regional.png")
            subprocess.run([sys.executable, "tools/render_fastsurfer_qc.py", "--folder", str(folder), "--subject", key,
                            "--asset-output", str(image)], cwd=ROOT, check=True)
            states[subject].update(state="visual_review_pending", regions=rows, image=str(image.relative_to(ROOT)),
                                   segmentation_sha256=sha256(seg), clinical_validation=False,
                                   method="FastSurfer VINN bilateral DKT gray-matter groups; excludes white matter, cingulate and insula")
            write_json(folder / "regional_measures.json", states[subject])
            print(f"SEGMENTED {subject}; visual review pending", flush=True)
        except Exception as error:
            states[subject].update(state="failed", error=str(error))
            print(f"FAILED {subject}: {error}", flush=True)
        publish()


if __name__ == "__main__":
    main()
