"""Run pinned SynthSeg 1.0 on the unchanged OAS30 native MRI, then render QC.

Produces candidate segmentation only. Does not approve QC or infer regional ages.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-dependencies", action="store_true")
    parser.add_argument("--candidate-input", type=Path, help="Experimental geometry candidate; never overwrites native input")
    parser.add_argument("--output-folder", type=Path)
    args = parser.parse_args()
    output = args.output_folder or ROOT / "artifacts/alzheimer/segmentation_OAS1_0030"
    if args.candidate_input and not args.output_folder:
        parser.error("Geometry candidates require a separate --output-folder")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = output / "status.json"
    native = args.candidate_input or ROOT / "artifacts/alzheimer/prepared/OAS1_0030_MR1/raw.nii.gz"
    repo = ROOT / ".cache/SynthSeg"
    python = ROOT / ".cache/synthseg-env/bin/python"
    model = repo / "models/synthseg_1.0.h5"
    try:
        if sha256(model) != "7d2e32d298fe38dc51ea38e6a8e8fc5c665d56b63cddb409e8b199535f8b5298":
            raise ValueError("SynthSeg weights differ from pinned model")
        if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != "2a2aa3bbfccb83f8253a51ca8b329b9938a2646d":
            raise ValueError("SynthSeg code differs from pinned commit")
        dependency = ROOT / ".cache/synthseg-env/lib/python3.8/site-packages/tensorflow/__init__.py"
        deadline = time.monotonic() + (900 if args.wait_for_dependencies else 0)
        while not dependency.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("SynthSeg dependencies not installed; install requirements first")
            write_json(status, {"state": "waiting_for_dependencies", "pid": os.getpid()})
            time.sleep(5)
        write_json(status, {"state": "running", "pid": os.getpid(), "input_sha256": sha256(native)})
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "TF_CPP_MIN_LOG_LEVEL": "2"}
        command = [str(python), str(repo / "scripts/commands/SynthSeg_predict.py"), "--i", str(native),
                   "--o", str(output / "segmentation.nii.gz"), "--resample", str(output / "resampled.nii.gz"),
                   "--vol", str(output / "candidate_volumes.csv"), "--v1", "--cpu", "--threads", "2"]
        with (output / "run.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # SynthSeg may log per-image errors without a nonzero exit code.
        import nibabel as nib
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        image = nib.load(output / "resampled.nii.gz")
        labels = nib.load(output / "segmentation.nii.gz")
        if image.shape != labels.shape or not np.allclose(image.affine, labels.affine):
            raise ValueError("Resampled MRI and segmentation grids do not match")
        data, seg = image.get_fdata(dtype=np.float32), labels.get_fdata(dtype=np.float32)
        if not np.isfinite(seg).all() or not np.any(np.isin(seg, [17, 53])):
            raise ValueError("Missing hippocampus labels or invalid segmentation")
        hip = np.isin(seg, [17, 53])
        center = np.rint(np.argwhere(hip).mean(axis=0)).astype(int)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, ax in enumerate(axes):
            base = np.rot90(np.take(data, center[axis], axis=axis))
            mask = np.rot90(np.take(hip, center[axis], axis=axis))
            ax.imshow(base, cmap="gray")
            ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", vmin=0, vmax=1, alpha=.65)
            ax.set_title("Hippocampus candidate · axis " + str(axis))
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output / "hippocampus_qc.png", dpi=140)
        plt.close(fig)
        write_json(status, {"state": "candidate_complete", "input_sha256": sha256(native),
            "segmentation_sha256": sha256(output / "segmentation.nii.gz"),
            "visual_qc": "human_review_required", "regional_age_available": False,
            "experimental_geometry": bool(args.candidate_input),
            "limitations": "Candidate SynthSeg 1.0 segmentation. Cortical lobes not parcellated. Physical units and QC must be reviewed before dashboard volume export."})
    except Exception as exc:
        write_json(status, {"state": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
