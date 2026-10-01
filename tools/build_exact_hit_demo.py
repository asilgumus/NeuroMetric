"""Replay selected held-out predictions and produce their own signed Grad-CAM.

Reject mismatched input/model replays. Selected examples are not overall accuracy.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from tools.predict_oasis_prepared import CHECKPOINT, HASH
from tools.gradcam_oasis30 import signed_cam
from v3_model import SFCN, expected_age
from v3_train import normalize_sfcn_array


def main():
    if sha256(CHECKPOINT) != HASH:
        raise ValueError("Checkpoint mismatch")
    torch.set_num_threads(2)
    model = SFCN().eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cases = {}
    targets = [("ixi170", "IXI:IXI170", "ixi_test"), ("ixi361", "IXI:IXI361", "ixi_test"),
               ("ixi566", "IXI:IXI566", "ixi_test"), ("dlbs3898", "DLBS:sub-3898", "dlbs_external")]
    for key, subject, cohort in targets:
        frame = pd.read_csv(ROOT / "artifacts/v3/heldout_contrast_cont4/outputs/evaluation" / (cohort + "_predictions.csv"))
        row = frame.set_index("subject_id").loc[subject]
        paths = list((ROOT / "artifacts/demo_exact").rglob(Path(row.sfcn_array).name))
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one actual held-out MRI input for {subject}: {paths}")
        with np.load(paths[0], allow_pickle=False) as saved:
            brain = saved["x"].astype(np.float32)
        array = normalize_sfcn_array(brain)
        features = {}
        def capture(module, inputs, output):
            features["activation"] = output.detach().requires_grad_(True)
            return features["activation"]
        hook = model.feature_extractor.conv_5.register_forward_hook(capture)
        try:
            prediction = expected_age(model(torch.from_numpy(array[None, None].copy())))[0]
            gradient = torch.autograd.grad(prediction, features["activation"])[0]
            heat = signed_cam(features["activation"], gradient, array.shape).detach().numpy()
            predicted = float(prediction.detach())
        finally:
            hook.remove()
        # Held-out evaluation used CUDA autocast (FP16); attribution runs CPU
        # FP32. Require the same rounded result and <=0.1-year replay difference.
        if abs(predicted - float(row.prediction)) > .1 or np.rint(predicted) != np.rint(row.age):
            raise ValueError(f"Replay mismatch or no exact rounded match for {subject}: {predicted}")
        mask = brain > 0
        heat = np.where(mask, heat, 0)
        scale = np.quantile(np.abs(heat[mask]), .99)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid attribution")
        heat = np.clip(heat / scale, -1, 1)
        prefix = key + "-gradcam"
        for name, axis, index in [("axial", 2, 80), ("coronal", 1, 96), ("sagittal", 0, 80)]:
            plane, cam = np.rot90(np.take(brain, index, axis=axis)), np.rot90(np.take(heat, index, axis=axis))
            fig = plt.figure(figsize=(5, 5), facecolor="#080e17")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(plane, cmap="gray", vmin=0, vmax=np.percentile(brain[mask], 99))
            ax.imshow(cam, cmap="coolwarm", vmin=-1, vmax=1, alpha=np.where(plane > 0, .55 * np.abs(cam), 0))
            ax.axis("off")
            fig.savefig(ROOT / "assets" / (prefix + "-" + name + ".png"), dpi=140)
            plt.close(fig)
        age = float(row.age)
        gap = predicted - age
        gap_text = ("+" if gap >= 0 else "−") + f"{abs(gap):.1f}"
        cases[key] = {"real": True, "patientId": subject, "id": subject + " · selected held-out example",
                      "age": f"Age: {age:.1f}", "pred": f"{predicted:.1f}", "gap": gap_text,
                      "ci": "Individual uncertainty not calibrated", "pct": 0, "pctTxt": "—", "regions": [], "spots": [], "ring": None,
                      "heatPrefix": prefix, "cap": "Real signed Grad-CAM · coarse model attribution, not regional age or disease.",
                      "trend": [["Baseline only", gap_text]], "status": "Rounded chronological age matched · selected example",
                      "interp": f"Actual age {age:.2f}; prediction {predicted:.2f}. Both round to {int(np.rint(age))}. Selected post-hoc from held-out results; not an estimate of overall accuracy or evidence of brain health."}
        folder = ROOT / "artifacts/demo_exact" / key
        write_json(folder / "provenance.json", {"patient_id": subject, "age": age, "prediction": predicted,
                   "recorded_heldout_prediction": float(row.prediction), "input_sha256": sha256(paths[0]), "checkpoint_sha256": HASH,
                   "replay_precision": "CPU FP32; original held-out CUDA autocast", "replay_difference_years": predicted-float(row.prediction),
                   "cohort": cohort, "rounded_match": True, "selection": "post_hoc_demonstration", "clinical_validation": False})
        np.save(folder / "signed_gradcam.npy", heat, allow_pickle=False)
        print(f"READY {subject}: {age:.2f} -> {predicted:.2f}", flush=True)
    (ROOT / "assets/exact-hit-cases.js").write_text("window.NEUROMETRIC_EXACT_CASES = " + json.dumps(cases).replace("<", "\\u003c") + ";\n")


if __name__ == "__main__":
    main()
