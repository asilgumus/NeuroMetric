"""Signed 3D Grad-CAM for one real MRI and checksum-locked SFCN checkpoint."""
import json
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from v3_model import SFCN, expected_age
from v3_train import normalize_sfcn_array


def signed_cam(activations, gradients, shape):
    if activations.shape != gradients.shape or activations.ndim != 5:
        raise ValueError("Expected matching 3D activations and gradients")
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    coarse = (weights * activations).sum(dim=1, keepdim=True)
    if not torch.isfinite(coarse).all() or coarse.abs().max() == 0:
        raise ValueError("Empty/non-finite attribution")
    return F.interpolate(coarse, size=shape, mode="trilinear", align_corners=False)[0, 0]


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", type=Path, default=ROOT / "artifacts/alzheimer/prepared/OAS1_0030_MR1")
    parser.add_argument("--output-folder", type=Path, default=ROOT / "artifacts/alzheimer/gradcam_OAS1_0030")
    parser.add_argument("--image-output", type=Path, default=ROOT / "assets/oasis30-gradcam.png")
    parser.add_argument("--subject-id", default="OAS1_0030_MR1")
    parser.add_argument("--age", type=float, default=65)
    args = parser.parse_args()
    folder = args.input_folder
    status = json.loads((folder / "status.json").read_text())
    if status["state"] != "prepared":
        raise ValueError("Preparation/QC not complete")
    checkpoint = ROOT / "artifacts/v3/autopilot/brainage-v3-train-sfcn-contrast-cont4/outputs/training/best.pt"
    digest = "6646617b6ac64877b03fbdc94f42395ef22974240237121e0d9dd9e3d53abc40"
    if sha256(checkpoint) != digest:
        raise ValueError("Checkpoint mismatch")
    array = normalize_sfcn_array(np.load(folder / "sfcn.npy", allow_pickle=False))
    model = SFCN().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    saved = {}
    def capture(module, inputs, output):
        # Detach the feature tensor: only classifier-to-feature gradients needed.
        saved["activation"] = output.detach().requires_grad_(True)
        return saved["activation"]
    handle = model.feature_extractor.conv_5.register_forward_hook(capture)
    try:
        prediction = expected_age(model(torch.from_numpy(array[None, None].copy())))[0]
        gradient = torch.autograd.grad(prediction, saved["activation"])[0]
        heat = signed_cam(saved["activation"], gradient, array.shape).detach().numpy()
        prediction = prediction.detach()
    finally:
        handle.remove()
    mask = array > 0
    heat = np.where(mask, heat, 0)
    scale = float(np.quantile(np.abs(heat[mask]), .99))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("No usable brain attribution")
    heat = np.clip(heat / scale, -1, 1)
    destination = args.output_folder
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "signed_gradcam.npy", heat, allow_pickle=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slices = [("Axial", 2, 80), ("Coronal", 1, 96), ("Sagittal", 0, 80)]
    fig, axes = plt.subplots(3, 1, figsize=(5, 11), facecolor="#f8fafc")
    vmax = np.percentile(array[mask], 99)
    for ax, (name, axis, index) in zip(axes, slices):
        brain = np.rot90(np.take(array, index, axis=axis))
        cam = np.rot90(np.take(heat, index, axis=axis))
        ax.imshow(brain, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        overlay = ax.imshow(cam, cmap="coolwarm", vmin=-1, vmax=1,
                            alpha=np.where(brain > 0, .55 * np.abs(cam), 0))
        ax.set_title(name + " · MNI array slice " + str(index), fontsize=10)
        ax.axis("off")
    fig.suptitle(args.subject_id + " · SFCN contrast-cont4\nReal age " + str(args.age) + " · predicted " + f"{float(prediction):.1f}" + "\nSigned Grad-CAM · coarse 5 × 6 × 5 features", fontsize=11)
    fig.subplots_adjust(top=.89, bottom=.08, hspace=.18)
    colorbar = fig.colorbar(overlay, ax=axes.tolist(), orientation="horizontal", fraction=.025, pad=.025)
    colorbar.set_label("Negative ← normalized attribution → Positive\nNot a disease map or regional age", fontsize=9)
    path = args.image_output
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    write_json(destination / "metadata.json", {
        "subject_id": args.subject_id, "chronological_age": args.age,
        "prediction": float(prediction), "brain_age_gap": float(prediction)-args.age,
        "checkpoint_sha256": digest, "input_sha256": sha256(folder / "sfcn.npy"),
        "layer": "feature_extractor.conv_5", "coarse_shape": list(saved["activation"].shape[2:]),
        "target": "expected_age", "method": "signed Grad-CAM, no ReLU",
        "normalization": "brain-masked 99th percentile absolute attribution",
        "image": str(path), "visual_qc": status.get("visual_qc", "pending"),
        "limitations": "Coarse post-hoc attribution, not attention weights, causal proof, anatomical segmentation, Alzheimer risk or regional age"})
    print(f"GRADCAM_READY prediction={float(prediction):.6f} image={path}", flush=True)


if __name__ == "__main__":
    main()
