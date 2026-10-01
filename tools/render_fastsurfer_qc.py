"""Scientific overlay QC for all experimental DKT lobe groups."""
from pathlib import Path
import argparse
import json
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder", type=Path, default=ROOT / "artifacts/alzheimer/fastsurfer_OAS1_0030")
parser.add_argument("--subject", default="OAS1_0030_candidate")
parser.add_argument("--asset-output", type=Path)
args = parser.parse_args()
folder = args.folder
image = nib.load(folder / args.subject / "mri/orig.mgz")
labels = nib.load(folder / args.subject / "mri/aparc.DKTatlas+aseg.deep.mgz")
if image.shape != labels.shape or not np.allclose(image.affine, labels.affine):
    raise ValueError("MRI and labels do not share a grid")
data, seg = image.get_fdata(dtype=np.float32), labels.get_fdata(dtype=np.float32)
mapping = json.loads((ROOT / "data/dkt_lobe_groups.json").read_text())
group_map = np.zeros(seg.shape, dtype=np.uint8)
for i, ids in enumerate(mapping.values(), 1):
    group_map[np.isin(seg, ids)] = i
cmap = ListedColormap(["#ffd54f", "#ff8a65", "#81c784", "#64b5f6", "#ba68c8"])
fig, axes = plt.subplots(3, 5, figsize=(16, 10))
for axis in range(3):
    coords = np.argwhere(group_map > 0)[:, axis]
    indices = np.rint(np.quantile(coords, [.15, .325, .5, .675, .85])).astype(int)
    for column, index in enumerate(indices):
        plane = np.rot90(np.take(data, index, axis=axis))
        overlay = np.rot90(np.take(group_map, index, axis=axis))
        ax = axes[axis, column]
        ax.imshow(plane, cmap="gray", vmin=0, vmax=255)
        ax.imshow(np.ma.masked_where(overlay == 0, overlay), cmap=cmap, vmin=1, vmax=5, interpolation="nearest", alpha=.55)
        ax.set_title(f"Axis {axis}, slice {index}", fontsize=10)
        ax.axis("off")
fig.suptitle(args.subject + " · yellow hippocampus / orange temporal / green frontal / blue parietal / purple occipital", fontsize=11)
fig.tight_layout()
fig.savefig(folder / "cortical_qc.png", dpi=130)
if args.asset_output:
    args.asset_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.asset_output, dpi=130)
plt.close(fig)
