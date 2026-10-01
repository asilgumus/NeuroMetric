"""Inspect native OAS30 geometry and rejected candidate segmentation."""
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
folder = root / "artifacts/alzheimer/segmentation_OAS1_0030"
mri = nib.load(folder / "resampled.nii.gz").get_fdata(dtype=np.float32)
seg = nib.load(folder / "segmentation.nii.gz").get_fdata(dtype=np.float32)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for axis in range(3):
    index = mri.shape[axis] // 2
    plane = np.rot90(np.take(mri, index, axis=axis))
    labels = np.rot90(np.take(seg, index, axis=axis))
    for row in range(2):
        axes[row, axis].imshow(plane, cmap="gray", vmin=0, vmax=np.percentile(mri, 99))
        axes[row, axis].axis("off")
        axes[row, axis].set_title(f"Array axis {axis}, slice {index}" + (" · rejected labels" if row else ""))
    axes[1, axis].imshow(np.ma.masked_where(labels == 0, labels), cmap="tab20", alpha=.5, interpolation="nearest")
fig.tight_layout()
fig.savefig(folder / "rejected_native_qc.png", dpi=140)
plt.close(fig)
