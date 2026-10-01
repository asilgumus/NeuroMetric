"""Render label-blind SFCN/template overlays for a pending visual QC review."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("array", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.array, allow_pickle=False) as saved:
        array = saved["x"].astype(np.float32)
    template = nib.load(str(args.template)).get_fdata(dtype=np.float32)
    if array.shape != (160, 192, 160) or not np.isfinite(array).all():
        raise ValueError("Expected a finite SFCN input")
    starts = [(a - b) // 2 for a, b in zip(template.shape, array.shape)]
    if any(start < 0 for start in starts):
        raise ValueError("Template smaller than the SFCN input")
    template = template[tuple(slice(start, start + size)
                              for start, size in zip(starts, array.shape))]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    # Compare three offsets per axis, not only the central slice.
    for axis in range(3):
        for column, offset in enumerate((-12, 0, 12)):
            index = array.shape[axis] // 2 + offset
            scan = np.rot90(np.take(array, index, axis=axis))
            reference = np.rot90(np.take(template, index, axis=axis))
            ax = axes[axis, column]
            ax.imshow(scan, cmap="gray", vmin=0, vmax=float(np.percentile(array[array > 0], 99)))
            ax.contour(reference > 0, levels=[0.5], colors=["cyan"], linewidths=0.8)
            ax.set_title(f"axis {axis}, slice {index}")
            ax.axis("off")
    fig.suptitle(f"{args.array.stem}: cyan = MNI template boundary; pending reviewer decision")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
