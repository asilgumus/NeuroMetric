"""Native-space, segmentation-based volumes; never regional brain-age estimates.

Label image must be aligned to the native MRI, not simply a template atlas.
JSON label mapping: {"Hippocampus": [17, 53], ...}.
Run with --qc-approved only after reviewing segmentation alignment.
"""
import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainage import sha256, write_json


def measure_regions(mri, segmentation, mapping):
    if len(mri.shape) != 3 or mri.shape != segmentation.shape:
        raise ValueError("Native MRI and segmentation must share a 3D grid")
    if not np.allclose(mri.affine, segmentation.affine, atol=1e-5):
        raise ValueError("Segmentation affine does not match native MRI")
    if tuple(mri.header.get_xyzt_units())[0] != "mm":
        raise ValueError("MRI spatial units must explicitly be millimetres")
    labels = segmentation.get_fdata(dtype=np.float32)
    if not np.isfinite(labels).all() or not np.equal(labels, np.rint(labels)).all():
        raise ValueError("Expected finite integer segmentation labels")
    voxel_mm3 = abs(float(np.linalg.det(mri.affine[:3, :3])))
    if not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError("Invalid voxel volume")
    if not mapping:
        raise ValueError("Region mapping is empty")
    used = set()
    result = []
    for name, identifiers in mapping.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(identifiers, list) or not identifiers:
            raise ValueError("Each named region requires a nonempty label list")
        if any(type(i) is not int or i <= 0 for i in identifiers):
            raise ValueError("Positive integer labels required; background excluded")
        if len(set(identifiers)) != len(identifiers) or used.intersection(identifiers):
            raise ValueError("Region labels must be unique and nonoverlapping")
        used.update(identifiers)
        count = int(np.isin(labels, identifiers).sum())
        result.append({"region": name, "voxel_count": count,
                       "volume_ml": count * voxel_mm3 / 1000 if count else None,
                       "status": "measured" if count else "label_missing",
                       "regional_brain_age": None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qc-approved", action="store_true")
    args = parser.parse_args()
    if not args.qc_approved:
        parser.error("Review native-space segmentation alignment before --qc-approved")
    rows = measure_regions(nib.load(args.mri), nib.load(args.segmentation), json.loads(args.labels.read_text()))
    write_json(args.output, {"method": "native-space segmentation voxel volume",
        "mri_sha256": sha256(args.mri), "segmentation_sha256": sha256(args.segmentation),
        "mapping_sha256": sha256(args.labels), "visual_qc": "user_approved",
        "regions": rows, "limitations": "Volumes are not ages, age percentiles, or a diagnosis. Atlas template volumes are not individual segmentation."})


if __name__ == "__main__":
    main()
