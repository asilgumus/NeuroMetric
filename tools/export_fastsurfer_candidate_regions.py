"""Compute experimental bilateral DKT gray-matter parcel group volumes.

Does not promote uncertain geometry to approved clinical measurements.
"""
import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from tools.regional_measures import measure_regions


def grouped_volumes(image, labels, mapping):
    if image.shape != labels.shape or not np.allclose(image.affine, labels.affine):
        raise ValueError("Conformed MRI and DKT segmentation grids differ")
    # MGH geometry has physical millimetre coordinates, but no NIfTI units field.
    reference = nib.Nifti1Image(np.asanyarray(image.dataobj), image.affine)
    reference.header.set_xyzt_units("mm")
    segmentation = nib.Nifti1Image(np.asanyarray(labels.dataobj), labels.affine)
    rows = measure_regions(reference, segmentation, mapping)
    found = set(np.unique(labels.get_fdata()).astype(int))
    for row in rows:
        missing = sorted(set(mapping[row["region"]]) - found)
        row["missing_labels"] = missing
        if missing:
            row["status"] = "incomplete_parcellation"
            row["volume_ml"] = None
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Publish explicitly experimental results after visual inspection")
    args = parser.parse_args()
    folder = ROOT / "artifacts/alzheimer/fastsurfer_OAS1_0030"
    mri = folder / "OAS1_0030_candidate/mri/orig.mgz"
    seg = folder / "OAS1_0030_candidate/mri/aparc.DKTatlas+aseg.deep.mgz"
    geometry = ROOT / "artifacts/alzheimer/geometry_candidate_OAS1_0030"
    provenance = json.loads((geometry / "provenance.json").read_text())
    native = ROOT / "artifacts/alzheimer/prepared/OAS1_0030_MR1/raw.nii.gz"
    candidate = geometry / "experimental_asr.nii.gz"
    # Verify that FastSurfer's saved original is the actual geometry candidate.
    input_copy = nib.load(folder / "OAS1_0030_candidate/mri/orig/001.mgz")
    input_image = nib.load(candidate)
    if not np.allclose(input_copy.affine, input_image.affine, atol=1e-4) or not np.array_equal(input_copy.get_fdata(), input_image.get_fdata()):
        raise ValueError("FastSurfer original does not match the recorded candidate")
    if provenance["source_sha256"] != sha256(native) or provenance["candidate_sha256"] != sha256(candidate) or not provenance["units_verified_from_official_metadata"]:
        raise ValueError("Source or physical-unit provenance mismatch")
    mapping_file = ROOT / "data/dkt_lobe_groups.json"
    rows = grouped_volumes(nib.load(mri), nib.load(seg), json.loads(mapping_file.read_text()))
    if any(row["status"] != "measured" for row in rows):
        raise ValueError("One or more anatomical groups have incomplete labels; publication withheld")
    checkpoints = ROOT / ".cache/FastSurfer/checkpoints"
    result = {"subject_id": "OAS1_0030", "status": "experimental_geometry_review_pending", "approved": False,
        "native_source_sha256": sha256(native), "candidate_sha256": sha256(candidate), "segmentation_sha256": sha256(seg),
        "mapping_sha256": sha256(mapping_file), "regions": rows, "cortical_parcellation_available": True,
        "method": "FastSurfer v2.4.2 VINN DKT hard-label bilateral gray-matter parcel groups",
        "checkpoint_sha256": {p.name: sha256(p) for p in checkpoints.glob("aparc_vinn_*.pkl")},
        "limitations": "Experimental orientation; LR and full segmentation review pending. Gray-matter parcel sums exclude white matter, cingulate and insula. Not regional ages, normative percentiles or diagnosis."}
    write_json(folder / "candidate_regions.json", result)
    if args.publish:
        payload = json.dumps(result, ensure_ascii=True).replace("<", "\\u003c")
        (ROOT / "assets/oasis30-regional-candidate.js").write_text("window.NEUROMETRIC_REGIONAL_CANDIDATE = " + payload + ";\n")


if __name__ == "__main__":
    main()
