"""Preserve OASIS sagittal Analyze voxels, assign documented ASL geometry.

Conversion convention: https://brainder.org/2011/08/13/converting-oasis-brains-to-nifti/
Cross-check: https://4dfp.readthedocs.io/en/latest/format.html
This does not claim patient-specific left/right marker or registration QC.
"""
import argparse
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from v3_prepare import sfcn_array, save_qc


def convert(source, metadata, target):
    document = ET.parse(metadata).getroot()
    ns = {"x": "http://nrg.wustl.edu/xnat"}
    scan = document.find("x:scans/x:scan[@ID='mpr-1']", ns)
    if scan is None:
        raise ValueError("Official mpr-1 scan metadata required")
    spacing = scan.find("x:parameters/x:voxelRes", ns)
    orientation = scan.find("x:parameters/x:orientation", ns)
    if spacing is None or spacing.attrib.get("units") != "mm" or orientation is None or orientation.text != "Sag":
        raise ValueError("Only documented millimetre sagittal acquisitions supported")
    image = nib.load(source)
    array = np.asanyarray(image.dataobj)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    orient = image.header["orient"].tobytes()[0]
    if array.ndim != 3 or orient != 2:
        raise ValueError("Expected sagittal-unflipped Analyze header and 3D data")
    zooms = np.array([float(spacing.attrib[k]) for k in ("x", "y", "z")])
    if not np.allclose(zooms, image.header.get_zooms()[:3]):
        raise ValueError("Header and official acquisition spacing disagree")
    affine = np.eye(4)
    affine[:3, :3] = [[0, 0, -zooms[2]], [zooms[0], 0, 0], [0, zooms[1], 0]]
    affine[:3, 3] = -affine[:3, :3] @ ((np.array(array.shape) - 1) / 2)
    converted = nib.Nifti1Image(array, affine)
    converted.header.set_xyzt_units("mm")
    converted.set_qform(affine, code=1)
    converted.set_sform(affine, code=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    nib.save(converted, target)
    return {"source_sha256": sha256(source), "header_sha256": sha256(source.with_suffix(".hdr")),
            "metadata_sha256": sha256(metadata), "converted_sha256": sha256(target),
            "axis_codes": list(nib.aff2axcodes(affine)), "voxels_modified": False,
            "geometry_basis": "Published OASIS Analyze sagittal ASL conversion convention",
            "references": ["https://brainder.org/2011/08/13/converting-oasis-brains-to-nifti/",
                           "https://4dfp.readthedocs.io/en/latest/format.html"],
            "patient_specific_lr_marker_review": "pending", "registration_visual_qc": "pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-sfcn", action="store_true")
    args = parser.parse_args()
    target = args.output / "native_asl.nii.gz"
    provenance = convert(args.source, args.metadata, target)
    write_json(args.output / "provenance.json", provenance)
    if args.prepare_sfcn:
        os.environ["FSLDIR"] = str(ROOT / ".cache/brainage/fsl")
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
        write_json(args.output / "status.json", {"state": "preparing", "visual_qc": "pending"})
        try:
            array = sfcn_array(target, args.output / "registration", ROOT / ".cache/brainage/fsl")
            np.save(args.output / "sfcn.npy", array, allow_pickle=False)
            np.savez_compressed(args.output / "sfcn.npz", x=array)
            save_qc(array, args.output / "qc.png", args.source.stem + " | corrected ASL, review required")
            write_json(args.output / "status.json", {"state": "prepared", "visual_qc": "pending"})
        except Exception as error:
            write_json(args.output / "status.json", {"state": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
