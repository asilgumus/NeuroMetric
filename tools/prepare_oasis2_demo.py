"""Prepare genuine repeated OASIS-2 NIfTI acquisitions; no invented geometry."""
import os
import argparse
import re
import fcntl
from pathlib import Path
import sys
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from v3_prepare import sfcn_array, save_qc
from tools.oasis_visit_selection import visit_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="OAS2_0001")
    parser.add_argument("--visit", help="Prepare only this recorded visit, e.g. MR3")
    args = parser.parse_args()
    if not re.fullmatch(r"OAS2_\d{4}", args.subject):
        raise ValueError("Expected OAS2 subject identifier")
    selected = visit_ids(ROOT / "data/oasis2/demographics.xlsx", args.subject)
    if args.visit and args.subject + "_" + args.visit not in selected:
        raise ValueError("Visit is not one of this patient's recorded selected visits")
    os.environ["FSLDIR"] = str(ROOT / ".cache/brainage/fsl")
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    lock_folder = ROOT / "artifacts/longitudinal"
    lock_folder.mkdir(parents=True, exist_ok=True)
    lock = (lock_folder / (args.subject + ".prepare.lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    for source in sorted((ROOT / "data/oasis2/raw").glob(args.subject + "_MR*/*.img")):
        if source.parent.name not in selected:
            continue
        if args.visit and source.parent.name != args.subject + "_" + args.visit:
            continue
        image = nib.load(source)
        expected = int(np.prod(image.shape)) * image.get_data_dtype().itemsize
        if source.stat().st_size < expected:
            continue
        folder = ROOT / "artifacts/longitudinal" / source.parent.name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "sfcn.npy").exists():
            continue
        write_json(folder / "status.json", {"state": "preparing", "visual_qc": "pending"})
        print(f"Preparing {source.parent.name}", flush=True)
        try:
            image = nib.load(source)
            if not isinstance(image, nib.Nifti1Pair) or image.header.get_xyzt_units()[0] != "mm" or not (image.header["sform_code"] or image.header["qform_code"]):
                raise ValueError("Expected official NIfTI pair with explicit millimetre geometry")
            array = image.get_fdata(dtype=np.float32)
            if array.ndim == 4 and array.shape[-1] == 1:
                array = array[..., 0]
            if array.ndim != 3:
                raise ValueError("Single three-dimensional acquisition required")
            native = folder / "native.nii.gz"
            converted = nib.Nifti1Image(array, image.affine)
            converted.header.set_xyzt_units("mm")
            nib.save(converted, native)
            write_json(folder / "provenance.json", {"source_sha256": sha256(source), "header_sha256": sha256(source.with_suffix(".hdr")),
                       "native_sha256": sha256(native), "axis_codes": list(nib.aff2axcodes(image.affine)),
                       "geometry_basis": "Official OASIS-2 coded NIfTI affine, unchanged", "voxels_modified": False})
            prepared = sfcn_array(native, folder / "registration", ROOT / ".cache/brainage/fsl")
            np.save(folder / "sfcn.npy", prepared, allow_pickle=False)
            np.savez_compressed(folder / "sfcn.npz", x=prepared)
            save_qc(prepared, folder / "qc.png", source.parent.name + " | registration review pending")
            write_json(folder / "status.json", {"state": "prepared", "visual_qc": "pending"})
        except Exception as error:
            write_json(folder / "status.json", {"state": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
