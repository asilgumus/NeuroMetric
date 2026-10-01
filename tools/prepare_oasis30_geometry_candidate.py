"""Experimental ASR geometry from observed sagittal acquisition, not approved orientation.

Native image axis 2 visibly contains sagittal slices, despite a generic Analyze
affine declaring it superior/inferior. Left/right direction is unresolved.
This candidate is for bilateral segmentation diagnostics, not production prediction.
"""
from pathlib import Path
import nibabel as nib
import numpy as np
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json

source = ROOT / "artifacts/alzheimer/prepared/OAS1_0030_MR1/raw.nii.gz"
output = ROOT / "artifacts/alzheimer/geometry_candidate_OAS1_0030"
output.mkdir(parents=True, exist_ok=True)
image = nib.load(source)
metadata = ROOT / "artifacts/alzheimer/oas30_reference/disc1/OAS1_0030_MR1/OAS1_0030_MR1.xml"
ns = {"xnat": "http://nrg.wustl.edu/xnat"}
document = ET.parse(metadata).getroot()
scan = document.find("xnat:scans/xnat:scan[@ID='mpr-1']", ns)
resolution = scan.find("xnat:parameters/xnat:voxelRes", ns)
orientation = scan.find("xnat:parameters/xnat:orientation", ns)
if document.attrib.get("ID") != "OAS1_0030_MR1" or resolution.attrib.get("units") != "mm" or orientation.text != "Sag":
    raise ValueError("Official subject metadata does not support this acquisition geometry")
if not np.allclose([float(resolution.attrib[k]) for k in ("x", "y", "z")], image.header.get_zooms()):
    raise ValueError("Acquisition spacing conflicts with native MRI")
if image.shape != (256, 256, 128) or not np.allclose(image.header.get_zooms(), (1, 1, 1.25)):
    raise ValueError("Candidate only applies to the inspected OAS30 acquisition")
affine = np.array([[0, 0, 1.25, -79.375], [1, 0, 0, -127.5], [0, 1, 0, -127.5], [0, 0, 0, 1]], dtype=float)
candidate = nib.Nifti1Image(np.asanyarray(image.dataobj), affine)
candidate.header.set_xyzt_units("mm")
target = output / "experimental_asr.nii.gz"
nib.save(candidate, target)
write_json(output / "provenance.json", {"source_sha256": sha256(source), "candidate_sha256": sha256(target), "metadata_sha256": sha256(metadata),
    "method": "Experimental header-only ASR assignment from visible plane anatomy and acquisition voxel spacing",
    "voxels_modified": False, "left_right_verified": False, "approved_for_prediction": False, "units_verified_from_official_metadata": True,
    "limitations": "Orientation hypothesis only. Sagittal acquisition and millimetre spacing verified from official subject metadata; full directional signs still require confirmation. No native source changed."})
