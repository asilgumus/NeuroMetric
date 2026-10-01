"""Export bilateral experimental measurements, explicitly NOT approved volumes."""
from pathlib import Path
import json
import sys
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from tools.regional_measures import measure_regions

def main():
    folder = ROOT / "artifacts/alzheimer/segmentation_geometry_candidate_OAS1_0030"
    geometry = ROOT / "artifacts/alzheimer/geometry_candidate_OAS1_0030"
    status = json.loads((folder / "status.json").read_text())
    provenance = json.loads((geometry / "provenance.json").read_text())
    source = ROOT / "artifacts/alzheimer/prepared/OAS1_0030_MR1/raw.nii.gz"
    if status.get("state") != "candidate_complete" or status.get("experimental_geometry") is not True:
        raise ValueError("Completed experimental segmentation required")
    if provenance.get("source_sha256") != sha256(source) or not provenance.get("units_verified_from_official_metadata"):
        raise ValueError("Native source or physical units are unverified")
    if status["input_sha256"] != provenance["candidate_sha256"] or status["input_sha256"] != sha256(geometry / "experimental_asr.nii.gz"):
        raise ValueError("Geometry candidate changed")
    segmentation = folder / "segmentation.nii.gz"
    if sha256(segmentation) != status["segmentation_sha256"]:
        raise ValueError("Segmentation changed")
    # FreeSurfer/SynthSeg label dictionary. Use bilateral sums because LR is unresolved.
    mapping = {"Hippocampus (bilateral)": [17, 53], "Amygdala (bilateral)": [18, 54],
               "Thalamus (bilateral)": [10, 49], "Cerebral cortex (bilateral)": [3, 42]}
    rows = measure_regions(nib.load(folder / "resampled.nii.gz"), nib.load(segmentation), mapping)
    result = {"subject_id": "OAS1_0030", "status": "experimental_geometry_review_pending",
              "approved": False, "native_source_sha256": sha256(source),
              "segmentation_sha256": sha256(segmentation), "regions": rows,
              "method": "SynthSeg 1.0 hard-label volume on the subject's 1mm resampled MRI",
              "limitations": "Bilateral candidate volumes; complete orientation and segmentation review pending. Not regional ages, normative percentiles, diagnosis, or approved clinical measurements. Cortical lobes have not been parcellated."}
    write_json(folder / "candidate_regions.json", result)
    payload = json.dumps(result, ensure_ascii=True).replace("<", "\\u003c")
    (ROOT / "assets/oasis30-regional-candidate.js").write_text("window.NEUROMETRIC_REGIONAL_CANDIDATE = " + payload + ";\n")

if __name__ == "__main__":
    main()
