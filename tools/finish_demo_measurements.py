"""Export hash-bound, technically reviewed demo results, not clinical diagnoses."""
import argparse
import json
from pathlib import Path
import sys
import nibabel as nib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from tools.predict_oasis_prepared import CHECKPOINT, HASH
from tools.export_fastsurfer_candidate_regions import grouped_volumes
from tools.longitudinal_measures import summarize_visits
from v3_model import SFCN, expected_age
from v3_train import normalize_sfcn_array


def publish(path, variable, data):
    path.write_text("window." + variable + " = " + json.dumps(data).replace("<", "\\u003c") + ";\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longitudinal", action="store_true")
    parser.add_argument("--regional-reviewed", action="store_true", help="Use only after inspecting cortical_qc.png")
    args = parser.parse_args()
    if args.longitudinal:
        if sha256(CHECKPOINT) != HASH:
            raise ValueError("Checkpoint mismatch")
        torch.set_num_threads(2)
        model = SFCN().eval()
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state"])
        frame = pd.read_excel(ROOT / "data/oasis2/demographics.xlsx").set_index("MRI ID")
        visits = []
        for identifier in ("OAS2_0001_MR1", "OAS2_0001_MR2"):
            folder = ROOT / "artifacts/longitudinal" / identifier
            status = json.loads((folder / "status.json").read_text())
            array_path = folder / "sfcn.npy"
            if status.get("visual_qc") != "approved" or status.get("input_sha256") != sha256(array_path):
                raise ValueError("Current input has not passed technical visual QC")
            array = normalize_sfcn_array(np.load(array_path, allow_pickle=False))
            with torch.inference_mode():
                prediction = float(expected_age(model(torch.from_numpy(array[None, None].copy())))[0])
            person = frame.loc[identifier]
            visits.append({"patient_id": "OAS2_0001", "scan_id": identifier,
                           "age": float(person.Age), "prediction": prediction, "mr_delay_days": int(person["MR Delay"]),
                           "checkpoint_sha256": HASH, "preprocessing_signature": "official-nifti-synthstrip-fast-flirt12-mni1mm-sfcn160x192x160",
                           "visual_qc": "approved", "qc_scope": "technical_registration_only", "input_sha256": sha256(array_path)})
        result = summarize_visits(visits)
        write_json(ROOT / "artifacts/longitudinal/OAS2_0001_predictions.json", result)
        reference_path = ROOT / "artifacts/longitudinal/OAS2_0001_reference.json"
        reference = json.loads(reference_path.read_text())
        reference["ai_longitudinal"] = result
        for visit, prediction in zip(reference["visits"], visits):
            visit.update(model_prediction=prediction["prediction"], model_qc="technical_review_completed")
        write_json(reference_path, reference)
        publish(ROOT / "assets/oasis2-longitudinal-reference.js", "NEUROMETRIC_OASIS2_REFERENCE", reference)
        print(json.dumps(result, indent=2))
    if args.regional_reviewed:
        folder = ROOT / "artifacts/alzheimer/fastsurfer_verified_OAS1_0030"
        subject = folder / "OAS1_0030_verified/mri"
        native = ROOT / "artifacts/alzheimer/verified_geometry_OAS1_0030/native_asl.nii.gz"
        source, saved = nib.load(native), nib.load(subject / "orig/001.mgz")
        if not np.allclose(source.affine, saved.affine, atol=1e-4) or not np.array_equal(source.get_fdata(), saved.get_fdata()):
            raise ValueError("Segmentation input does not match corrected MRI")
        evidence = folder / "cortical_qc.png"
        if not evidence.is_file():
            raise ValueError("Visual review evidence required")
        mapping = ROOT / "data/dkt_lobe_groups.json"
        seg = subject / "aparc.DKTatlas+aseg.deep.mgz"
        rows = grouped_volumes(nib.load(subject / "orig.mgz"), nib.load(seg), json.loads(mapping.read_text()))
        if any(row["status"] != "measured" for row in rows):
            raise ValueError("Incomplete anatomical labels")
        result = {"subject_id": "OAS1_0030", "visual_qc": "approved", "qc_scope": "technical_segmentation_only",
                  "reviewer": "coding_assistant_visual_review", "clinical_validation": False,
                  "method": "FastSurfer v2.4.2 VINN bilateral DKT gray-matter parcel groups on conformed 1 mm grid",
                  "native_sha256": sha256(native), "segmentation_sha256": sha256(seg), "mapping_sha256": sha256(mapping),
                  "evidence_sha256": sha256(evidence), "regions": rows,
                  "limitations": "Technical review, not clinician validation. Gray-matter groups exclude white matter, cingulate and insula. Not regional ages or normative percentiles."}
        write_json(folder / "regional_measures.json", result)
        publish(ROOT / "assets/oasis30-regional.js", "NEUROMETRIC_REGIONAL", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
