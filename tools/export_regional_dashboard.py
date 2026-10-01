"""Publish reviewed native-space OAS30 volumes as static dashboard data."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256


def validate_report(report, expected_mri_sha256):
    if report.get("visual_qc") != "user_approved":
        raise ValueError("Reviewed segmentation is required")
    if report.get("mri_sha256") != expected_mri_sha256:
        raise ValueError("Volume report does not match the selected patient's MRI")
    rows = report.get("regions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("No regional measurements")
    for row in rows:
        if not isinstance(row.get("region"), str) or not row["region"].strip():
            raise ValueError("Region name is missing")
        if row.get("regional_brain_age") is not None:
            raise ValueError("This exporter does not support regional age estimates")
        value = row.get("volume_ml")
        if row.get("status") == "measured":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Measured volumes must be finite and positive")
        elif row.get("status") != "label_missing" or value is not None:
            raise ValueError("Unsupported measurement status")
    return {"subject_id": "OAS1_0030", "visual_qc": "approved",
            "mri_sha256": expected_mri_sha256, "regions": rows,
            "method": report.get("method"), "limitations": report.get("limitations")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/oasis30-regional.js")
    args = parser.parse_args()
    native = ROOT / "artifacts/alzheimer/prepared/OAS1_0030_MR1/raw.nii.gz"
    result = validate_report(json.loads(args.input.read_text()), sha256(native))
    payload = json.dumps(result, ensure_ascii=True).replace("<", "\\u003c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("window.NEUROMETRIC_REGIONAL = " + payload + ";\n")


if __name__ == "__main__":
    main()
