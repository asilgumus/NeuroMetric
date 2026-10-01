"""Connect the actual paired predictions when ready; withhold unreviewed slopes."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import write_json
from tools.longitudinal_measures import summarize_visits
from tools.predict_oasis_prepared import HASH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    source = ROOT / "artifacts/longitudinal/OAS2_0007_screening.json"
    deadline = time.monotonic() + 10800
    while not source.exists():
        if not args.wait or time.monotonic() > deadline:
            raise TimeoutError("Paired predictions not available")
        time.sleep(10)
    result = json.loads(source.read_text())
    if result["checkpoint_sha256"] != HASH or result["patient_id"] != "OAS2_0007":
        raise ValueError("Wrong subject or model")
    visits = []
    for v in result["visits"]:
        status = json.loads((ROOT / "artifacts/longitudinal" / v["scan_id"] / "status.json").read_text())
        visits.append({**v, "patient_id": result["patient_id"], "checkpoint_sha256": HASH,
                       "preprocessing_signature": "official-nifti-synthstrip-fast-flirt12-mni1mm-sfcn160x192x160",
                       "visual_qc": status.get("visual_qc", "pending")})
    trajectory = summarize_visits(visits)
    write_json(ROOT / "artifacts/longitudinal/OAS2_0007_longitudinal.json", trajectory)
    payload = json.dumps(trajectory).replace("<", "\\u003c")
    (ROOT / "assets/oasis7-longitudinal.js").write_text("window.NEUROMETRIC_OASIS7_LONGITUDINAL = " + payload + ";\n")
    print("Actual OAS2_0007 follow-up connected; change rate remains withheld until technical QC", flush=True)


if __name__ == "__main__":
    main()
