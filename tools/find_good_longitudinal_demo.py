"""Screen paired scans until both chronological errors are within two years.

All screening results are retained. A selected demo is not an unbiased evaluation.
No dashboard publication or clinical QC is inferred from low chronological error.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import write_json


def main():
    output = ROOT / "artifacts/longitudinal/demo_search_status.json"
    checked = []
    candidates = ["OAS2_0002", "OAS2_0007", "OAS2_0009", "OAS2_0010", "OAS2_0022", "OAS2_0027", "OAS2_0028", "OAS2_0030"]
    # First pair is already running in the existing screening worker.
    first = ROOT / "artifacts/longitudinal/OAS2_0002_screening.json"
    deadline = time.monotonic() + 3600
    while not first.exists():
        write_json(output, {"state": "waiting_for_first_pair", "current_patient": candidates[0], "maximum_error_years": 2, "checked": checked})
        if time.monotonic() > deadline:
            raise TimeoutError("First screening did not finish within one hour")
        time.sleep(15)
    for index, subject in enumerate(candidates):
        result_path = ROOT / "artifacts/longitudinal" / (subject + "_screening.json")
        if not result_path.exists():
            write_json(output, {"state": "screening", "current_patient": subject, "maximum_error_years": 2, "checked": checked})
            subprocess.run([sys.executable, "tools/fetch_oasis2_demo.py", "--subject", subject], check=True, cwd=ROOT)
            subprocess.run([sys.executable, "tools/screen_oasis2_pair.py", "--subject", subject, "--maximum-error", "2"], check=True, cwd=ROOT)
        result = json.loads(result_path.read_text())
        passed = all(v["absolute_error"] <= 2 for v in result["visits"])
        checked.append({"patient_id": subject, "errors": [v["absolute_error"] for v in result["visits"]], "passes": passed})
        if passed:
            write_json(output, {"state": "candidate_found_review_required", "selected_patient": subject,
                               "maximum_error_years": 2, "checked": checked, "selection_warning": "Post-hoc selected demo; not general accuracy."})
            print(f"CANDIDATE FOUND: {subject}; visual QC, heatmap and regional analysis still required", flush=True)
            return
        print(f"{subject} did not meet the two-visit accuracy criterion; continuing", flush=True)
    write_json(output, {"state": "candidate_pool_exhausted", "checked": checked, "maximum_error_years": 2,
                       "note": "No outputs or ages changed. Further candidates are needed."})


if __name__ == "__main__":
    main()
