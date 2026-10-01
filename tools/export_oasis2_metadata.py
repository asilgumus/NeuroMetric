"""Export genuine longitudinal reference measurements, separately from AI output."""
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json


def main():
    source = ROOT / "data/oasis2/demographics.xlsx"
    frame = pd.read_excel(source)
    frame = frame[frame["MRI ID"].isin(["OAS2_0001_MR1", "OAS2_0001_MR2"])].sort_values("MR Delay")
    if len(frame) != 2 or frame["Subject ID"].nunique() != 1:
        raise ValueError("Expected exactly two visits from one subject")
    visits = []
    for _, row in frame.iterrows():
        visits.append({"scan_id": row["MRI ID"], "age": int(row.Age), "mr_delay_days": int(row["MR Delay"]),
                       "cdr": float(row.CDR), "mmse": float(row.MMSE), "etiv_ml": float(row.eTIV),
                       "normalized_whole_brain_volume": float(row.nWBV),
                       "model_prediction": None, "model_qc": "pending"})
    payload = {"patient_id": "OAS2_0001", "dataset": "OASIS-2", "metadata_sha256": sha256(source),
               "source_url": "https://sites.wustl.edu/oasisbrains/home/oasis-2/", "time_basis": "days_from_baseline",
               "visits": visits, "elapsed_days": visits[-1]["mr_delay_days"] - visits[0]["mr_delay_days"],
               "nwbv_change_percentage_points": (visits[-1]["normalized_whole_brain_volume"] - visits[0]["normalized_whole_brain_volume"]) * 100,
               "limitations": "Dataset reference nWBV, not model-predicted brain age or regional volume. No calendar dates or diagnoses inferred."}
    write_json(ROOT / "artifacts/longitudinal/OAS2_0001_reference.json", payload)
    (ROOT / "assets/oasis2-longitudinal-reference.js").write_text("window.NEUROMETRIC_OASIS2_REFERENCE = " + json.dumps(payload) + ";\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
