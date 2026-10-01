"""Compare dated visits from one patient; report descriptive change, not diagnosis."""
import argparse
import csv
from datetime import date
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainage import write_json


def summarize_visits(visits):
    if not visits:
        raise ValueError("At least one visit required")
    patients = {v.get("patient_id") for v in visits}
    if len(patients) != 1 or not next(iter(patients)):
        raise ValueError("Visits must belong to exactly one identified patient")
    clean = []
    relative = all(v.get("mr_delay_days") not in (None, "") for v in visits)
    for visit in visits:
        age, prediction = float(visit["age"]), float(visit["prediction"])
        if not all(math.isfinite(x) and 0 <= x <= 130 for x in (age, prediction)):
            raise ValueError("Finite ages between 0 and 130 required")
        stamp = visit.get("scan_date") or None
        if stamp:
            stamp = date.fromisoformat(stamp).isoformat()
        clean.append({**visit, "scan_date": stamp, "age": age,
                      "prediction": prediction, "brain_age_gap": prediction - age})
        if relative:
            delay = float(visit["mr_delay_days"])
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("Nonnegative finite MR delay required")
            clean[-1]["mr_delay_days"] = delay
    dates = [v["scan_date"] for v in clean]
    times = [v["mr_delay_days"] for v in clean] if relative else dates
    if len(clean) > 1 and (None in times or len(set(times)) != len(times)):
        raise ValueError("Distinct scan dates required for multiple visits")
    clean.sort(key=lambda v: v["mr_delay_days"] if relative else v["scan_date"] or "")
    reasons = []
    if len(clean) < 2:
        reasons.append("follow_up_required")
    for field in ("checkpoint_sha256", "preprocessing_signature"):
        tokens = {v.get(field) for v in clean}
        if len(tokens) != 1 or not next(iter(tokens)):
            reasons.append(field + "_missing_or_mismatched")
    if any(v.get("visual_qc") != "approved" for v in clean):
        reasons.append("visual_qc_pending")
    result = {"patient_id": clean[0]["patient_id"], "visits": clean,
              "time_basis": "days_from_baseline" if relative else "calendar_date",
              "comparable": not reasons, "withheld_reasons": reasons,
              "elapsed_years": None, "gap_change_years": None,
              "gap_change_per_year": None, "predicted_age_change_per_year": None,
              "interpretation": "Descriptive model change only; not a validated aging rate or disease assessment."}
    if not reasons:
        first, last = clean[0], clean[-1]
        elapsed = ((last["mr_delay_days"] - first["mr_delay_days"]) if relative else
                   (date.fromisoformat(last["scan_date"]) - date.fromisoformat(first["scan_date"])).days) / 365.25
        expected_age_change = elapsed
        # Broad guard against assigning another patient's age to a visit.
        if abs((last["age"] - first["age"]) - expected_age_change) > 1.1:
            raise ValueError("Age progression conflicts with scan dates")
        change = last["brain_age_gap"] - first["brain_age_gap"]
        result.update(elapsed_years=elapsed, gap_change_years=change,
                      gap_change_per_year=change / elapsed,
                      predicted_age_change_per_year=(last["prediction"] - first["prediction"]) / elapsed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with patient_id, scan_date, age, prediction, checkpoint_sha256, preprocessing_signature, visual_qc")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--javascript-output", type=Path, help="Optional static dashboard data export")
    args = parser.parse_args()
    with args.input.open(newline="") as source:
        result = summarize_visits(list(csv.DictReader(source)))
    write_json(args.output, result)
    if args.javascript_output:
        args.javascript_output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result, ensure_ascii=True).replace("<", "\\u003c")
        args.javascript_output.write_text("window.NEUROMETRIC_LONGITUDINAL = " + payload + ";\n")


if __name__ == "__main__":
    main()
