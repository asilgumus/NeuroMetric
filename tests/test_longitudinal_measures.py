import pytest
from tools.longitudinal_measures import summarize_visits


def visit(stamp="2024-01-01", age=65, prediction=63.4, **changes):
    return {"patient_id": "sample", "scan_date": stamp, "age": age,
            "prediction": prediction, "checkpoint_sha256": "same-model",
            "preprocessing_signature": "same-prep", "visual_qc": "approved", **changes}


def test_single_baseline_never_has_a_rate():
    result = summarize_visits([visit(scan_date=None)])
    assert result["gap_change_per_year"] is None
    assert "follow_up_required" in result["withheld_reasons"]


def test_oasis_relative_days_without_invented_dates():
    result = summarize_visits([visit(None, 67, 66.4, mr_delay_days=730),
                               visit(None, mr_delay_days=0)])
    assert result["comparable"]
    assert result["time_basis"] == "days_from_baseline"
    assert result["elapsed_years"] == pytest.approx(730 / 365.25)
    assert all(v["scan_date"] is None for v in result["visits"])


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf"), 0])
def test_invalid_or_duplicate_relative_days_rejected(delay):
    with pytest.raises(ValueError):
        summarize_visits([visit(None, mr_delay_days=0), visit(None, mr_delay_days=delay)])


def test_dated_change_is_sorted_and_annualized():
    result = summarize_visits([visit("2026-01-01", 67, 66.4), visit()])
    assert result["comparable"]
    assert result["gap_change_years"] == pytest.approx(1)
    assert result["gap_change_per_year"] == pytest.approx(1 / (731 / 365.25))


@pytest.mark.parametrize("change", [{"checkpoint_sha256": "other"}, {"visual_qc": "pending"}, {"preprocessing_signature": "other"}])
def test_incomparable_visits_withhold_rates(change):
    result = summarize_visits([visit(), visit("2026-01-01", 67, 66.4, **change)])
    assert not result["comparable"]
    assert result["gap_change_per_year"] is None


def test_mixed_patients_missing_dates_and_duplicate_dates_rejected():
    for second in [visit(patient_id="other"), visit(scan_date=None), visit()]:
        with pytest.raises(ValueError):
            summarize_visits([visit(), second])
