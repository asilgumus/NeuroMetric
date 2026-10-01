import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_longitudinal_demo_uses_one_patient_real_days_and_locked_model():
    result = json.loads((ROOT / "artifacts/longitudinal/OAS2_0001_predictions.json").read_text())
    assert result["comparable"]
    assert result["time_basis"] == "days_from_baseline"
    assert [v["mr_delay_days"] for v in result["visits"]] == [0, 457]
    assert len({v["patient_id"] for v in result["visits"]}) == 1
    assert len({v["checkpoint_sha256"] for v in result["visits"]}) == 1
    assert all(v["scan_date"] is None for v in result["visits"])
    assert result["gap_change_years"] == pytest.approx(1.7607345581054688)


def test_regional_export_has_five_complete_groups_without_regional_ages():
    result = json.loads((ROOT / "artifacts/alzheimer/fastsurfer_verified_OAS1_0030/regional_measures.json").read_text())
    assert result["clinical_validation"] is False
    assert result["qc_scope"] == "technical_segmentation_only"
    assert len(result["regions"]) == 5
    for region in result["regions"]:
        assert region["status"] == "measured"
        assert not region["missing_labels"]
        assert region["regional_brain_age"] is None
        assert region["volume_ml"] > 0


def test_oasis30_trend_matches_corrected_gradcam_prediction():
    trend = json.loads((ROOT / "artifacts/alzheimer/longitudinal_OAS1_0030.json").read_text())
    heat = json.loads((ROOT / "artifacts/alzheimer/gradcam_verified_OAS1_0030/metadata.json").read_text())
    assert trend["visits"][0]["prediction"] == pytest.approx(heat["prediction"])
    assert trend["gap_change_per_year"] is None
