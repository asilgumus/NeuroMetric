import pytest
from tools.export_regional_dashboard import validate_report


def report():
    return {"visual_qc":"user_approved", "mri_sha256":"patient-hash",
            "regions":[{"region":"Hippocampus","status":"measured","volume_ml":3.5,"regional_brain_age":None}]}


def test_valid_reviewed_report():
    data = validate_report(report(), "patient-hash")
    assert data["visual_qc"] == "approved"
    assert data["regions"][0]["volume_ml"] == 3.5


def test_wrong_patient_and_unreviewed_report_rejected():
    with pytest.raises(ValueError, match="MRI"):
        validate_report(report(), "different-patient")
    data = report()
    data["visual_qc"] = "pending"
    with pytest.raises(ValueError, match="Reviewed"):
        validate_report(data, "patient-hash")


@pytest.mark.parametrize("volume", [0, -1, float("nan"), True])
def test_invalid_volumes_rejected(volume):
    data = report()
    data["regions"][0]["volume_ml"] = volume
    with pytest.raises(ValueError, match="positive"):
        validate_report(data, "patient-hash")
