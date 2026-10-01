from pathlib import Path
from tools.oasis_visit_selection import visit_ids

METADATA = Path(__file__).resolve().parents[1] / "data/oasis2/demographics.xlsx"


def test_missing_mr2_is_not_invented():
    assert visit_ids(METADATA, "OAS2_0007") == ["OAS2_0007_MR1", "OAS2_0007_MR3"]


def test_consecutive_visits_still_work():
    assert visit_ids(METADATA, "OAS2_0001") == ["OAS2_0001_MR1", "OAS2_0001_MR2"]
