import json
from pathlib import Path


def test_project_dkt_grouping_is_explicit_bilateral_and_nonoverlapping():
    mapping = json.loads((Path(__file__).resolve().parents[1] / "data/dkt_lobe_groups.json").read_text())
    used = set()
    for name, labels in mapping.items():
        assert "bilateral" in name
        assert not used.intersection(labels)
        assert len(labels) == len(set(labels))
        used.update(labels)
        if "cortical" in name:
            left = {i for i in labels if 1000 <= i < 2000}
            assert {i + 1000 for i in left} == {i for i in labels if 2000 <= i < 3000}
    assert "Hippocampus (bilateral)" in mapping
