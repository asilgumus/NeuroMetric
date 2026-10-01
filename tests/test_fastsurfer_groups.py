import nibabel as nib
import numpy as np
import pytest
from tools.export_fastsurfer_candidate_regions import grouped_volumes


def test_bilateral_group_volume_and_incomplete_mapping():
    data = np.zeros((4, 4, 4), dtype=np.int16)
    data[:2] = 1003
    data[2:] = 2003
    image = nib.Nifti1Image(np.ones(data.shape, dtype=np.float32), np.diag([2., 2., 2., 1.]))
    labels = nib.Nifti1Image(data, image.affine)
    rows = grouped_volumes(image, labels, {"Bilateral frontal": [1003, 2003], "Absent": [1030, 2030]})
    assert rows[0]["volume_ml"] == pytest.approx(.512)
    assert rows[0]["missing_labels"] == []
    assert rows[1]["volume_ml"] is None
    assert rows[1]["status"] == "incomplete_parcellation"
