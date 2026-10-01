import nibabel as nib
import numpy as np
import pytest
from tools.regional_measures import measure_regions


def images():
    mri = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.diag([2., 2., 2., 1.]))
    mri.header.set_xyzt_units("mm")
    labels = np.zeros((4, 4, 4), dtype=np.int16)
    labels[:2] = 17
    seg = nib.Nifti1Image(labels, mri.affine)
    return mri, seg


def test_native_voxel_volume_and_missing_label():
    mri, seg = images()
    rows = measure_regions(mri, seg, {"Hippocampus": [17], "Missing": [53]})
    assert rows[0]["volume_ml"] == pytest.approx(0.256)
    assert rows[0]["regional_brain_age"] is None
    assert rows[1]["volume_ml"] is None
    assert rows[1]["status"] == "label_missing"


def test_wrong_grid_and_double_counting_rejected():
    mri, seg = images()
    with pytest.raises(ValueError, match="unique"):
        measure_regions(mri, seg, {"A": [17], "B": [17]})
    bad = nib.Nifti1Image(seg.get_fdata(), np.eye(4))
    with pytest.raises(ValueError, match="affine"):
        measure_regions(mri, bad, {"A": [17]})


def test_unknown_spatial_units_rejected():
    mri, seg = images()
    mri.header.set_xyzt_units("unknown")
    with pytest.raises(ValueError, match="millimetres"):
        measure_regions(mri, seg, {"A": [17]})
