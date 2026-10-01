import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from brainage import check_volume, clean_demographics, metrics, split_participants


class PipelineTests(unittest.TestCase):
    def population(self):
        return pd.DataFrame([{'subject_id': f'IXI{i:03}', 'site': ['HH', 'Guys', 'IOP'][i % 3],
                              'age': 20 + (i // 3) % 65} for i in range(540)])

    def test_split_disjoint_reproducible_and_order_independent(self):
        frame = self.population()
        first, strategy = split_participants(frame)
        second, _ = split_participants(frame.sample(frac=1, random_state=9))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(strategy, 'site_and_age_tertile')
        groups = {s: set(first.loc[first.split == s, 'subject_id']) for s in ['train', 'val', 'test']}
        self.assertEqual(len(groups['train']), 378)
        self.assertEqual(len(groups['test']), 81)
        self.assertFalse(groups['train'] & groups['val'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['val'] & groups['test'])
        for split in groups:
            self.assertEqual(set(first.loc[first.split == split, 'site']), {'HH', 'Guys', 'IOP'})

    def test_duplicate_participants_rejected(self):
        frame = self.population()
        with self.assertRaisesRegex(ValueError, 'Repeated participant'):
            split_participants(pd.concat([frame, frame.iloc[:1]]))

    def test_conflicting_ages_excluded_equal_duplicates_collapsed(self):
        frame = pd.DataFrame({'IXI_ID': [1, 1, 2, 2, 3, 4], 'AGE': [40, 40, 50, 55, np.nan, 60]})
        clean, excluded = clean_demographics(frame)
        self.assertEqual(clean.IXI_ID.tolist(), [1, 4])
        self.assertEqual({r['reason'] for r in excluded}, {'conflicting_ages', 'missing_age'})

    def test_metrics_known_values(self):
        result = metrics([40., 60.], [43., 56.])
        self.assertEqual(result['mae'], 3.5)
        self.assertAlmostEqual(result['rmse'], np.sqrt(12.5))
        self.assertEqual(result['mean_gap'], -.5)
        self.assertIsNone(metrics([40.], [43.])['r2'])

    def test_nonfinite_prediction_rejected(self):
        with self.assertRaises(ValueError):
            metrics([40], [float('nan')])

    def test_invalid_volumes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scan.nii.gz'
            for data in [np.zeros((40, 40, 40)), np.zeros((40, 40)), np.full((40, 40, 40), np.nan)]:
                nib.save(nib.Nifti1Image(data, np.eye(4)), path)
                with self.assertRaises(ValueError):
                    check_volume(path)
            rng = np.random.default_rng(42)
            nib.save(nib.Nifti1Image(rng.normal(size=(40, 40, 40)), np.eye(4)), path)
            check_volume(path)


if __name__ == '__main__':
    unittest.main()
