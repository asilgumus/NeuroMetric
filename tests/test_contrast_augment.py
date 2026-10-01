import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from v3_train import gamma_contrast, Volumes


class ContrastTests(unittest.TestCase):
    def test_contrast_preserves_background_mean_and_order(self):
        x = np.array([0., -0.01, 1., 2., 8.], dtype=np.float32)
        for gamma in (.8, 1., 1.2):
            y = gamma_contrast(x, gamma)
            np.testing.assert_array_equal(y[:2], x[:2])
            self.assertAlmostEqual(float(y.mean()), float(x.mean()), places=6)
            self.assertTrue(np.all(np.diff(y[2:]) > 0))
            self.assertEqual(y.dtype, np.float32)
            self.assertTrue(np.isfinite(y).all())
        np.testing.assert_allclose(gamma_contrast(x, 1.), x)
        self.assertFalse(np.allclose(gamma_contrast(x, .8), x))
        np.testing.assert_array_equal(x, [0., np.float32(-.01), 1., 2., 8.])

    def test_invalid_gamma_or_input_rejected(self):
        for gamma in (0., 2., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                gamma_contrast(np.ones(2), gamma)
        for x in (np.zeros(2), np.array([1., np.nan])):
            with self.assertRaises(ValueError):
                gamma_contrast(x, 1.)

    def test_validation_never_applies_contrast(self):
        rows = pd.DataFrame([{'dataset': 'SALD', 'subject_id': 'one',
                              'age': 40., 'sfcn_array': 'unused.npy'}])
        with patch('v3_train.np.load', return_value=np.ones((160, 192, 160))), \
             patch('v3_train.gamma_contrast') as contrast:
            _, age = Volumes(rows, {'SALD': __import__('pathlib').Path('.')},
                             'sfcn', augment=False, contrast_augment=True)[0]
            contrast.assert_not_called()
            self.assertEqual(float(age), 40.)

    def test_opt_in_train_augmentation_applies(self):
        rows = pd.DataFrame([{'dataset': 'SALD', 'subject_id': 'one',
                              'age': 40., 'sfcn_array': 'unused.npy'}])
        with patch('v3_train.np.load', return_value=np.ones((160, 192, 160))), \
             patch('v3_train.np.random.random', return_value=0.), \
             patch('v3_train.gamma_contrast', wraps=gamma_contrast) as contrast:
            image, age = Volumes(rows, {'SALD': __import__('pathlib').Path('.')},
                                'sfcn', augment=True, contrast_augment=True)[0]
            contrast.assert_called_once()
            self.assertTrue(np.isfinite(image.numpy()).all())
            self.assertEqual(float(age), 40.)
