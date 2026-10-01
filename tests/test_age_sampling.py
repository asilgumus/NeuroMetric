import unittest
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from v3_train import age_sampling_weights


class AgeSamplingTests(unittest.TestCase):
    def test_frequency_cap_and_alignment(self):
        rows = pd.DataFrame({"age": [80, 30, 30, 30, 30] + [20] * 16,
                             "split": ["train"] * 21})
        weights = age_sampling_weights(rows)
        self.assertEqual(weights.tolist(), [2.] * 5 + [1.] * 16)
        sampler = lambda: list(WeightedRandomSampler(weights, len(rows), replacement=True,
                                      generator=torch.Generator().manual_seed(42)))
        self.assertEqual(sampler(), sampler())
        self.assertEqual(len(sampler()), len(rows))
        self.assertTrue(all(0 <= i < len(rows) for i in sampler()))

    def test_rejects_validation_and_invalid_age(self):
        for ages, splits in [([40], ["val"]), ([float("nan")], ["train"]),
                             ([17], ["train"]), ([101], ["train"]), ([], [])]:
            with self.assertRaises(ValueError):
                age_sampling_weights(pd.DataFrame({"age": ages, "split": splits}))
