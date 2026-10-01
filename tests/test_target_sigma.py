import unittest
import numpy as np
import torch
from v3_model import gaussian_targets, expected_age
from v3_train import metric


class TargetSigmaTest(unittest.TestCase):
    def test_narrow_concentrates_without_rounding_labels(self):
        age = torch.tensor([18., 35.25, 59.5, 100.])
        narrow, broad = gaussian_targets(age, 1.), gaussian_targets(age, 2.)
        torch.testing.assert_close(narrow.sum(1), torch.ones(4))
        self.assertTrue(torch.isfinite(narrow).all())
        self.assertGreater(float(narrow[1, 35 - 18]), float(broad[1, 35 - 18]))
        torch.testing.assert_close(expected_age(narrow[1:2].clamp_min(1e-30).log()), age[1:2])
        self.assertFalse(torch.equal(narrow[1], gaussian_targets(torch.tensor([35.]), 1.)[0]))

    def test_invalid_sigma_and_age(self):
        for sigma in [0., -.1, .49, 2.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): gaussian_targets(torch.tensor([30.]), sigma)
        for age in [17., 101., float('nan')]:
            with self.assertRaises(ValueError): gaussian_targets(torch.tensor([age]), 1.)

    def test_loss_gradients_finite(self):
        logits = torch.zeros(3, 83, requires_grad=True)
        age = torch.tensor([18., 39.25, 100.])
        loss = torch.nn.functional.kl_div(logits.log_softmax(1), gaussian_targets(age, 1.), reduction='batchmean')
        loss = loss + .05 * torch.nn.functional.l1_loss(expected_age(logits), age)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_metrics_keep_bankers_rounding_and_continuous_tolerance(self):
        y = np.array([20.5, 21.5, 30.2, 40.])
        p = np.array([20., 22., 31.21, 41.])
        score = metric(y, p)
        self.assertEqual(score['rounded_year_match'], .5)
        self.assertEqual(score['within_1_year'], .75)
