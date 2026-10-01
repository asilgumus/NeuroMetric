import unittest
import numpy as np
import torch
from v3_model import rounded_interval_loss, expected_age, gaussian_targets


class RoundedIntervalLossTests(unittest.TestCase):
    def test_fractional_label_gradient_targets_hit_interval(self):
        # 35.49 is already an exact hit. A prediction at 35.6 must move down,
        # even though both are close to the continuous label.
        age = torch.tensor([35.49])
        pred = torch.tensor([35.6], requires_grad=True)
        rounded_interval_loss(pred, age).backward()
        self.assertGreater(pred.grad.item(), 0.)
        self.assertLess(rounded_interval_loss(torch.tensor([35.]), age).item(),
                        rounded_interval_loss(pred.detach(), age).item())
        self.assertEqual(age.item(), torch.tensor([35.49]).item())

    def test_ties_to_even_and_boundaries(self):
        age = torch.tensor([34.5, 35.5, 18., 100.])
        np.testing.assert_array_equal(torch.round(age).numpy(), np.rint(age.numpy()))
        center = torch.round(age)
        torch.testing.assert_close(rounded_interval_loss(center - .5, age),
                                   rounded_interval_loss(center + .5, age))
        self.assertLess(rounded_interval_loss(center, age).item(),
                        rounded_interval_loss(center + .5, age).item())

    def test_bounded_and_far_gradient_from_base_loss(self):
        age = torch.tensor([18., 35.49, 100.])
        logits = torch.zeros(3, 83, requires_grad=True)
        pred = expected_age(logits)
        aux = rounded_interval_loss(pred, age)
        self.assertTrue(0 <= aux.item() <= 1)
        loss = torch.nn.functional.kl_div(logits.log_softmax(1), gaussian_targets(age, 1.), reduction='batchmean')
        loss = loss + .05 * (pred - age).abs().mean() + .25 * aux
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs().sum(1) > 0).all())

    def test_invalid_inputs(self):
        for pred, age in [(torch.tensor([]), torch.tensor([])),
                          (torch.tensor([30.]), torch.tensor([17.])),
                          (torch.tensor([float('nan')]), torch.tensor([30.])),
                          (torch.tensor([30., 31.]), torch.tensor([30.]))]:
            with self.assertRaises(ValueError):
                rounded_interval_loss(pred, age)
