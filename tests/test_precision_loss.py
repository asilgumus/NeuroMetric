import unittest
import torch
from v3_model import precision_loss, expected_age, gaussian_targets


class PrecisionLossTests(unittest.TestCase):
    def test_continuous_labels_and_bounds(self):
        age = torch.tensor([35.25])
        self.assertEqual(float(precision_loss(age, age)), 0.)
        self.assertGreater(float(precision_loss(torch.tensor([35.]), age)), 0.)
        near = precision_loss(age + .25, age)
        far = precision_loss(age + 10., age)
        self.assertLess(float(near), float(far))
        self.assertLessEqual(float(far), 1.)

    def test_gradient_concentrates_near_half_year(self):
        age = torch.tensor([40., 40.])
        pred = torch.tensor([40.5, 43.], requires_grad=True)
        precision_loss(pred, age).backward()
        self.assertGreater(float(pred.grad[0]), 0.)
        self.assertGreater(float(pred.grad[0]), float(pred.grad[1]) * 100)

    def test_combined_loss_reaches_far_errors(self):
        logits = torch.zeros(3, 83, requires_grad=True)
        age = torch.tensor([18., 35.25, 100.])
        pred = expected_age(logits)
        loss = torch.nn.functional.kl_div(logits.log_softmax(1), gaussian_targets(age, 1.), reduction='batchmean')
        loss = loss + .05 * (pred - age).abs().mean() + .25 * precision_loss(pred, age)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs().sum(1) > 0).all())

    def test_invalid_inputs(self):
        for pred, age in [(torch.tensor([]), torch.tensor([])),
                          (torch.tensor([30.]), torch.tensor([17.])),
                          (torch.tensor([float('nan')]), torch.tensor([30.]))]:
            with self.assertRaises(ValueError):
                precision_loss(pred, age)
