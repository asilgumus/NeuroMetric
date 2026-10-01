import unittest
import numpy as np
import torch
import subprocess
import sys
from pathlib import Path
from v3_model import rounded_interval_nll


class RoundedIntervalNLLTests(unittest.TestCase):
    def test_conflicting_auxiliaries_rejected_by_both_entrypoints(self):
        root = Path(__file__).resolve().parents[1]
        cases = [
            ('v3_train.py', ['--catalog', 'unused', '--prepared-root', 'unused',
                             '--output', 'unused', '--model', 'sfcn']),
            ('tools/build_v3_train_kaggle.py', ['sfcn']),
        ]
        for script, args in cases:
            result = subprocess.run([sys.executable, str(root/script), *args,
                                     '--sfcn-rounded-interval-nll', '--sfcn-rounded-interval-loss'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Interval NLL requires refinement', result.stderr)

    def test_matches_log_probability_where_subtraction_is_stable(self):
        age = torch.tensor([35.49, 34.5, 35.5, 18., 100.], dtype=torch.float64)
        pred = torch.round(age) + .7
        center = torch.round(age)
        mass = torch.sigmoid((pred-center+.5)/.25)-torch.sigmoid((pred-center-.5)/.25)
        torch.testing.assert_close(rounded_interval_nll(pred, age), -.25*mass.log().mean())
        np.testing.assert_array_equal(center.numpy(), np.rint(age.numpy()))

    def test_distant_gradients_finite_directed_and_bounded(self):
        for offset in [-82., -10., -.6, 0., .6, 10., 82.]:
            age = torch.tensor([35.49])
            pred = torch.tensor([35.+offset], requires_grad=True)
            loss = rounded_interval_nll(pred, age)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertLessEqual(abs(pred.grad.item()), 1.)
            if offset:
                self.assertGreater(pred.grad.item()*offset, 0.)
            else:
                self.assertAlmostEqual(pred.grad.item(), 0.)
            self.assertEqual(age.item(), torch.tensor([35.49]).item())

    def test_invalid_inputs(self):
        for p, y in [([], []), ([30.], [17.]), ([float('nan')], [30.]),
                     ([30., 31.], [30.]), ([30.], [101.])]:
            with self.assertRaises(ValueError):
                rounded_interval_nll(torch.tensor(p), torch.tensor(y))
