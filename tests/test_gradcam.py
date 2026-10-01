import unittest
import torch
from tools.gradcam_oasis30 import signed_cam


class GradCamTest(unittest.TestCase):
    def test_preserves_positive_and_negative_attribution(self):
        activations = torch.tensor([[[[[1., -1.]]]]])
        result = signed_cam(activations, torch.ones_like(activations), (1, 1, 2))
        self.assertEqual(result.shape, (1, 1, 2))
        self.assertGreater(float(result[0,0,0]), 0)
        self.assertLess(float(result[0,0,1]), 0)

    def test_rejects_degenerate_map(self):
        with self.assertRaises(ValueError):
            signed_cam(torch.zeros(1,1,2,2,2), torch.ones(1,1,2,2,2), (4,4,4))
