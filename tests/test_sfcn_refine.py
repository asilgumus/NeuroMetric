import unittest

import numpy as np
import torch

from v3_model import SFCN
from v3_train import shift_volume, sfcn_refine_groups


class RefineTests(unittest.TestCase):
    def test_deeper_refinement_unfreezes_only_one_additional_block(self):
        model = SFCN()
        original = {k: v.clone() for k, v in model.state_dict().items()}
        groups = sfcn_refine_groups(model, .5, deeper=True)
        self.assertEqual([g["lr"] for g in groups], [5e-7, 1.5e-6, 5e-6, 5e-5])
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith((
                "feature_extractor.conv_3.", "feature_extractor.conv_4.",
                "feature_extractor.conv_5.", "classifier.conv_6.")), name)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_continuation_preserves_halved_terminal_rates(self):
        model = SFCN()
        groups = sfcn_refine_groups(model, .5)
        self.assertEqual([g["lr"] for g in groups], [1.5e-6, 5e-6, 5e-5])
        for invalid in (0, -1, 2, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sfcn_refine_groups(model, invalid)

    def test_translation_zero_pads_without_wrapping(self):
        x = np.arange(64).reshape(4, 4, 4)
        np.testing.assert_array_equal(shift_volume(x, (0, 0, 0)), x)
        shifted = shift_volume(x, (1, -1, 0))
        np.testing.assert_array_equal(shifted[1:, :-1], x[:-1, 1:])
        self.assertFalse(shifted[0].any())
        self.assertFalse(shifted[:, -1].any())

    def test_only_selected_blocks_train_with_distinct_rates(self):
        model = SFCN()
        groups = sfcn_refine_groups(model)
        self.assertEqual([group["lr"] for group in groups], [3e-6, 1e-5, 1e-4])
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        for name, parameter in model.named_parameters():
            expected = name.startswith(("feature_extractor.conv_4.",
                                        "feature_extractor.conv_5.", "classifier.conv_6."))
            self.assertEqual(parameter.requires_grad, expected, name)
        optimizer = torch.optim.AdamW(groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=.5, threshold=0., min_lr=1e-7)
        for _ in range(7):
            scheduler.step(4.5)
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1.5e-6, 5e-6, 5e-5])


if __name__ == "__main__":
    unittest.main()
