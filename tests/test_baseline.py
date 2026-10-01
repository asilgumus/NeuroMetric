import unittest

import pandas as pd
import torch

from v3_baseline import select_validation, released_sfcn_age


class BaselineTests(unittest.TestCase):
    def test_provisional_baseline_cannot_select_held_out_subjects(self):
        rows = pd.DataFrame({"dataset": ["IXI", "IXI", "IXI", "DLBS"],
                             "subject_id": ["train", "val", "test", "external"],
                             "split": ["train", "val", "test", "external"]})
        self.assertEqual(select_validation(rows).subject_id.tolist(), ["val"])
        with self.assertRaises(ValueError):
            select_validation(rows.loc[rows.split == "test"])
        rows.loc[3, "split"] = "val"
        with self.assertRaises(ValueError):
            select_validation(rows)

    def test_original_sfcn_half_year_bin_centers_are_preserved(self):
        logits = torch.full((2, 40), -100.)
        logits[0, 0] = 100.
        logits[1, 39] = 100.
        self.assertEqual(released_sfcn_age(logits).tolist(), [42.5, 81.5])
        with self.assertRaises(ValueError):
            released_sfcn_age(torch.zeros(1, 83))


if __name__ == "__main__":
    unittest.main()
