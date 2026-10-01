import unittest
import pandas as pd
from v3_test_sfcn import heldout_groups


class HeldoutTest(unittest.TestCase):
    def test_no_training_or_validation_rows(self):
        rows = pd.DataFrame([{"dataset": "IXI", "split": "test"}] * 85 +
                            [{"dataset": "DLBS", "split": "external"}] * 464 +
                            [{"dataset": "IXI", "split": "train"}] * 10 +
                            [{"dataset": "IXI", "split": "val"}] * 10)
        groups = heldout_groups(rows)
        self.assertEqual(len(groups["ixi_test"]), 85)
        self.assertEqual(len(groups["dlbs_external"]), 464)
        self.assertFalse(any(g.split.isin(["train", "val"]).any() for g in groups.values()))

    def test_incomplete_cohort_rejected(self):
        with self.assertRaises(ValueError):
            heldout_groups(pd.DataFrame([{"dataset": "IXI", "split": "test"}]))
