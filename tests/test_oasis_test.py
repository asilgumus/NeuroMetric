import unittest

import numpy as np
import pandas as pd
import torch

from oasis_test import eligible_clinical, matched_manifest, matched_statistics, LOCK, ROOT
from v3_model import SFCN
from brainage import sha256
import json


class OasisTests(unittest.TestCase):
    def test_baseline_only_and_no_missing_diagnosis(self):
        data = pd.DataFrame({"ID": ["OAS1_0001_MR1", "OAS1_0001_MR2", "OAS1_0002_MR1", "OAS1_0003_MR1"],
                             "Age": [70, 72, 75, 40], "M/F": ["F"] * 4,
                             "CDR": [1, 1, np.nan, 0]})
        self.assertEqual(eligible_clinical(data).ID.tolist(), ["OAS1_0001_MR1"])

    def test_matching_sex_caliper_without_replacement(self):
        ids = [f"OAS1_{i:04d}_MR1" for i in range(5)]
        data = pd.DataFrame({"ID": ids, "Age": [70, 71, 72, 70, 90],
                             "M/F": ["F", "F", "F", "M", "F"], "CDR": [1, 1, 0, 0, 0]})
        cohort, missing = matched_manifest(data, dict.fromkeys(ids, "/raw.nii.gz"))
        self.assertEqual(len(cohort), 2)
        self.assertFalse(cohort.subject_id.duplicated().any())
        self.assertEqual(cohort.sex.unique().tolist(), ["F"])
        self.assertEqual(missing, [])

    def test_paired_gap_not_age_prediction_accuracy(self):
        rows = [{"subject_id": f"{group}{i}", "pair_id": i, "group": group,
                 "brain_age_gap": gap} for i in range(6)
                for group, gap in [("control", 2), ("dementia_AD_cohort", 7)]]
        stats = matched_statistics(pd.DataFrame(rows), replicates=100)
        self.assertEqual(stats["mean_paired_gap_difference_years"], 5.)
        self.assertEqual(stats["bootstrap_95_ci"], [5., 5.])

    def test_locked_real_model_loads(self):
        lock = json.loads(LOCK.read_text())
        checkpoint = ROOT / lock["checkpoint"]
        self.assertEqual(sha256(checkpoint), lock["sha256"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = SFCN()
        model.load_state_dict(state["model_state"], strict=True)
        self.assertEqual(state["model"], "sfcn")


if __name__ == "__main__":
    unittest.main()
