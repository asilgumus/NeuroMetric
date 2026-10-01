import unittest
import pandas as pd
from tools.predict_oasis_prepared import rank_predictions


class RankingTest(unittest.TestCase):
    def test_ranks_without_discarding_bad_predictions(self):
        frame = pd.DataFrame({"subject_id":["c","b","a"],"age":[70,60,50],"prediction":[60,60.4,50.4]})
        ranked = rank_predictions(frame)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked.subject_id.tolist(), ["a","b","c"])
        self.assertEqual(ranked.within_1_year.tolist(), [True,True,False])
        self.assertEqual(ranked.rounded_year_match.tolist(), [True,True,False])
