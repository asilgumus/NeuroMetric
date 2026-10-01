import unittest
from v3_train import hit_selection_key


class HitSelectionTest(unittest.TestCase):
    def score(self, exact, within, mae):
        return {"rounded_year_match": exact, "within_1_year": within, "mae": mae}

    def test_exact_has_priority(self):
        self.assertGreater(hit_selection_key(self.score(.15, .20, 3.6)),
                           hit_selection_key(self.score(.14, .25, 3.4)))

    def test_within_one_breaks_exact_tie(self):
        self.assertGreater(hit_selection_key(self.score(.15, .23, 3.6)),
                           hit_selection_key(self.score(.15, .22, 3.4)))

    def test_mae_breaks_hit_tie(self):
        self.assertGreater(hit_selection_key(self.score(.15, .23, 3.4)),
                           hit_selection_key(self.score(.15, .23, 3.6)))

    def test_equal_scores_retain_parent(self):
        score = self.score(.15, .23, 3.4)
        self.assertFalse(hit_selection_key(score) > hit_selection_key(score))
