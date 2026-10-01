import csv
import json
from pathlib import Path
import tempfile
import unittest

from tools.v3_autopilot import completion_evidence, should_handoff


class AutopilotTests(unittest.TestCase):
    def test_interval_nll_keeps_strict_legacy_mae_terminal_proof(self):
        folder = self.case([2, 3, 3], 40, 2, [2, 2])
        path = folder / 'training/config.json'
        config = json.loads(path.read_text())
        config['sfcn_rounded_interval_nll'] = True
        path.write_text(json.dumps(config))
        result = completion_evidence(folder)
        self.assertEqual(result['reason'], 'early_stopping')
        self.assertTrue(should_handoff(result, False))
        with (folder / 'training/history.csv').open('a') as stream:
            stream.write('block,4,1.9\n')
        with self.assertRaises(ValueError):
            completion_evidence(folder)

    def test_finite_budget_disables_agent_handoff(self):
        self.assertFalse(should_handoff({"needs_improvement": True}, True))
        self.assertTrue(should_handoff({"needs_improvement": True}, False))
        self.assertFalse(should_handoff({"needs_improvement": False}, False))

    def test_history_cannot_continue_after_patience(self):
        with self.assertRaises(ValueError):
            completion_evidence(self.case([2, 3, 3, 2], 4, 2, [2, 2]))

    def case(self, scores, limit, patience, errors):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        p = folder / "training"
        p.mkdir()
        best = sum(errors) / len(errors)
        for name, value in {
            "config.json": {"model": "sfcn", "epochs_head": 0, "epochs_block": limit,
                            "patience": patience, "val_n": len(errors)},
            "selection.json": {"validation_mae": best, "test_used_for_selection": False},
            "baseline_metrics.json": {"mae": best + 1},
        }.items():
            (p / name).write_text(json.dumps(value))
        with (p / "history.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["stage", "epoch", "val_mae"])
            writer.writeheader()
            for i, score in enumerate(scores, 1):
                writer.writerow({"stage": "block", "epoch": i, "val_mae": score})
        with (p / "best_validation_predictions.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["subject_id", "age", "prediction"])
            writer.writeheader()
            for i, error in enumerate(errors):
                writer.writerow({"subject_id": str(i), "age": 40, "prediction": 40 + error})
        return folder

    def test_verified_early_stopping_triggers(self):
        result = completion_evidence(self.case([2, 3, 3], 80, 2, [2, 2]))
        self.assertEqual(result["reason"], "early_stopping")
        self.assertTrue(result["needs_improvement"])

    def test_limit_at_exactly_seventy_percent_triggers(self):
        errors = [0.] * 7 + [2.] * 3
        result = completion_evidence(self.case([.6], 1, 20, errors))
        self.assertEqual(result["within_1_year"], .7)
        self.assertTrue(result["needs_improvement"])

    def test_deeper_block_early_stopping_triggers_even_above_seventy(self):
        folder = self.case([.4, .5, .5], 40, 2, [0.] * 8 + [2.] * 2)
        path = folder / "training/config.json"
        config = json.loads(path.read_text())
        config["sfcn_deeper_refine"] = True
        path.write_text(json.dumps(config))
        result = completion_evidence(folder)
        self.assertEqual(result["reason"], "early_stopping")
        self.assertEqual(result["within_1_year"], .8)
        self.assertTrue(result["needs_improvement"])

    def test_limit_above_seventy_does_not_trigger(self):
        result = completion_evidence(self.case([.4], 1, 20, [0.] * 8 + [2.] * 2))
        self.assertFalse(result["needs_improvement"])

    def test_contrast_block_successor_reapplies_terminal_conditions(self):
        folder = self.case([.6], 1, 20, [0.] * 7 + [2.] * 3)
        path = folder / "training/config.json"
        config = json.loads(path.read_text())
        config["sfcn_contrast_augment"] = True
        path.write_text(json.dumps(config))
        result = completion_evidence(folder)
        self.assertEqual(result["reason"], "epoch_limit")
        self.assertTrue(should_handoff(result, False))

    def test_narrow_loss_preserves_mae_terminal_verification(self):
        folder = self.case([.6], 1, 20, [0.] * 7 + [2.] * 3)
        path = folder / "training/config.json"
        config = json.loads(path.read_text())
        config["sfcn_target_sigma"] = 1.
        path.write_text(json.dumps(config))
        result = completion_evidence(folder)
        self.assertEqual(result["reason"], "epoch_limit")
        self.assertTrue(should_handoff(result, False))
        with (folder / "training/history.csv").open("a") as stream:
            stream.write("block,2,0.5\n")
        with self.assertRaises(ValueError):
            completion_evidence(folder)

    def test_incomplete_history_does_not_mean_early_stopping(self):
        with self.assertRaises(ValueError):
            completion_evidence(self.case([2, 3], 80, 20, [2, 2]))

    def test_interval_loss_retains_strict_terminal_and_continuation(self):
        folder = self.case([2, 3, 3], 40, 2, [2, 2])
        path = folder / "training/config.json"
        config = json.loads(path.read_text())
        config["sfcn_rounded_interval_loss"] = True
        path.write_text(json.dumps(config))
        result = completion_evidence(folder)
        self.assertEqual(result["reason"], "early_stopping")
        self.assertTrue(should_handoff(result, False))
        selection = folder / "training/selection.json"
        selection.write_text(json.dumps({"validation_mae": 1., "test_used_for_selection": False}))
        with self.assertRaises(ValueError):
            completion_evidence(folder)

    def test_mismatching_predictions_rejected(self):
        with self.assertRaises(ValueError):
            completion_evidence(self.case([1.], 1, 20, [2, 2]))


if __name__ == "__main__":
    unittest.main()
