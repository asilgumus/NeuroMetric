"""Guard against unattended launches without QC and duplicate running jobs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.v3_training_watch import inspect_once, inspect_exact


class TrainingWatchTests(unittest.TestCase):
    def test_exact_ref_only_observes_and_never_restarts(self):
        for remote in ("RUNNING", "MISSING"):
            with tempfile.TemporaryDirectory() as folder, \
                 patch("tools.v3_training_watch.ROOT", Path(folder)), \
                 patch("tools.v3_training_watch.status", return_value=remote) as status, \
                 patch("tools.v3_training_watch.subprocess.run") as run:
                result = inspect_exact(None, "owner/successor", "_successor")
                self.assertEqual(result, "training_running" if remote == "RUNNING" else "observation_missing")
                status.assert_called_once_with(None, "owner/successor")
                run.assert_not_called()

    def test_missing_qc_never_submits_training(self):
        with tempfile.TemporaryDirectory() as folder:
            def remote(_api, ref):
                return "COMPLETE" if "merge-" in ref else "MISSING"
            with patch("tools.v3_training_watch.ROOT", Path(folder)), \
                 patch("tools.v3_training_watch.status", side_effect=remote), \
                 patch("tools.v3_training_watch.subprocess.run") as run:
                self.assertEqual(inspect_once(None), "awaiting_human_qc")
                run.assert_not_called()

    def test_running_training_is_observed_without_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            def remote(_api, ref):
                return "COMPLETE" if "merge-" in ref else "RUNNING"
            with patch("tools.v3_training_watch.ROOT", Path(folder)), \
                 patch("tools.v3_training_watch.status", side_effect=remote), \
                 patch("tools.v3_training_watch.subprocess.run") as run:
                self.assertEqual(inspect_once(None), "training_running")
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
