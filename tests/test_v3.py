"""Fast invariants for the V3 architecture and locked data catalogue."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image

from v3_model import N_CLASSES, SFCN, expected_age, gaussian_targets, load_pretrained
from v3_train import Volumes, metric
from v3_evaluate import paired_mae_difference
from v3_merge import merge, resnet_central_occupancy
from v3_inputs import discover_cohorts, verify_qc_approval
from v3_predict import selected_components, checked_checkpoint, validate_checkpoint_set
from tools.v3_qc_sheets import select_review, save_sheets
from tools.v3_queue import KaggleTransientError, fill_once, push
from brainage import sha256
from v3_prepare import resnet_center_occupancy, valid_array


ROOT = Path(__file__).resolve().parents[1]


class V3Tests(unittest.TestCase):
    def test_queue_push_uses_importable_kaggle_cli(self):
        with patch("tools.v3_queue.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0,
                                               stdout="successfully pushed",
                                               stderr="")
            self.assertTrue(push("owner/example", ROOT / "kaggle/example"))
            command = run.call_args.args[0]
            self.assertEqual(command[1:5], ["-m", "kaggle", "kernels", "push"])
            run.return_value = SimpleNamespace(returncode=1, stdout="",
                                               stderr="429 Too Many Requests")
            with self.assertRaises(KaggleTransientError):
                push("owner/example", ROOT / "kaggle/example")

    def test_queue_refreshes_completed_merge_when_marker_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            refresh = Path(folder)
            marker = refresh / "nimh.pending"
            marker.write_text("updated QC")
            def fake_status(_api, ref):
                return ("COMPLETE" if "prepare-nimh-" in ref or
                        ref.endswith("merge-nimh") else "MISSING")
            with patch("tools.v3_queue.REFRESH", refresh), \
                 patch("tools.v3_queue.status", side_effect=fake_status), \
                 patch("tools.v3_queue.subprocess.run"), \
                 patch("tools.v3_queue.push", return_value=True) as pushed:
                fill_once(None, max_active=1)
            self.assertFalse(marker.exists())
            self.assertEqual(pushed.call_args.args[0],
                             "asildoangm/brainage-v3-merge-nimh")

    def test_sfcn_pretrained_shape_and_head(self):
        path = ROOT / ".cache/brainage/sfcn_pretrained.p"
        if not path.exists():
            self.skipTest("official SFCN weights are not cached")
        model = load_pretrained(path)
        self.assertEqual(model.classifier.conv_6.out_channels, N_CLASSES)
        self.assertEqual(tuple(model.classifier.conv_6.weight.shape), (83, 64, 1, 1, 1))
        self.assertTrue(torch.all(model.classifier.conv_6.bias[:24] == -3.))
        self.assertTrue(torch.all(model.classifier.conv_6.bias[64:] == -3.))

    def test_gaussian_target_normalization(self):
        ages = torch.tensor([18., 57.5, 100.])
        labels = gaussian_targets(ages)
        self.assertTrue(torch.allclose(labels.sum(1), torch.ones(3), atol=1e-5))
        self.assertEqual(labels.argmax(1).tolist(), [0, 39, 82])

    def test_expected_age(self):
        logits = torch.full((1, N_CLASSES), -100.)
        logits[0, 39] = 100.
        self.assertAlmostEqual(expected_age(logits).item(), 57., places=4)

    def test_prediction_uses_only_validation_locked_selection(self):
        self.assertEqual(selected_components({"candidate": "v2+sfcn",
                                              "test_used_for_selection": False}),
                         ("v2", "sfcn"))
        with self.assertRaises(ValueError):
            selected_components({"candidate": "sfcn+sfcn",
                                 "test_used_for_selection": False})
        with self.assertRaises(ValueError):
            selected_components({"candidate": "sfcn", "test_used_for_selection": True})
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model.pt"
            checkpoint.write_bytes(b"selected-weights")
            self.assertEqual(checked_checkpoint(checkpoint, sha256(checkpoint), "test"),
                             checkpoint)
            with self.assertRaises(ValueError):
                checked_checkpoint(checkpoint, "incorrect", "test")
            selection = {"checkpoints": {"sfcn": sha256(checkpoint)}}
            validate_checkpoint_set(("sfcn",), selection,
                                    SimpleNamespace(sfcn_checkpoint=checkpoint))
            with self.assertRaises(FileNotFoundError):
                validate_checkpoint_set(("sfcn",), selection,
                                        SimpleNamespace(sfcn_checkpoint=Path(folder) / "missing.pt"))

    def test_metric_is_regression_not_classification_accuracy(self):
        score = metric([30., 40., 50.], [30., 42., 49.])
        self.assertAlmostEqual(score["mae"], 1.)
        self.assertAlmostEqual(score["within_1_year"], 2 / 3)
        self.assertAlmostEqual(score["rounded_year_match"], 1 / 3)

    def test_paired_improvement_and_invalid_metrics(self):
        comparison = paired_mae_difference([30., 40., 50.], [31., 41., 51.],
                                           [33., 43., 53.], draws=200)
        self.assertAlmostEqual(comparison["delta_mae_years"], -2.)
        self.assertEqual(comparison["delta_mae_ci95"], [-2., -2.])
        with self.assertRaises(ValueError):
            metric([30., 40.], [30., float("nan")])

    def test_catalogue_is_subject_disjoint(self):
        path = ROOT / "artifacts/v3/catalog/catalog.csv"
        if not path.exists():
            self.skipTest("V3 catalogue not generated")
        frame = pd.read_csv(path)
        self.assertFalse(frame.subject_id.duplicated().any())
        self.assertTrue((frame.loc[frame.dataset == "DLBS", "split"] == "external").all())
        self.assertEqual(set(frame.loc[frame.dataset == "IXI", "split"]), {"train", "val", "test"})
        self.assertTrue(np.isfinite(frame.age).all())
        self.assertTrue(frame.age.between(18, 100).all())
        self.assertFalse(frame.site.isna().any())
        self.assertFalse(frame.license.isna().any())
        remote = frame.loc[frame.dataset != "IXI"]
        self.assertTrue(remote.source_url.str.startswith("https://").all())
        self.assertTrue(remote.source_etag.str.len().ge(32).all())

    def test_compressed_sfcn_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sfcn").mkdir()
            np.savez_compressed(root / "sfcn/example.npz",
                                x=np.ones((160, 192, 160), dtype=np.float16))
            rows = pd.DataFrame([{"dataset": "SALD", "subject_id": "SALD:example",
                                  "age": 57., "sfcn_array": "sfcn/example.npz"}])
            image, age = Volumes(rows, {"SALD": root}, "sfcn")[0]
            self.assertEqual(tuple(image.shape), (1, 160, 192, 160))
            self.assertAlmostEqual(float(image.mean()),
                                   182 * 218 * 182 / (160 * 192 * 160), places=5)
            self.assertEqual(float(age), 57.)
            self.assertTrue(valid_array(root / "sfcn/example.npz", (160, 192, 160)))
            (root / "sfcn/broken.npz").write_bytes(b"incomplete")
            self.assertFalse(valid_array(root / "sfcn/broken.npz", (160, 192, 160)))

    def test_qc_sheets_cover_age_extremes_and_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "qc").mkdir()
            (root / "qc_resnet").mkdir()
            rows = []
            for age in (20., 40., 80.):
                subject = f"TEST:{int(age)}"
                Image.new("RGB", (80, 30), "white").save(
                    root / "qc" / f"TEST_{int(age)}.png")
                Image.new("RGB", (80, 30), "black").save(
                    root / "qc_resnet" / f"TEST_{int(age)}.png")
                rows.append({"dataset": "TEST", "site": "one", "subject_id": subject,
                             "age": age, "qc_status": "automatic_checks_passed"})
            rows.append({"dataset": "TEST", "site": "one", "subject_id": "TEST:failed",
                         "age": 55., "qc_status": "failed"})
            pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
            (root / "prepare_summary.json").write_text(json.dumps({"complete": True,
                                                                      "total": 4}))
            review, failures = select_review([root], per_site=2)
            self.assertEqual(set(review.subject_id), {"TEST:20", "TEST:80"})
            self.assertEqual(failures.subject_id.tolist(), ["TEST:failed"])
            save_sheets(review, root / "review")
            self.assertTrue((root / "review/qc_sheet_001.png").exists())

    def test_materialized_source_merge_is_portable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = root / "catalog.csv"
            pd.DataFrame([{"dataset": "IXI", "subject_id": "IXI:one"}]).to_csv(catalog,
                                                                                index=False)
            shard = root / "shard"
            for subdir in ("resnet", "sfcn", "qc"):
                (shard / subdir).mkdir(parents=True)
            np.save(shard / "resnet/one.npy", np.zeros((80, 96, 80), dtype=np.float32))
            (shard / "sfcn/one.npz").write_bytes(b"sfcn-array")
            Image.new("RGB", (20, 20), "white").save(shard / "qc/IXI_one.png")
            pd.DataFrame([{"dataset": "IXI", "subject_id": "IXI:one",
                           "site": "one", "age": 35.,
                           "qc_status": "automatic_checks_passed",
                           "resnet_array": "resnet/one.npy",
                           "sfcn_array": "sfcn/one.npz"}]).to_csv(shard / "manifest.csv", index=False)
            (shard / "prepare_summary.json").write_text(json.dumps({
                "source": "IXI", "catalog_sha256": sha256(catalog), "complete": True,
                "total": 1, "shard_count": 1, "shard_index": 0}))
            merged = root / "merged"
            merge(catalog, [shard], merged, sources=("IXI",), materialize=True)
            self.assertEqual(np.load(merged / "ixi/resnet/one.npy").shape, (80, 96, 80))
            self.assertFalse((merged / "ixi/resnet/one.npy").is_symlink())
            self.assertTrue((merged / "ixi/qc/IXI_one.png").is_file())
            self.assertTrue((merged / "ixi/qc_resnet/IXI_one.png").is_file())
            review, failures = select_review([merged / "ixi"], per_site=2)
            self.assertEqual(len(review), 1)
            self.assertEqual(len(failures), 0)
            save_sheets(review, root / "review")
            self.assertTrue((root / "review/qc_sheet_001.png").is_file())

    def test_merge_excludes_empty_resnet_registration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = root / "catalog.csv"
            pd.DataFrame([{"dataset": "NIMH", "subject_id": "NIMH:bad"},
                          {"dataset": "NIMH", "subject_id": "NIMH:good"}]).to_csv(
                              catalog, index=False)
            shard = root / "shard"
            for name in ("resnet", "sfcn", "qc"):
                (shard / name).mkdir(parents=True)
            for subject, value in (("bad", -1.), ("good", 0.)):
                np.save(shard / "resnet" / f"{subject}.npy",
                        np.full((80, 96, 80), value, dtype=np.float32))
                (shard / "sfcn" / f"{subject}.npz").write_bytes(b"array")
                Image.new("RGB", (20, 20), "white").save(
                    shard / "qc" / f"NIMH_{subject}.png")
            pd.DataFrame([{"dataset": "NIMH", "subject_id": f"NIMH:{subject}",
                           "qc_status": "automatic_checks_passed",
                           "resnet_array": f"resnet/{subject}.npy",
                           "sfcn_array": f"sfcn/{subject}.npz"}
                          for subject in ("bad", "good")]).to_csv(
                              shard / "manifest.csv", index=False)
            (shard / "prepare_summary.json").write_text(json.dumps({
                "source": "NIMH", "catalog_sha256": sha256(catalog),
                "complete": True, "total": 2, "shard_count": 1,
                "shard_index": 0}))
            self.assertEqual(resnet_central_occupancy(shard / "resnet/bad.npy"), 0.)
            self.assertEqual(resnet_center_occupancy(shard / "resnet/bad.npy"), 0.)
            merge(catalog, [shard], root / "merged", sources=("NIMH",),
                  materialize=True)
            summary = json.loads((root / "merged/nimh/prepare_summary.json").read_text())
            self.assertEqual((summary["passed"], summary["failed"]), (1, 1))
            manifest = pd.read_csv(root / "merged/nimh/manifest.csv")
            bad = manifest.set_index("subject_id").loc["NIMH:bad"]
            self.assertEqual(bad.qc_status, "failed")
            self.assertIn("central anatomy absent", bad.error)
            self.assertFalse((root / "merged/nimh/resnet/bad.npy").exists())

    def test_gpu_jobs_use_portable_cohort_outputs(self):
        for job, count in (("v3_train_sfcn", 3), ("v3_train_resnet", 3),
                           ("v3_evaluate", 10)):
            metadata = json.loads((ROOT / "kaggle" / job / "kernel-metadata.json").read_text())
            self.assertEqual(len(metadata["kernel_sources"]), count)
            self.assertTrue(all("brainage-v3-prepare" not in source
                                for source in metadata["kernel_sources"]))
        for source, count in (("ixi", 8), ("sald", 8), ("nimh", 4), ("dlbs", 8)):
            metadata = json.loads((ROOT / "kaggle" / f"v3_merge_{source}" /
                                   "kernel-metadata.json").read_text())
            self.assertEqual(len(metadata["kernel_sources"]), count)

    def test_discover_cohorts_checks_catalog_and_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = root / "catalog.csv"
            catalog.write_text("dataset,subject_id\nIXI,IXI:one\n")
            cohort = root / "inputs/brainage-v3-merge-ixi/prepared/ixi"
            cohort.mkdir(parents=True)
            (cohort / "manifest.csv").write_text("subject_id\nIXI:one\n")
            summary = {"source": "IXI", "complete": True,
                       "catalog_sha256": sha256(catalog)}
            path = cohort / "prepare_summary.json"
            path.write_text(json.dumps(summary))
            self.assertEqual(discover_cohorts(root / "inputs", ("IXI",), catalog),
                             {"IXI": cohort})
            qc = cohort.parent.parent / "qc_review"
            qc.mkdir()
            (qc / "qc_review_index.csv").write_text(
                "subject_id,qc_image,resnet_qc_image\nIXI:one,sfcn.png,resnet.png\n")
            (qc / "qc_automatic_failures.csv").write_text("subject_id\n")
            Image.new("RGB", (20, 20), "white").save(qc / "qc_sheet_001.png")
            approval = root / "approval.json"
            approval.write_text(json.dumps({
                "catalog_sha256": sha256(catalog), "sources": {"IXI": {
                    "decision": "approved", "reviewer": "test-reviewer",
                    "qc_protocol": "sfcn+resnet-v1",
                    "summary_sha256": sha256(path),
                    "manifest_sha256": sha256(cohort / "manifest.csv"),
                    "review_index_sha256": sha256(qc / "qc_review_index.csv"),
                    "failures_sha256": sha256(qc / "qc_automatic_failures.csv"),
                    "sheets": {"qc_sheet_001.png": sha256(qc / "qc_sheet_001.png")}}}}))
            verify_qc_approval({"IXI": cohort}, approval, catalog)
            Image.new("RGB", (20, 20), "black").save(qc / "qc_sheet_001.png")
            with self.assertRaises(ValueError):
                verify_qc_approval({"IXI": cohort}, approval, catalog)
            summary["catalog_sha256"] = "wrong"
            path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                discover_cohorts(root / "inputs", ("IXI",), catalog)


if __name__ == "__main__":
    unittest.main()
