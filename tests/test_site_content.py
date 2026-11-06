"""Check evidence-backed copy and unchanged design; no patient inference."""
from html.parser import HTMLParser
from pathlib import Path
import csv
import unittest

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "backups/site-before-content-k7jdzLey"
ARTIFACTS = ROOT / "artifacts"


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids, self.links = [], []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])


class SiteContentTest(unittest.TestCase):
    @unittest.skipUnless(BACKUP.is_dir(), "backups/ snapshot is kept locally, not in the repo")
    def test_original_backup_and_design_preserved(self):
        for file in ("styles.css", "app.js", "logo.png"):
            self.assertEqual((ROOT / file).read_bytes(), (BACKUP / file).read_bytes())
        for file in ("index.html", "report.html", "resources.html"):
            self.assertTrue((BACKUP / file).is_file())

    def test_local_links_and_unique_ids(self):
        for file in ("index.html", "report.html", "resources.html"):
            page = Page((ROOT / file).read_text())
            self.assertEqual(len(page.ids), len(set(page.ids)))
            for link in page.links:
                if ":" in link or link.startswith("//"):
                    continue
                path, _, fragment = link.partition("#")
                path = path.split("?")[0] or file
                target = ROOT / path
                self.assertTrue(target.exists(), link)
                if fragment:
                    self.assertIn(fragment, Page(target.read_text()).ids)

    def test_demo_and_pending_features_are_explicit(self):
        report = (ROOT / "report.html").read_text()
        self.assertIn("Invented visits for interface demonstration", report)
        self.assertIn("Longitudinal performance and reference trajectories are not validated", report)
        self.assertIn("Static demo data — the site runs no real analysis", report)
        self.assertIn("no diagnostic inference", report)
        self.assertNotIn("95% CI", report)
        self.assertNotIn("Consider further", report)
        index = (ROOT / "index.html").read_text()
        self.assertIn("they are not study results", index)
        self.assertIn("no inference runs", index)

    @unittest.skipUnless(ARTIFACTS.is_dir(), "artifacts/ training outputs are kept locally, not in the repo")
    def test_validation_figures_match_saved_predictions(self):
        source = ROOT / "artifacts/v3/autopilot/brainage-v3-train-sfcn-deeper-cont3/outputs/training/best_validation_predictions.csv"
        with source.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 218)
        self.assertEqual(sum(abs(float(r["age"])-float(r["prediction"])) <= 1 for r in rows), 42)
        self.assertEqual(sum(round(float(r["age"])) == round(float(r["prediction"])) for r in rows), 28)
        mae = sum(abs(float(r["age"])-float(r["prediction"])) for r in rows) / len(rows)
        self.assertAlmostEqual(mae, 3.4460451668555585)


if __name__ == "__main__":
    unittest.main()
