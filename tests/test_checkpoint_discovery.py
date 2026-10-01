import tempfile
from pathlib import Path
import unittest

from brainage import sha256
from v3_inputs import find_checkpoint_by_hash


class CheckpointDiscoveryTests(unittest.TestCase):
    def test_hit_checkpoint_can_be_pinned(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            hit = root / "best_hits.pt"
            hit.write_bytes(b"hit-parent")
            (root / "best.pt").write_bytes(b"mae-parent")
            self.assertEqual(find_checkpoint_by_hash(root, sha256(hit)), hit)

    def test_nested_kaggle_mount_is_selected_by_content(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = root / "notebooks/owner/parent/training/best.pt"
            parent.parent.mkdir(parents=True)
            parent.write_bytes(b"approved-parent")
            other = root / "other/training/best.pt"
            other.parent.mkdir(parents=True)
            other.write_bytes(b"another-model")
            digest = sha256(parent)
            self.assertEqual(find_checkpoint_by_hash(root, digest), parent)
            with self.assertRaises(FileNotFoundError):
                find_checkpoint_by_hash(root, "0" * 64)
            other.write_bytes(parent.read_bytes())
            with self.assertRaises(FileNotFoundError):
                find_checkpoint_by_hash(root, digest)
