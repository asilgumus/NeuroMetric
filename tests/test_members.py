import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from brainage import MEMBER_SHA256, bootstrap


class MemberSelectionTests(unittest.TestCase):
    def test_all_member_numbers_survive_archive_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            for name in ['model/model.py', 'model/modules.py', 'transforms/transforms.py',
                         'transforms/load_transform.py', 'utils/misc.py', 'LICENSE']:
                p = source / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('x')
            archive = root / 'CNN1.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for member in range(5):
                    z.writestr(f'ResNet3D_3x_{member}.pth', bytes([member]))
            def fake_download(url, path):
                path = Path(path)
                if path.name == 'CNN1.zip':
                    return archive
                relative = path.relative_to(root / 'cache' / 'upstream')
                return source / relative
            hashes = {m: __import__('hashlib').sha256(bytes([m])).hexdigest() for m in range(5)}
            with patch('brainage.download', side_effect=fake_download), patch.dict(MEMBER_SHA256, hashes, clear=True):
                for member in range(5):
                    selected, provenance = bootstrap(root / 'cache', member=member)
                    self.assertEqual(selected.name, f'ResNet3D_3x_{member}.pth')
                    self.assertEqual(provenance['ensemble_member'], selected.name)


if __name__ == '__main__':
    unittest.main()
