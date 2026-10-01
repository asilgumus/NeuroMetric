"""Build provisional pretrained validation inference without fine-tuning."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "kaggle/v3_baseline"
    target.mkdir(parents=True, exist_ok=True)
    files = {name: base64.b64encode((ROOT / name).read_bytes()).decode("ascii")
             for name in ("brainage.py", "v3_train.py", "v3_model.py", "v3_inputs.py",
                          "v3_baseline.py")}
    catalog = base64.b64encode((ROOT / "artifacts/v3/catalog/catalog.csv").read_bytes()).decode()
    code = f'''# Generated provisional zero-shot validation job.
import base64
import os
from pathlib import Path
import subprocess
import sys
import requests
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4",
                "pandas==2.2.3", "scipy==1.14.1", "scikit-learn==1.5.2",
                "nibabel==5.3.2", "nipype==1.10.0", "xlrd==2.0.1"], check=True)
root = Path("/kaggle/working")
for name, data in {files!r}.items():
    (root / name).write_bytes(base64.b64decode(data))
(root / "catalog.csv").write_bytes(base64.b64decode({catalog!r}))
sys.path.insert(0, str(root))
from v3_inputs import discover_cohorts
from brainage import sha256
from v3_baseline import SFCN_SHA256
cohorts = discover_cohorts(Path("/kaggle/input"), ("IXI", "SALD", "NIMH"), root / "catalog.csv")
prepared = Path("/tmp/zero-shot-prepared")
prepared.mkdir(parents=True, exist_ok=True)
for source, path in cohorts.items():
    (prepared / source.lower()).symlink_to(path)
weights = root / "sfcn_pretrained.p"
url = "https://raw.githubusercontent.com/ha-ha-ha-han/UKBiobank_deep_pretrain/master/brain_age/run_20190719_00_epoch_best_mae.p"
response = requests.get(url, timeout=(30, 180))
response.raise_for_status()
weights.write_bytes(response.content)
if sha256(weights) != SFCN_SHA256:
    raise ValueError("SFCN checkpoint checksum mismatch")
subprocess.run([sys.executable, str(root / "v3_baseline.py"), "--catalog", str(root / "catalog.csv"),
                "--prepared-root", str(prepared), "--sfcn-weights", str(weights),
                "--output", str(root / "baseline")], check=True)
'''
    (target / "run.py").write_text(code)
    metadata = {"id": "asildoangm/brainage-v3-provisional-pretrained-validation",
                "title": "BrainAGE V3 Provisional Pretrained Validation", "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "model_sources": [],
                "kernel_sources": [f"asildoangm/brainage-v3-merge-{s}" for s in ("ixi", "sald", "nimh")]}
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(target)


if __name__ == "__main__":
    main()
