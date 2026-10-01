"""Generate a private, checksum-pinned SFCN evaluation-only Kaggle job."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASH = "6646617b6ac64877b03fbdc94f42395ef22974240237121e0d9dd9e3d53abc40"


def main():
    target = ROOT / "kaggle/v3_test_sfcn_contrast_cont4"
    target.mkdir(parents=True, exist_ok=True)
    names = ("brainage.py", "v3_model.py", "v3_train.py", "v3_inputs.py", "v3_test_sfcn.py")
    files = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in names}
    files["catalog.csv"] = base64.b64encode((ROOT / "artifacts/v3/catalog/catalog.csv").read_bytes()).decode()
    files["qc_approval.json"] = base64.b64encode((ROOT / "artifacts/v3/qc/approval.json").read_bytes()).decode()
    launcher = f'''import base64, os, subprocess, sys
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
subprocess.run([sys.executable,"-m","pip","install","-q","numpy==1.26.4","pandas==2.2.3","scipy==1.14.1","scikit-learn==1.5.2","nibabel==5.3.2","nipype==1.10.0","xlrd==2.0.1","requests>=2.32,<3"],check=True)
root = Path("/kaggle/working")
for name, data in {files!r}.items():
    (root/name).write_bytes(base64.b64decode(data))
sys.path.insert(0,str(root))
from v3_inputs import discover_cohorts, verify_qc_approval, find_checkpoint_by_hash
inputs = Path("/kaggle/input")
cohorts = discover_cohorts(inputs,("IXI","SALD","NIMH","DLBS"),root/"catalog.csv")
verify_qc_approval(cohorts,root/"qc_approval.json",root/"catalog.csv")
prepared = Path("/tmp/brainage-heldout-prepared")
prepared.mkdir(exist_ok=True)
for source, path in cohorts.items():
    (prepared/source.lower()).symlink_to(path)
checkpoint = find_checkpoint_by_hash(inputs,{HASH!r})
subprocess.run([sys.executable,str(root/"v3_test_sfcn.py"),"--catalog",str(root/"catalog.csv"),"--prepared-root",str(prepared),"--checkpoint",str(checkpoint),"--sha256",{HASH!r},"--output",str(root/"evaluation")],check=True)
'''
    (target / "run.py").write_text(launcher)
    metadata = {"id": "asildoangm/brainage-sfcn-contrast-cont4-locked-held-out-test",
                "title": "BrainAGE SFCN contrast-cont4 locked held-out test",
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "model_sources": [],
                "kernel_sources": ["asildoangm/brainage-v3-merge-" + s for s in ("ixi","sald","nimh","dlbs")]
                                  + ["asildoangm/brainage-v3-train-sfcn-contrast-cont4"]}
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(target)


if __name__ == "__main__":
    main()
