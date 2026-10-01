"""Authorized download -> matched cohort -> locked-model exploratory test."""
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-dua", action="store_true")
    parser.add_argument("--cases", type=int)
    args = parser.parse_args()
    if not args.accept_dua:
        parser.error("Explicit OASIS DUA acceptance required")
    output = ROOT / "artifacts/alzheimer"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        steps = [
            ("download", ["tools/fetch_oasis1.py", "--accept-dua"] +
             (["--cases", str(args.cases)] if args.cases else [])),
            ("match", ["oasis_test.py", "manifest", "--clinical",
                       "data/oasis1/pilot_clinical.csv" if args.cases else
                       "data/oasis1/oasis_cross-sectional.csv", "--raw-root", "data/oasis1/raw"]),
            ("inference", ["oasis_test.py", "run", "--manifest",
                           "artifacts/alzheimer/cohort/manifest.csv", "--allow-exploratory-overlap"]),
        ]
        for stage, command in steps:
            write_json(output / "pipeline_status.json", {"stage": stage, "state": "running",
                       "at": datetime.now(timezone.utc).isoformat()})
            result = subprocess.run([sys.executable, *command], cwd=ROOT)
            if result.returncode:
                write_json(output / "pipeline_status.json", {"stage": stage, "state": "failed",
                           "returncode": result.returncode,
                           "at": datetime.now(timezone.utc).isoformat()})
                raise SystemExit(result.returncode)
        write_json(output / "pipeline_status.json", {"state": "inference_finished",
                   "interpretation": "Check report, failures and review registration QC before conclusions",
                   "at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
