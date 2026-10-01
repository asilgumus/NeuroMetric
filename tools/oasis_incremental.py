"""Process complete downloads without waiting for the full fixed pilot cohort."""
import json
from pathlib import Path
import subprocess
import sys
import time
import fcntl

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oasis_test import image_index, matched_manifest
from brainage import write_json


def main():
    output = ROOT / "artifacts/alzheimer/incremental"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempted = set()
        if (output / "failures.json").exists():
            attempted.update(row["subject_id"] for row in
                             json.loads((output / "failures.json").read_text()))
        while True:
            clinical = pd.read_csv(ROOT / "data/oasis1/pilot_clinical.csv")
            # Fix matching using all selected participants, not the arriving subset.
            cohort, _ = matched_manifest(clinical, dict.fromkeys(clinical.ID, "pending"))
            images = image_index(ROOT / "data/oasis1/raw")
            images = {sid: p for sid, p in images.items()
                      if p.suffix == ".img" and p.with_suffix(".hdr").is_file()}
            cohort["source_path"] = cohort.subject_id.map(images)
            ready = cohort.dropna(subset=["source_path"])
            closure = output / "download_closed.json"
            download_closed = closure.exists() and json.loads(closure.read_text()).get("download_closed", False)
            if download_closed:
                frozen_ids = set(pd.read_csv(output / "manifest.csv").subject_id) | attempted
                ready = ready[ready.subject_id.isin(frozen_ids)]
            target_ids = set(ready.subject_id) if download_closed else set(cohort.subject_id)
            done = set()
            results = output / "predictions.csv"
            if results.exists() and results.stat().st_size > 1:
                done = set(pd.read_csv(results).subject_id)
            write_json(output / "worker_status.json",
                       {"complete_downloads": len(ready), "predicted": len(done),
                        "target": len(target_ids), "state": "running"})
            new = set(ready.subject_id) - done - attempted
            if new:
                manifest = output / "manifest.csv"
                ready[ready.subject_id.isin(done | new)].to_csv(manifest, index=False)
                command = [sys.executable, "oasis_test.py", "run", "--manifest", str(manifest),
                           "--output", str(output), "--device", "cpu",
                           "--allow-exploratory-overlap"]
                if results.exists() and (output / "report.json").exists():
                    command.append("--resume")
                print(f"Processing {len(new)} new MRI(s)", flush=True)
                result = subprocess.run(command, cwd=ROOT)
                if result.returncode:
                    write_json(output / "worker_status.json", {"state": "failed", "returncode": result.returncode})
                    raise SystemExit(result.returncode)
                attempted.update(new)
                continue
            if target_ids <= done | attempted:
                write_json(output / "worker_status.json", {"state": "finished", "predicted": len(done),
                           "failed": sorted(attempted - done)})
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
