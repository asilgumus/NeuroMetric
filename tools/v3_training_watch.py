"""Monitor authorized V3 training; launch only after exact-cohort QC approval."""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import write_json
from tools.v3_queue import ACTIVE, status
from v3_inputs import verify_qc_approval


def inspect_exact(api, ref, output_tag):
    """Observe one submitted job; missing status never authorizes submission."""
    output = ROOT / ("artifacts/v3/training_watch" + output_tag)
    output.mkdir(parents=True, exist_ok=True)
    remote = status(api, ref)
    readiness = ("training_running" if remote in ACTIVE else
                 "training_complete" if remote == "COMPLETE" else
                 "observation_missing" if remote == "MISSING" else "training_failed")
    state = {"checked_at": datetime.now(timezone.utc).isoformat(),
             "ref": ref, "remote": remote, "readiness": readiness}
    if remote == "COMPLETE":
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", ref,
                        "-p", str(output / "outputs"), "-q", "--page-size", "200"],
                       check=True, timeout=600)
    write_json(output / "status.json", state)
    with (output / "checks.jsonl").open("a") as stream:
        stream.write(json.dumps(state) + "\n")
    print(json.dumps(state), flush=True)
    return readiness


def inspect_once(api, resnet_ref="asildoangm/brainage-v3-train-resnet", output_tag="",
                 sfcn_ref="asildoangm/brainage-v3-train-sfcn"):
    output = ROOT / ("artifacts/v3/training_watch" + output_tag)
    output.mkdir(parents=True, exist_ok=True)
    sources = ("IXI", "SALD", "NIMH")
    merges = {source: status(api, f"asildoangm/brainage-v3-merge-{source.lower()}")
              for source in sources}
    refs = {"sfcn": sfcn_ref, "resnet": resnet_ref}
    models = {name: status(api, ref) for name, ref in refs.items()}
    baseline_ref = "asildoangm/brainage-v3-provisional-pretrained-validation"
    baseline_status = status(api, baseline_ref)
    state = {"checked_at": datetime.now(timezone.utc).isoformat(),
             "merges": merges, "training": models, "baseline": baseline_status,
             "readiness": "pending"}
    baseline_output = ROOT / "artifacts/v3/pretrained_validation"
    if baseline_status == "COMPLETE" and not (baseline_output / "baseline/metrics.json").exists():
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", baseline_ref,
                        "-p", str(baseline_output), "--file-pattern",
                        r".*(metrics\.json|validation_predictions\.csv|\.log)$",
                        "--page-size", "200", "-q"], check=True, timeout=180)
    elif baseline_status not in ACTIVE | {"COMPLETE", "MISSING"}:
        state["baseline_error"] = "Inspect the provisional inference run; it was not restarted"
    # Preserve authoritative failures for diagnosis, never overwrite a failed run.
    failed = {name: value for name, value in models.items()
              if value not in ACTIVE | {"COMPLETE", "MISSING"}}
    if failed:
        for name in failed:
            subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output",
                            refs[name], "-p",
                            str(output / f"error_{name}"), "--file-pattern",
                            r".*(\.log|history\.csv|baseline_metrics\.json)$", "-q",
                            "--page-size", "200"],
                           check=False, timeout=180)
        state["readiness"] = "training_failed"
    elif all(value == "COMPLETE" for value in models.values()):
        state["readiness"] = "training_complete"
        for name in models:
            subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output",
                            refs[name], "-p",
                            str(ROOT / (f"artifacts/v3/trained_{name}" +
                                        (output_tag if refs[name] != f"asildoangm/brainage-v3-train-{name}" else ""))), "-q",
                            "--page-size", "200"],
                           check=True, timeout=600)
    elif any(value in ACTIVE for value in models.values()) and all(
            value != "MISSING" for value in models.values()):
        state["readiness"] = "training_running"
    elif all(value == "COMPLETE" for value in merges.values()):
        approval = ROOT / "artifacts/v3/qc/approval.json"
        cohorts = {source: ROOT / f"artifacts/v3/qc_review_{source.lower()}_current"
                   / "prepared" / source.lower() for source in sources}
        try:
            verify_qc_approval(cohorts, approval, ROOT / "artifacts/v3/catalog/catalog.csv")
        except (FileNotFoundError, ValueError, KeyError) as exc:
            state["readiness"] = "awaiting_human_qc"
            state["reason"] = str(exc)
        else:
            for name, value in models.items():
                if value != "MISSING":
                    continue
                if refs[name] != f"asildoangm/brainage-v3-train-{name}":
                    raise ValueError("Configured run missing; do not replace it with a default run")
                subprocess.run([sys.executable, str(ROOT / "tools/build_v3_train_kaggle.py"),
                                name], check=True, cwd=ROOT, timeout=60)
                subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p",
                                str(ROOT / f"kaggle/v3_train_{name}")],
                               check=True, cwd=ROOT, timeout=180)
            state["readiness"] = "training_submitted"
    write_json(output / "status.json", state)
    with (output / "checks.jsonl").open("a") as stream:
        stream.write(json.dumps(state) + "\n")
    print(json.dumps(state), flush=True)
    return state["readiness"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=2.)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--resnet-ref", default="asildoangm/brainage-v3-train-resnet")
    parser.add_argument("--sfcn-ref", default="asildoangm/brainage-v3-train-sfcn")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--exact-ref", help="Observe only this ref without automatic submission")
    args = parser.parse_args()
    if args.output_tag and (not args.output_tag.startswith("_") or
                           not args.output_tag[1:].replace("_", "").isalnum()):
        parser.error("Output tag must be an underscore-prefixed identifier")
    if args.hours <= 0:
        parser.error("hours must be positive")
    if args.exact_ref and not args.output_tag:
        parser.error("Exact-ref monitoring requires a separate output tag")
    output = ROOT / ("artifacts/v3/training_watch" + args.output_tag)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "watch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    deadline = time.monotonic() + args.hours * 3600
    intervals = (180, 480, 600)
    success = 0
    if args.exact_ref and not args.once:
        time.sleep(intervals[0])
        success = 1
    while time.monotonic() < deadline:
        try:
            readiness = (inspect_exact(api, args.exact_ref, args.output_tag) if args.exact_ref else
                         inspect_once(api, args.resnet_ref, args.output_tag, args.sfcn_ref))
            if readiness in {"training_complete", "training_failed"} or args.once:
                return
            delay = intervals[min(success, len(intervals) - 1)]
            success += 1
        except Exception as exc:
            print(f"V3_WATCH observation failed: {type(exc).__name__}: {exc}", flush=True)
            success = 0
            delay = intervals[0]
            if args.once:
                raise
        # Persistent monitor runs outside the interactive agent turn.
        time.sleep(min(delay, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
