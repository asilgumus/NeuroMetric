"""Poll every 15 minutes; hand verified terminal training to a research agent."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.v3_queue import ACTIVE, status


def completion_evidence(folder: Path):
    training = folder / "training"
    config = json.loads((training / "config.json").read_text())
    selection = json.loads((training / "selection.json").read_text())
    baseline = json.loads((training / "baseline_metrics.json").read_text())
    with (training / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (training / "best_validation_predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    if config["epochs_head"] != 0 or config["model"] != "sfcn" or not rows:
        raise ValueError("Autopilot requires a block-only SFCN continuation")
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("Selection must not use test data")
    best, stale = float(baseline["mae"]), 0
    if not math.isfinite(best) or best < 0:
        raise ValueError("Invalid baseline MAE")
    limit, patience = config["epochs_block"], config["patience"]
    if not isinstance(limit, int) or not isinstance(patience, int) or min(limit, patience) < 1:
        raise ValueError("Invalid epoch limit or patience")
    for epoch, row in enumerate(rows, 1):
        if stale >= patience:
            raise ValueError("History continued after early stopping")
        score = float(row["val_mae"])
        if row["stage"] != "block" or int(row["epoch"]) != epoch or not math.isfinite(score):
            raise ValueError("Invalid or nonconsecutive history")
        if score < best:
            best, stale = score, 0
        else:
            stale += 1
    if len(rows) == limit:
        reason = "epoch_limit"
    elif len(rows) < limit and stale >= config["patience"]:
        reason = "early_stopping"
    else:
        raise ValueError("Terminal completion does not match epoch limit or patience")
    if len(predictions) != config["val_n"] or len(predictions) == 0:
        raise ValueError("Validation predictions are incomplete")
    if len({r["subject_id"] for r in predictions}) != len(predictions):
        raise ValueError("Duplicate validation participants")
    errors = [abs(float(r["prediction"]) - float(r["age"])) for r in predictions]
    if not all(math.isfinite(e) for e in errors):
        raise ValueError("Non-finite predictions")
    mae = sum(errors) / len(errors)
    if abs(mae - best) > 1e-5 or abs(float(selection["validation_mae"]) - best) > 1e-5:
        raise ValueError("Best predictions, selection and history disagree")
    within = sum(e <= 1 for e in errors) / len(errors)
    return {"reason": reason, "epochs": len(rows), "max_epochs": limit,
            "stale_epochs": stale, "validation_mae": mae,
            "within_1_year": within,
            "needs_improvement": reason == "early_stopping" or within <= .70}


def should_handoff(evidence, terminal_only):
    """A finite-budget watcher still verifies conditions but never launches work."""
    return evidence["needs_improvement"] and not terminal_only


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="asildoangm/brainage-v3-train-sfcn-refine")
    parser.add_argument("--check-now", action="store_true",
                        help="Observe immediately on explicitly requested recovery")
    parser.add_argument("--terminal-only", action="store_true",
                        help="Verify terminal conditions without launching another research agent")
    args = parser.parse_args()
    if not args.ref.startswith("asildoangm/brainage-v3-train-sfcn-") or not all(
            c.isalnum() or c in "/-" for c in args.ref):
        parser.error("Only project SFCN continuation jobs are supported")
    work = ROOT / "artifacts/v3/autopilot" / args.ref.split("/")[-1]
    work.mkdir(parents=True, exist_ok=True)
    with (work / "watch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Exact user request: first observation after 15 min, then every 15 min.
        save(work / "status.json", {"state": "sleeping", "pid": os.getpid(),
                                   "next_check_seconds": 900})
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        first_check = True
        while True:
            if not (first_check and args.check_now):
                time.sleep(900)
            first_check = False
            try:
                remote = status(api, args.ref)
                state = {"checked_at": datetime.now(timezone.utc).isoformat(),
                         "remote": remote, "ref": args.ref}
                print(json.dumps(state), flush=True)
                with (work / "checks.jsonl").open("a") as log:
                    log.write(json.dumps(state) + "\n")
                save(work / "status.json", state)
                if remote in ACTIVE:
                    continue
                if remote == "MISSING":
                    raise RuntimeError("Exact job status unavailable; retry without replacement")
                if remote != "COMPLETE":
                    save(work / "status.json", {**state, "state": "remote_failure",
                         "reason": "Not early stopping; no automatic replacement"})
                    return
                subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output",
                                args.ref, "-p", str(work / "outputs"), "--page-size", "200",
                                "--file-pattern", r".*(best\.pt|best_hits\.pt|history\.csv|selection\.json|hit_selection\.json|config\.json|baseline_metrics\.json|baseline_validation_predictions\.csv|best_validation_predictions\.csv|best_hits_validation_predictions\.csv)$",
                                "--quiet"], check=True, timeout=600)
                evidence = completion_evidence(work / "outputs")
                save(work / "completion.json", evidence)
                save(work / "status.json", {**state, **evidence, "state": "terminal_verified"})
                if not evidence["needs_improvement"]:
                    return
                if not should_handoff(evidence, args.terminal_only):
                    save(work / "status.json", {**state, **evidence,
                         "state": "terminal_verified_budget_exhausted",
                         "reason_for_no_handoff": "Finite one-successor experiment budget"})
                    return
                marker = work / "agent_started.json"
                if marker.exists():
                    print("Research agent already launched; refusing duplicate", flush=True)
                    return
                prompt = (ROOT / "tools/v3_autopilot_task.md").read_text()
                prompt += f"\nCompleted ref: {args.ref}\nEvidence: {work / 'completion.json'}\nOutputs: {work / 'outputs'}\n"
                command = [shutil.which("codex") or "/home/asil/.local/bin/codex",
                           "exec", "--skip-git-repo-check", "-C", str(ROOT),
                           "-s", "danger-full-access", "-c", 'approval_policy="never"',
                           "-c", 'web_search="live"', "--json", "-o", str(work / "agent_summary.md"), "-"]
                with (work / "agent.jsonl").open("a") as log:
                    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log,
                                             stderr=subprocess.STDOUT, cwd=ROOT, text=True)
                    save(marker, {"pid": child.pid, "started_at": datetime.now(timezone.utc).isoformat()})
                    save(work / "status.json", {**state, **evidence, "state": "research_agent_running",
                                               "agent_pid": child.pid})
                    child.communicate(prompt)
                save(work / "agent_exit.json", {"returncode": child.returncode})
                save(work / "status.json", {**state, **evidence,
                     "state": "research_agent_finished" if child.returncode == 0 else "research_agent_failed",
                     "returncode": child.returncode})
                return
            except Exception as exc:
                print(f"Observation failure (will retry same job in 15 min): {exc}", flush=True)
                save(work / "status.json", {"state": "observation_error", "error": str(exc),
                                           "checked_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
