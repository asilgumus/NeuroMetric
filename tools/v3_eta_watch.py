"""Wait 25 minutes, inspect actual training, repeat using measured epoch pace."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.v3_queue import ACTIVE, status

REF = "asildoangm/brainage-v3-train-sfcn-agebalance-cont2"
OUTPUT = ROOT / "artifacts/v3/eta_watch_agebalance_cont2"


def latest_epoch():
    command = [sys.executable, "-m", "kaggle", "kernels", "logs", "-f", REF]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
    try:
        stdout, _ = child.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        stdout, _ = child.communicate()
    records = []
    for line in stdout.splitlines():
        if "V3_EPOCH " in line:
            try:
                record = ast.literal_eval(line.split("V3_EPOCH ", 1)[1])
                if isinstance(record, dict) and record.get("stage") == "block":
                    records.append(record)
            except (ValueError, SyntaxError):
                pass
    return records[-1] if records else None


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous, previous_time = latest_epoch(), time.monotonic()
    delay = 1500
    while True:
        state = {"state": "sleeping", "ref": REF, "sleep_seconds": delay,
                 "at": datetime.now(timezone.utc).isoformat()}
        (OUTPUT / "status.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
        time.sleep(delay)
        try:
            remote = status(api, REF)
            current = latest_epoch() if remote in ACTIVE or remote == "COMPLETE" else None
            now = time.monotonic()
            record = {"checked_at": datetime.now(timezone.utc).isoformat(),
                      "ref": REF, "remote": remote, "latest_epoch": current}
            if remote == "COMPLETE":
                record["state"] = "complete"
                (OUTPUT / "status.json").write_text(json.dumps(record, indent=2))
                with (OUTPUT / "checks.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                return
            if remote not in ACTIVE and remote != "MISSING":
                record["state"] = "remote_failure_not_early_stopping"
                (OUTPUT / "status.json").write_text(json.dumps(record, indent=2))
                print(json.dumps(record), flush=True)
                return
            delay = 900
            if current and previous and current["epoch"] > previous["epoch"]:
                pace = (now - previous_time) / (current["epoch"] - previous["epoch"])
                remaining = max(0, 40 - current["epoch"])
                # Wake no later than 25 minutes to also notice early stopping.
                delay = max(180, min(1500, int(remaining * pace)))
                record.update(seconds_per_epoch=pace, remaining_epochs=remaining)
            if current:
                previous, previous_time = current, now
            record["next_sleep_seconds"] = delay
            with (OUTPUT / "checks.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        except Exception as exc:
            print(f"Observation error; retry same job: {exc}", flush=True)
            delay = 180


if __name__ == "__main__":
    main()
