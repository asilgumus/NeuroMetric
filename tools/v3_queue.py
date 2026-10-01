"""Fill Kaggle's five CPU slots with V3 preparation and cohort merge jobs.

Run with KAGGLE_CONFIG_DIR set. A failed job stops the queue for inspection;
it is never silently replaced or retried. Completed merge outputs remain on
Kaggle for manual QC and the later validation-locked GPU jobs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from requests import ConnectionError as RequestsConnectionError
from requests import HTTPError, Timeout as RequestsTimeout


ROOT = Path(__file__).resolve().parents[1]
SHARDS = {"IXI": 8, "SALD": 8, "NIMH": 4, "DLBS": 8}
REFRESH = ROOT / "artifacts/v3/queue_refresh"
ACTIVE = {"RUNNING", "QUEUED", "PENDING", "STARTING"}
COMPLETE = {"COMPLETE"}


class KaggleTransientError(RuntimeError):
    """The remote API or network can recover without changing the job plan."""


def plan():
    # Round-robin source cohorts so the independent DLBS evaluation data do
    # not wait behind every IXI/SALD shard when a CPU slot opens.
    for index in range(max(SHARDS.values())):
        for source, count in SHARDS.items():
            if index >= count:
                continue
            ref = f"asildoangm/brainage-v3-prepare-{source.lower()}-{index}-of-{count}"
            folder = ROOT / "kaggle" / f"v3_prepare_{source.lower()}_{index}_of_{count}"
            yield source, index, count, ref, folder


def merge_plan():
    for source in SHARDS:
        ref = f"asildoangm/brainage-v3-merge-{source.lower()}"
        folder = ROOT / "kaggle" / f"v3_merge_{source.lower()}"
        yield source, ref, folder


def status(api, ref: str) -> str:
    try:
        result = api.kernels_status(ref)
    except HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return "MISSING"
        raise
    except ValueError as exc:
        # Kaggle currently reports some never-created private slugs as a
        # wrapped 403, rather than the 404 returned for other missing slugs.
        if "Permission 'kernels.get' was denied" in str(exc):
            return "MISSING"
        raise
    return getattr(result.status, "name", str(result.status).split(".")[-1])


def push(ref: str, folder: Path) -> bool:
    # User services do not necessarily inherit ~/.local/bin in PATH. Invoke
    # the CLI through the interpreter that already imported kaggle above.
    result = subprocess.run([sys.executable, "-m", "kaggle", "kernels",
                             "push", "-p", str(folder)],
                            cwd=ROOT, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if "Maximum batch CPU session count of 5 reached" in output:
        print(f"V3_QUEUE Kaggle slot not released yet; retry later: {ref}", flush=True)
        return False
    if result.returncode and any(marker in output for marker in
                                 ("429", "Too Many Requests", "NameResolutionError",
                                  "ConnectionError", "Max retries exceeded",
                                  "Read timed out")):
        raise KaggleTransientError(f"Kaggle push temporarily unavailable: {output[-500:]}")
    if result.returncode or "successfully pushed" not in output:
        raise RuntimeError(f"Kaggle push failed for {ref}: {output[-1000:]}")
    return True


def fill_once(api, max_active: int) -> tuple[int, int, int]:
    items = [(item, status(api, item[3])) for item in plan()]
    merges = [(item, status(api, item[1])) for item in merge_plan()]
    failures = [(item[3], state) for item, state in items
                if state not in ACTIVE | COMPLETE | {"MISSING"}]
    failures.extend((item[1], state) for item, state in merges
                    if state not in ACTIVE | COMPLETE | {"MISSING"})
    if failures:
        raise RuntimeError(f"Inspect failed Kaggle V3 jobs: {failures}")
    running = sum(state in ACTIVE for _, state in items + merges)
    finished = sum(state in COMPLETE for _, state in items + merges)
    total = len(items) + len(merges)
    print(f"V3_QUEUE complete={finished}/{total} active={running}/{max_active}", flush=True)
    # A complete cohort can be materialized immediately, leaving its QC
    # sheets ready for inspection while the other sources are still preparing.
    for (source, ref, folder), state in merges:
        if running >= max_active:
            break
        refresh = REFRESH / f"{source.lower()}.pending"
        if state != "MISSING" and not (state == "COMPLETE" and refresh.is_file()):
            continue
        if not all(prep_state in COMPLETE for (prep, prep_state) in items
                   if prep[0] == source):
            continue
        subprocess.run([sys.executable, str(ROOT / "tools/build_v3_merge_kaggle.py"),
                        source], cwd=ROOT, check=True)
        if not push(ref, folder):
            return finished, running, total
        refresh.unlink(missing_ok=True)
        running += 1
        print(f"V3_QUEUE launched {ref}; active={running}/{max_active}", flush=True)
    for (source, index, count, ref, folder), state in items:
        if running >= max_active:
            break
        if state != "MISSING":
            continue
        subprocess.run([sys.executable, str(ROOT / "tools/build_v3_kaggle.py"), source,
                        "--shard-index", str(index), "--shard-count", str(count),
                        "--workers", "2"], cwd=ROOT, check=True)
        if not push(ref, folder):
            return finished, running, total
        running += 1
        print(f"V3_QUEUE launched {ref}; active={running}/{max_active}", flush=True)
    return finished, running, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="fill free slots and exit")
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--max-active", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.max_active <= 5 or args.poll_seconds < 30:
        parser.error("max-active must be 1..5 and poll-seconds at least 30")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    while True:
        try:
            finished, _, total = fill_once(api, args.max_active)
        except HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429:
                raise
            header = exc.response.headers.get("Retry-After", "")
            retry_after = int(header) if header.isdigit() else 900
            delay = max(args.poll_seconds, retry_after)
            print(f"V3_QUEUE Kaggle rate limit; retrying in {delay}s", flush=True)
            time.sleep(delay)
            continue
        except (RequestsConnectionError, RequestsTimeout) as exc:
            delay = max(args.poll_seconds, 300)
            print(f"V3_QUEUE network unavailable ({type(exc).__name__}); "
                  f"retrying in {delay}s", flush=True)
            time.sleep(delay)
            continue
        except KaggleTransientError as exc:
            delay = max(args.poll_seconds, 300)
            print(f"V3_QUEUE {exc}; retrying in {delay}s", flush=True)
            time.sleep(delay)
            continue
        if args.once or (finished == total and not list(REFRESH.glob("*.pending"))):
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("V3_QUEUE stopped by operator", flush=True)
