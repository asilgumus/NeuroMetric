"""Compact read-only status of BrainAGE V3 Kaggle preparation jobs."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

from kaggle.api.kaggle_api_extended import KaggleApi


ROOT = Path(__file__).resolve().parents[1]
PROGRESS = re.compile(r"V3_PREPARE\s+(\w+)\s+(\d+)\s+/\s+(\d+)\s+(\S+)\s+(\S+)")


def main():
    api = KaggleApi()
    api.authenticate()
    for metadata in sorted((ROOT / "kaggle").glob("v3_prepare_*/kernel-metadata.json")):
        ref = json.loads(metadata.read_text())["id"]
        try:
            status = api.kernels_status(ref).status
            # Kaggle's snapshot log endpoint is empty while a session is
            # running; the streaming CLI does expose completed progress lines.
            stream = subprocess.run(["timeout", "6s", "kaggle", "kernels", "logs", "-f", ref],
                                    capture_output=True, text=True, check=False)
            logs = stream.stdout + stream.stderr
            matches = PROGRESS.findall(logs)
            progress = f"{matches[-1][1]}/{matches[-1][2]}" if matches else "–"
            failed = sum(match[4] == "failed" for match in matches)
            print(f"{ref:58} {str(status):13} {progress:9} failed={failed}")
        except Exception as exc:
            print(f"{ref:58} unavailable: {exc.__class__.__name__}")


if __name__ == "__main__":
    main()
