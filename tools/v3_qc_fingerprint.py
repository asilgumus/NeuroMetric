"""Print a pending-review fingerprint for a downloaded V3 cohort output.

This never approves QC. Examine every listed sheet and the failure CSV first;
only then may a reviewer record an explicit approval in approval.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256
from v3_inputs import qc_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path,
                        help="downloaded merged output's prepared/<source> directory")
    parser.add_argument("--catalog", type=Path,
                        default=ROOT / "artifacts/v3/catalog/catalog.csv")
    args = parser.parse_args()
    summary = json.loads((args.cohort / "prepare_summary.json").read_text())
    if summary.get("catalog_sha256") != sha256(args.catalog):
        raise ValueError("Cohort and locked catalogue hashes differ")
    print(json.dumps({"catalog_sha256": sha256(args.catalog),
                      "source": summary["source"],
                      "review_record": {"decision": "pending_review", "reviewer": "",
                                        **qc_fingerprint(args.cohort)}}, indent=2))


if __name__ == "__main__":
    main()
