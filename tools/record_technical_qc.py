"""Record explicit technical image review, never clinician or diagnostic approval."""
import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--note", required=True)
    args = parser.parse_args()
    status_path = args.folder / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("state") != "prepared" or not args.evidence.is_file():
        raise ValueError("Prepared input and inspected evidence required")
    status.update(visual_qc="approved", qc_scope="technical_registration_only", reviewer="coding_assistant_visual_review",
                  evidence_sha256=sha256(args.evidence), input_sha256=sha256(args.folder / "sfcn.npy"),
                  review_note=args.note, clinical_validation=False)
    write_json(status_path, status)
