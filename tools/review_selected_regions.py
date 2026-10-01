"""Record hash-bound technical review separately from the live segmentation queue."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("ixi170", "ixi361", "ixi566", "dlbs3898"))
    parser.add_argument("--note", required=True)
    args = parser.parse_args()
    folder = ROOT / "artifacts/demo_exact/regional" / args.case
    result = json.loads((folder / "regional_measures.json").read_text())
    evidence = folder / "cortical_qc.png"
    segmentation = folder / args.case / "mri/aparc.DKTatlas+aseg.deep.mgz"
    if sha256(segmentation) != result["segmentation_sha256"] or not evidence.is_file() or len(result["regions"]) != 5:
        raise ValueError("Incomplete segmentation or changed evidence")
    path = ROOT / "assets/selected-regional-review.json"
    reviews = json.loads(path.read_text()) if path.exists() else {}
    reviews[result["patient_id"]] = {"segmentation_sha256": sha256(segmentation), "evidence_sha256": sha256(evidence),
        "state": "technically_reviewed", "scope": "gross_anatomical_alignment_only", "clinical_validation": False,
        "reviewer": "coding_assistant_visual_review", "note": args.note}
    write_json(path, reviews)
    (ROOT / "assets/selected-regional-review.js").write_text("window.NEUROMETRIC_REGIONAL_REVIEW = " + json.dumps(reviews) + ";\n")


if __name__ == "__main__":
    main()
