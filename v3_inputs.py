"""Locate complete, portable V3 cohort outputs mounted as Kaggle inputs."""
from __future__ import annotations

import json
import csv
from pathlib import Path

from brainage import sha256


def find_checkpoint_by_hash(inputs: Path, expected_sha256: str) -> Path:
    matches = [path for path in inputs.rglob("*.pt")
               if path.name in {"best.pt", "best_hits.pt"}
               if path.is_file() and sha256(path) == expected_sha256]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one checkpoint with SHA-256 {expected_sha256}; found {len(matches)}")
    return matches[0]


def discover_cohorts(inputs: Path, sources: tuple[str, ...], catalog: Path):
    expected = set(sources)
    catalog_hash = sha256(catalog)
    cohorts = {}
    for summary_path in inputs.rglob("prepare_summary.json"):
        root = summary_path.parent
        # A cohort merge exports prepared/<source>/manifest.csv. Source shard
        # outputs have a different shape and must never be used here.
        if root.parent.name != "prepared":
            continue
        summary = json.loads(summary_path.read_text())
        source = summary.get("source")
        if source not in expected:
            continue
        if source in cohorts:
            raise ValueError(f"Duplicate merged V3 cohort: {source}")
        if not summary.get("complete") or summary.get("catalog_sha256") != catalog_hash:
            raise ValueError(f"Incomplete or wrong-catalog V3 cohort: {root}")
        if root.name != source.lower() or not (root / "manifest.csv").is_file():
            raise ValueError(f"Malformed merged V3 cohort: {root}")
        cohorts[source] = root
    if set(cohorts) != expected:
        raise FileNotFoundError(f"Expected merged V3 cohorts {expected}; found {set(cohorts)}")
    return cohorts


def verify_qc_approval(cohorts: dict[str, Path], approval_path: Path, catalog: Path):
    """Require a human-review record tied to the exact mounted cohort outputs."""
    approval = json.loads(approval_path.read_text())
    if approval.get("catalog_sha256") != sha256(catalog):
        raise ValueError("QC approval catalogue hash mismatch")
    records = approval.get("sources", {})
    for source, root in cohorts.items():
        record = records.get(source, {})
        if record.get("decision") != "approved" or not record.get("reviewer"):
            raise ValueError(f"Human QC approval missing for {source}")
        for key, value in qc_fingerprint(root).items():
            if record.get(key) != value:
                raise ValueError(f"QC approval {key} hash mismatch: {source}")


def qc_fingerprint(cohort_root: Path):
    """Hash all review material so an approval binds to an exact output."""
    qc = cohort_root.parent.parent / "qc_review"
    index_path = qc / "qc_review_index.csv"
    with index_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"subject_id", "qc_image", "resnet_qc_image"}.issubset(
                reader.fieldnames or []):
            raise ValueError("QC review index must cover both SFCN and ResNet")
        if not any(row["qc_image"] and row["resnet_qc_image"] for row in reader):
            raise ValueError("QC review index has no two-modality examples")
    sheets = {path.name: sha256(path) for path in qc.glob("qc_sheet_*.png")}
    if not sheets:
        raise FileNotFoundError(f"No human-review QC sheets in {qc}")
    return {"qc_protocol": "sfcn+resnet-v1",
            "summary_sha256": sha256(cohort_root / "prepare_summary.json"),
            "manifest_sha256": sha256(cohort_root / "manifest.csv"),
            "review_index_sha256": sha256(index_path),
            "failures_sha256": sha256(qc / "qc_automatic_failures.csv"),
            "sheets": sheets}
