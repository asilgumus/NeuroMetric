"""Auditable public-data catalogue for BrainAGE V3.

Only source metadata is fetched by ``catalog``. MRI payloads are fetched by
``prepare`` after the participant split has been locked.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split


S3 = "http://s3.amazonaws.com/doc/2006-03-01/"
SALD_SHEET = "https://fcon_1000.projects.nitrc.org/indi/retro/SALD/sub_information.xlsx"
SALD_SHEET_SHA256 = "52b4edd11332382615ab5f39091360b0b4106d36637b6991c535fa5febae1e19"
SALD_BUCKET = "fcp-indi"
SALD_PREFIX = "data/Projects/INDI/SALD/RawData_BIDS/"
OPENNEURO_BUCKET = "openneuro.org"
NIMH_PREFIX = "ds005752/"
DLBS_PREFIX = "ds004856/"
DATASET_VERSIONS = {"IXI": "existing-ixi-prepare-manifest", "SALD": "2018-public-release",
                    "NIMH": "OpenNeuro-ds005752-v2.0.0", "DLBS": "OpenNeuro-ds004856-v1.2.0"}
DATASET_LICENSES = {"IXI": "CC-BY-SA-3.0", "SALD": "CC-BY-NC",
                    "NIMH": "CC0-1.0", "DLBS": "CC0-1.0"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def get_bytes(url: str, *, insecure_nitrc: bool = False) -> bytes:
    if insecure_nitrc and url != SALD_SHEET:
        raise ValueError("Unverified TLS is permitted only for the pinned SALD spreadsheet")
    response = requests.get(url, timeout=90, verify=not insecure_nitrc)
    response.raise_for_status()
    value = response.content
    if insecure_nitrc and sha256_bytes(value) != SALD_SHEET_SHA256:
        raise ValueError("SALD spreadsheet checksum mismatch")
    return value


def s3_url(bucket: str, key: str) -> str:
    return f"https://s3.amazonaws.com/{bucket}/{urllib.parse.quote(key, safe='/')}"


def s3_objects(bucket: str, prefix: str):
    """List public S3 objects with pagination; yield (key, bytes, etag)."""
    continuation = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if continuation:
            params["continuation-token"] = continuation
        response = requests.get(f"https://s3.amazonaws.com/{bucket}", params=params, timeout=90)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall(f"{{{S3}}}Contents"):
            yield (item.findtext(f"{{{S3}}}Key"), int(item.findtext(f"{{{S3}}}Size")),
                   item.findtext(f"{{{S3}}}ETag").strip('"'))
        continuation = root.findtext(f"{{{S3}}}NextContinuationToken")
        if not continuation:
            break


def participant_objects(bucket: str, prefixes, workers=16):
    """Avoid listing every fMRI/PET object in large public BIDS archives."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for group in pool.map(lambda prefix: list(s3_objects(bucket, prefix)), prefixes):
            yield from group


def _record(dataset, subject, age, site, key, bucket, *, split="", checksum=""):
    return {"dataset": dataset, "subject_id": f"{dataset}:{subject}", "age": float(age),
            "site": site, "split": split, "source_url": s3_url(bucket, key),
            "source_name": Path(key).name, "source_etag": checksum,
            "dataset_version": DATASET_VERSIONS[dataset], "license": DATASET_LICENSES[dataset]}


def ixi_rows(manifest_path: Path):
    frame = pd.read_csv(manifest_path)
    expected = {"subject_id", "age", "site", "split", "source_path", "qc_status"}
    if not expected.issubset(frame):
        raise ValueError("IXI manifest is missing required columns")
    rows = []
    for row in frame.itertuples():
        if row.qc_status == "failed":
            continue
        rows.append({"dataset": "IXI", "subject_id": f"IXI:{row.subject_id}",
                     "age": float(row.age), "site": f"IXI-{row.site}", "split": row.split,
                     "source_url": "", "source_name": Path(row.source_path).name,
                     "source_etag": "", "dataset_version": DATASET_VERSIONS["IXI"],
                     "license": DATASET_LICENSES["IXI"]})
    return rows


def sald_rows():
    import urllib3
    # The original archive host currently serves a mismatched TLS certificate.
    # Restrict this exception to the official spreadsheet and pin its SHA-256.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    sheet = pd.read_excel(io.BytesIO(get_bytes(SALD_SHEET, insecure_nitrc=True)))
    image_objects = {key: etag for key, _, etag in s3_objects(SALD_BUCKET, SALD_PREFIX)
                     if key.endswith("_T1w.nii.gz")}
    rows = []
    for row in sheet.itertuples():
        if row.T1Img != 1 or not np.isfinite(row.Age):
            continue
        subject = f"{int(row.Sub_ID):06d}"
        key = f"{SALD_PREFIX}sub-{subject}/anat/sub-{subject}_T1w.nii.gz"
        if key not in image_objects:
            continue
        rows.append(_record("SALD", subject, row.Age, "SALD", key, SALD_BUCKET,
                            checksum=image_objects[key]))
    return rows


def nimh_rows():
    participants = pd.read_csv(io.BytesIO(get_bytes(s3_url(OPENNEURO_BUCKET,
                                                     NIMH_PREFIX + "participants.tsv"))), sep="\t")
    participants = participants.loc[(participants.MRI == 1) & (participants.eligibility == 1)].copy()
    prefixes = [f"{NIMH_PREFIX}{subject}/ses-01/anat/" for subject in participants.participant_id]
    objects = [(key, etag) for key, _, etag in participant_objects(OPENNEURO_BUCKET, prefixes)
               if re.search(r"/ses-01/anat/.*_T1w\.nii\.gz$", key)
               and "rec-SCIC" not in key and "HighResHippo" not in key]
    by_subject = {}
    for key, etag in objects:
        match = re.search(r"/(sub-ON\d+)/ses-01/anat/", key)
        if match:
            by_subject.setdefault(match[1], []).append((key, etag))
    rows = []
    for row in participants.itertuples():
        age = pd.to_numeric(row.age, errors="coerce")
        if row.MRI != 1 or row.eligibility != 1 or not np.isfinite(age):
            continue
        scans = by_subject.get(row.participant_id, [])
        # One canonical whole-brain T1 per person, deterministic across reruns.
        scans.sort(key=lambda item: ("MPRAGE" not in item[0].upper(), item[0]))
        if not scans:
            continue
        key, etag = scans[0]
        site = "NIMH-MPRAGE" if "MPRAGE" in key.upper() else "NIMH-FSPGR"
        rows.append(_record("NIMH", row.participant_id, age, site, key,
                            OPENNEURO_BUCKET, checksum=etag))
    return rows


def dlbs_rows():
    participants = pd.read_csv(io.BytesIO(get_bytes(s3_url(OPENNEURO_BUCKET,
                                                     DLBS_PREFIX + "participants.tsv"))), sep="\t")
    participants = participants.loc[pd.to_numeric(participants.AgeMRI_W1, errors="coerce").notna()].copy()
    prefixes = [f"{DLBS_PREFIX}{subject}/ses-wave1/anat/" for subject in participants.participant_id]
    objects = [(key, etag) for key, _, etag in participant_objects(OPENNEURO_BUCKET, prefixes)
               if re.search(r"/ses-wave1/anat/.*_T1w\.nii\.gz$", key)]
    by_subject = {}
    for key, etag in objects:
        match = re.search(r"/(sub-[^/]+)/ses-wave1/anat/", key)
        if match:
            by_subject.setdefault(match[1], []).append((key, etag))
    rows = []
    for row in participants.itertuples():
        age = pd.to_numeric(row.AgeMRI_W1, errors="coerce")
        if not np.isfinite(age):
            continue
        scans = sorted(by_subject.get(row.participant_id, []))
        if not scans:
            continue
        key, etag = scans[0]
        rows.append(_record("DLBS", row.participant_id, age, "DLBS-W1", key,
                            OPENNEURO_BUCKET, split="external", checksum=etag))
    return rows


def assign_splits(rows, seed=42):
    df = pd.DataFrame(rows)
    if df.subject_id.duplicated().any():
        raise ValueError("Duplicate participant IDs in V3 catalogue")
    df = df.loc[df.age.between(18, 100)].sort_values(["dataset", "subject_id"]).reset_index(drop=True)
    for dataset in ("SALD", "NIMH"):
        mask = df.dataset == dataset
        group = df.loc[mask]
        if len(group) < 20:
            raise ValueError(f"Too few {dataset} T1+age subjects: {len(group)}")
        bins = pd.qcut(group.age.rank(method="first"), q=4, labels=False)
        train, val = train_test_split(group.index, test_size=.20, random_state=seed, stratify=bins)
        df.loc[train, "split"] = "train"
        df.loc[val, "split"] = "val"
    if df.split.isna().any() or (df.split == "").any():
        raise ValueError("Unassigned split")
    return df


def catalogue(ixi_manifest: Path, output: Path):
    rows = ixi_rows(ixi_manifest) + sald_rows() + nimh_rows() + dlbs_rows()
    df = assign_splits(rows)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "catalog.csv", index=False)
    (output / "summary.json").write_text(json.dumps({
        "n": len(df), "by_dataset_split": {f"{d}:{s}": int(n) for (d, s), n in
            df.groupby(["dataset", "split"]).size().items()},
        "source_versions": DATASET_VERSIONS, "licenses": DATASET_LICENSES,
        "sald_spreadsheet_sha256": SALD_SHEET_SHA256,
        "ixi_manifest_sha256": sha256_bytes(ixi_manifest.read_bytes())}, indent=2))
    return df
