"""Download OASIS-1 research data only after explicit DUA acceptance."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import tarfile

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256, write_json
from oasis_test import eligible_clinical, matched_manifest

CSV_URL = "https://www.nitrc.org/frs/download.php/6348/oasis_cross-sectional.csv?i_agree=1&download_now=1"
DUA_URL = "https://www.nitrc.org/frs/download.php/6348/oasis_cross-sectional.csv"


def fetch(url, target):
    if target.exists():
        return
    partial = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    partial.replace(target)


def extract_raw(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = Path(member.name)
            if not member.isfile() or "RAW" not in path.parts:
                continue
            if not re.fullmatch(r"OAS1_\d{4}_MR1_mpr-1_anon\.(?:img|hdr)(?:\.gz)?", path.name):
                continue
            target = (destination / path).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
            tar.extract(member, destination, filter="data")


def fetch_pilot(clinical, output, cases):
    frame = eligible_clinical(pd.read_csv(clinical))
    case_ids = frame[frame.group != "control"].sort_values("ID").head(cases).ID
    frame = frame[(frame.group == "control") | frame.ID.isin(case_ids)]
    pairs, _ = matched_manifest(frame, dict.fromkeys(frame.ID, "pending"))
    if len(pairs) // 2 != cases:
        raise ValueError(f"Only {len(pairs)//2} matched pairs available; requested {cases}")
    pd.read_csv(clinical).query("ID in @pairs.subject_id").to_csv(
        output / "pilot_clinical.csv", index=False)
    wanted = set(pairs.subject_id)
    found = set()
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    for p in raw.rglob("*"):
        match = re.fullmatch(r"(OAS1_\d{4}_MR1)_mpr-1_anon\.(img|hdr)", p.name)
        if match and p.is_file() and p.stat().st_size > 0:
            found.add((match[1], match[2]))
    required = {(subject, extension) for subject in wanted for extension in ("img", "hdr")}
    provenance = {"requested_cases": cases, "controls": cases, "subjects": sorted(wanted),
                  "clinical_sha256": sha256(clinical), "archive_urls_read": [],
                  "note": "Compressed archive streams may transfer unselected data; only selected raw files retained"}
    for disc in range(1, 13):
        if required <= found:
            break
        url = f"https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc{disc}.tar.gz"
        print(f"Streaming disc {disc}: selected files {len(required & found)}/{len(required)}", flush=True)
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                for member in tar:
                    path = Path(member.name)
                    match = re.fullmatch(r"(OAS1_\d{4}_MR1)_mpr-1_anon\.(img|hdr)", path.name)
                    if not member.isfile() or "RAW" not in path.parts or not match:
                        continue
                    key = (match[1], match[2])
                    if key not in required or key in found:
                        continue
                    target = (raw / path).resolve()
                    if not target.is_relative_to(raw.resolve()):
                        raise ValueError("Unsafe archive path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    part = target.with_suffix(target.suffix + ".part")
                    with tar.extractfile(member) as source, part.open("wb") as destination:
                        while chunk := source.read(1024 * 1024):
                            destination.write(chunk)
                    if part.stat().st_size != member.size:
                        raise ValueError("Incomplete raw file")
                    part.replace(target)
                    found.add(key)
                    print(f"Selected {path.name}: {len(required & found)}/{len(required)}", flush=True)
                    if required <= found:
                        break
        provenance["archive_urls_read"].append(url)
        provenance["found_files"] = len(required & found)
        write_json(output / "pilot_provenance.json", provenance)
    if not required <= found:
        raise ValueError(f"Missing selected files: {sorted(required - found)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-dua", action="store_true",
                        help="Explicit acceptance of OASIS DUA; no reidentification, required acknowledgements")
    parser.add_argument("--discs", type=int, default=12)
    parser.add_argument("--cases", type=int,
                        help="Download only this many matched cases and controls via archive streams")
    parser.add_argument("--output", type=Path, default=ROOT / "data/oasis1")
    args = parser.parse_args()
    if not args.accept_dua:
        parser.error(f"Read and accept OASIS DUA before downloading: {DUA_URL}")
    if not 1 <= args.discs <= 12:
        parser.error("Discs must be between 1 and 12")
    if args.cases is not None and not 1 <= args.cases <= 100:
        parser.error("Cases must be between 1 and 100")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "dua_acceptance.json", {"source": DUA_URL,
               "accepted_via": "explicit --accept-dua flag", "at": datetime.now(timezone.utc).isoformat(),
               "research_only": True, "no_reidentification": True})
    clinical = args.output / "oasis_cross-sectional.csv"
    fetch(CSV_URL, clinical)
    if not {"ID", "Age", "CDR", "M/F"} <= set(pd.read_csv(clinical).columns):
        raise ValueError("Clinical download is not expected CSV; do not use HTML as data")
    if args.cases is not None:
        fetch_pilot(clinical, args.output, args.cases)
        return
    provenance = {"clinical_url": CSV_URL, "clinical_sha256": sha256(clinical), "archives": []}
    for disc in range(1, args.discs + 1):
        url = f"https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc{disc}.tar.gz"
        archive = args.output / f"disc{disc}.tar.gz"
        print(f"Downloading/extracting disc {disc}/{args.discs}", flush=True)
        fetch(url, archive)
        extract_raw(archive, args.output / "raw")
        provenance["archives"].append({"url": url, "sha256": sha256(archive)})
        write_json(args.output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
