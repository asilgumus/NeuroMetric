"""Download a small paired OASIS-2 sample without retaining the full archive.

Streaming selection preserves original files and never extracts arbitrary paths.
The public archive is ordered independently of the metadata: use --subject to
select another subject if the first sample lacks a second visit in part 1.
"""
import argparse
import json
from pathlib import Path
import tarfile
import re
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.oasis_visit_selection import visit_ids

ARCHIVE = "https://download.nrg.wustl.edu/data/OAS2_RAW_PART1.tar.gz"
DEMOGRAPHICS = "https://sites.wustl.edu/oasisbrains/files/2024/03/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/oasis2"))
    parser.add_argument("--subject", default="OAS2_0001")
    parser.add_argument("--subjects", nargs="+", help="Select several pairs in one archive pass")
    args = parser.parse_args()
    subjects = args.subjects or [args.subject]
    if not all(re.fullmatch(r"OAS2_\d{4}", subject) for subject in subjects):
        raise ValueError("Expected OAS2 subject identifier")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = args.output / "demographics.xlsx"
    if not metadata.exists():
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--max-time", "60",
                        "--output", str(metadata), DEMOGRAPHICS], check=True)
    selected = [identifier for subject in subjects for identifier in visit_ids(metadata, subject)]
    manifest = {"dataset": "OASIS-2", "archive_url": ARCHIVE,
                "demographics_url": DEMOGRAPHICS, "subject_id": args.subject,
                "subject_ids": subjects,
                "selected_visits": selected,
                "status": "downloading", "files": []}
    status = args.output / "download_status.json"
    def record():
        status.write_text(json.dumps(manifest, indent=2) + "\n")
    existing = [p for identifier in selected for p in (args.output / "raw" / identifier).glob("mpr-1*.img")]
    if len(existing) == 2 * len(subjects):
        import nibabel as nib
        for path in existing:
            image = nib.load(path)
            if image.header.get_xyzt_units()[0] != "mm" or not (image.header.get("sform_code", 0) or image.header.get("qform_code", 0)):
                raise ValueError("Downloaded image lacks physical NIfTI geometry")
            image.get_fdata()
        manifest.update(status="paired_sample_downloaded", files=[str(p) for p in existing])
        record()
        print("Existing paired MRI sample validated", flush=True)
        return
    record()
    seen = set()
    try:
        process = subprocess.Popen(["curl", "--fail", "--location", "--silent", "--show-error",
                                    "--connect-timeout", "30", "--speed-limit", "1024",
                                    "--speed-time", "60", ARCHIVE], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|gz") as archive:
                for member in archive:
                    name = member.name
                    match = re.search(r"(" + "|".join(selected) + r")(?:/|$)", name)
                    if not match or not member.isfile():
                        continue
                    basename = Path(name).name
                    wanted = ("mpr-1" in basename and basename.endswith((".img", ".hdr"))) or basename.endswith((".xml", ".txt"))
                    if not wanted or member.size > 100_000_000:
                        continue
                    target = args.output / "raw" / match[1] / basename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = archive.extractfile(member)
                    with target.open("wb") as destination:
                        while block := stream.read(1024 * 1024):
                            destination.write(block)
                    manifest["files"].append(str(target))
                    seen.add((match[1], target.suffix))
                    print(f"Downloaded {target} ({member.size} bytes)", flush=True)
                    record()
                    if all((identifier, suffix) in seen for identifier in selected for suffix in (".img", ".hdr")):
                        manifest["status"] = "paired_sample_downloaded"
                        record()
                        return
            if process.wait() != 0:
                raise RuntimeError("Archive transfer failed; see curl error above")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        manifest["status"] = "archive_finished_pair_incomplete"
        record()
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        record()
        raise


if __name__ == "__main__":
    main()
