"""Join immutable preparation shards into one manifest and array namespace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from brainage import sha256, write_json


def resnet_central_occupancy(array_path: Path) -> float:
    """Detect registrations with no anatomy in the CNN1 model's central crop."""
    x = np.load(array_path, allow_pickle=False)
    if x.shape != (80, 96, 80) or not np.isfinite(x).all():
        raise ValueError(f"Invalid ResNet input: {array_path}")
    return float((x[20:60, 24:72, 20:60] > -0.9).mean())


def save_resnet_qc(array_path: Path, output: Path, subject: str):
    """Render the actual CNN1 input, including IXI arrays reused from V2."""
    x = np.load(array_path, allow_pickle=False)
    if x.shape != (80, 96, 80) or not np.isfinite(x).all():
        raise ValueError(f"Invalid ResNet input for QC: {array_path}")
    canvas = Image.new("RGB", (720, 250), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{subject} | ResNet input", fill="white")
    for index, plane in enumerate((x[40, :, :], x[:, 48, :], x[:, :, 40])):
        pixels = np.clip((np.rot90(plane) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        image = ImageOps.contain(Image.fromarray(pixels, mode="L"), (230, 210))
        left = index * 240 + (240 - image.width) // 2
        canvas.paste(image.convert("RGB"), (left, 32 + (210 - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def merge(catalog_path: Path, shard_roots: list[Path], output: Path,
          sources=("IXI", "SALD", "NIMH", "DLBS"), materialize=False):
    catalog = pd.read_csv(catalog_path)
    catalog_hash = sha256(catalog_path)
    grouped: dict[str, list[tuple[Path, dict, pd.DataFrame]]] = {}
    for root in shard_roots:
        summary = json.loads((root / "prepare_summary.json").read_text())
        if summary.get("source") not in {"IXI", "SALD", "NIMH", "DLBS"}:
            raise ValueError(f"Not a V3 preparation shard: {root}")
        if summary.get("catalog_sha256") != catalog_hash or not summary.get("complete"):
            raise ValueError(f"Incomplete or wrong-catalog shard: {root}")
        frame = pd.read_csv(root / "manifest.csv")
        if len(frame) != summary["total"] or frame.subject_id.duplicated().any():
            raise ValueError(f"Shard manifest count/IDs invalid: {root}")
        grouped.setdefault(summary["source"], []).append((root, summary, frame))
    if set(grouped) != set(sources):
        raise ValueError(f"Missing source preparation: {set(grouped)}")
    output.mkdir(parents=True, exist_ok=True)
    for source, shards in grouped.items():
        counts = {summary["shard_count"] for _, summary, _ in shards}
        if len(counts) != 1:
            raise ValueError(f"Shard counts conflict: {source}")
        count = counts.pop()
        if {summary["shard_index"] for _, summary, _ in shards} != set(range(count)):
            raise ValueError(f"Missing/duplicate shard index: {source}")
        combined = pd.concat([frame for _, _, frame in shards], ignore_index=True)
        expected = set(catalog.loc[catalog.dataset == source, "subject_id"])
        if set(combined.subject_id) != expected or combined.subject_id.duplicated().any():
            raise ValueError(f"Subject coverage mismatch: {source}")
        destination = output / source.lower()
        destination.mkdir(exist_ok=True)
        origin = {row.subject_id: root for root, _, frame in shards
                  for row in frame.itertuples()}
        for row in combined.itertuples():
            if row.qc_status != "automatic_checks_passed":
                continue
            resnet_source = origin[row.subject_id] / row.resnet_array
            if not resnet_source.is_file():
                raise FileNotFoundError(resnet_source)
            occupancy = resnet_central_occupancy(resnet_source)
            if occupancy < 0.10:
                index = combined.index[combined.subject_id == row.subject_id][0]
                combined.loc[index, "qc_status"] = "failed"
                combined.loc[index, "error"] = (
                    f"ResNet central anatomy absent: occupancy={occupancy:.4f}")
                continue
            for name in (row.resnet_array, row.sfcn_array):
                source_file = origin[row.subject_id] / name
                if not source_file.exists():
                    raise FileNotFoundError(source_file)
                link = destination / name
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    if link.resolve() != source_file.resolve():
                        raise ValueError(f"Conflicting prepared link: {link}")
                elif link.exists():
                    raise ValueError(f"Refusing to overwrite prepared file: {link}")
                else:
                    if materialize:
                        shutil.copy2(source_file, link)
                    else:
                        link.symlink_to(source_file.resolve())
            if materialize:
                qc_name = row.subject_id.replace(":", "_") + ".png"
                qc_source = origin[row.subject_id] / "qc" / qc_name
                if not qc_source.is_file():
                    raise FileNotFoundError(qc_source)
                qc_destination = destination / "qc" / qc_name
                qc_destination.parent.mkdir(parents=True, exist_ok=True)
                if qc_destination.exists():
                    raise ValueError(f"Refusing to overwrite QC image: {qc_destination}")
                shutil.copy2(qc_source, qc_destination)
                save_resnet_qc(destination / row.resnet_array,
                               destination / "qc_resnet" / qc_name, row.subject_id)
        combined.sort_values("subject_id").to_csv(destination / "manifest.csv", index=False)
        write_json(destination / "prepare_summary.json", {
            "source": source, "total": len(combined),
            "passed": int((combined.qc_status == "automatic_checks_passed").sum()),
            "failed": int((combined.qc_status != "automatic_checks_passed").sum()),
            "catalog_sha256": catalog_hash, "complete": True,
            "manual_qc": "pending", "shard_count": count})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["IXI", "SALD", "NIMH", "DLBS"])
    parser.add_argument("--materialize", action="store_true",
                        help="Copy arrays and QC images for portable kernel output")
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    merge(args.catalog, args.shards, args.output, sources=args.sources,
          materialize=args.materialize)


if __name__ == "__main__":
    main()
