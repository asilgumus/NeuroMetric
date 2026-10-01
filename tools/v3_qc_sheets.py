"""Build age/site-stratified visual QC sheets from completed V3 shards.

This generates review material, not an automatic approval. Inspect the sheets
and failed-subject CSV before deciding whether the cohort is fit for training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def select_review(shards: list[Path], per_site: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for root in shards:
        summary = json.loads((root / "prepare_summary.json").read_text())
        if not summary.get("complete"):
            raise ValueError(f"Incomplete V3 shard: {root}")
        frame = pd.read_csv(root / "manifest.csv")
        if len(frame) != summary["total"] or frame.subject_id.duplicated().any():
            raise ValueError(f"Invalid V3 shard manifest: {root}")
        frame["shard_root"] = str(root.resolve())
        records.append(frame)
    if not records:
        raise ValueError("At least one completed shard is required")
    combined = pd.concat(records, ignore_index=True)
    if combined.subject_id.duplicated().any():
        raise ValueError("A subject occurs in multiple shards")
    failed = combined.loc[combined.qc_status != "automatic_checks_passed"].copy()
    passed = combined.loc[combined.qc_status == "automatic_checks_passed"].copy()
    selected = []
    for _, group in passed.groupby(["dataset", "site"], sort=True):
        ordered = group.sort_values(["age", "subject_id"])
        indices = np.linspace(0, len(ordered) - 1,
                              min(per_site, len(ordered)), dtype=int)
        selected.append(ordered.iloc[np.unique(indices)])
    review = pd.concat(selected, ignore_index=True).sort_values(
        ["dataset", "site", "age", "subject_id"])
    review["qc_image"] = [str(Path(root) / "qc" / f"{subject.replace(':', '_')}.png")
                          for root, subject in zip(review.shard_root, review.subject_id)]
    review["resnet_qc_image"] = [str(Path(root) / "qc_resnet" /
                                     f"{subject.replace(':', '_')}.png")
                                 for root, subject in zip(review.shard_root,
                                                          review.subject_id)]
    missing = [path for path in (*review.qc_image, *review.resnet_qc_image)
               if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing QC montage, e.g. {missing[0]}")
    return review, failed


def save_sheets(review: pd.DataFrame, output: Path, columns=3, rows=3):
    output.mkdir(parents=True, exist_ok=True)
    tile_w, modality_h, label_h = 400, 160, 44
    tile_h = 2 * modality_h
    page_size = columns * rows
    for page, start in enumerate(range(0, len(review), page_size), 1):
        sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(review.iloc[start:start + page_size].itertuples()):
            left = (offset % columns) * tile_w
            top = (offset // columns) * (tile_h + label_h)
            for modality, path, offset in (("SFCN", row.qc_image, 0),
                                           ("ResNet", row.resnet_qc_image, modality_h)):
                with Image.open(path) as source:
                    thumb = ImageOps.contain(source.convert("RGB"),
                                             (tile_w, modality_h - 18))
                sheet.paste(thumb, (left + (tile_w - thumb.width) // 2,
                                    top + offset + 18))
                draw.text((left + 8, top + offset + 2), modality, fill="black")
            label = f"{row.subject_id} | {row.site} | {row.age:.1f} y"
            draw.text((left + 8, top + tile_h + 8), label, fill="black")
        sheet.save(output / f"qc_sheet_{page:03d}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-site", type=int, default=12)
    args = parser.parse_args()
    if args.per_site < 2:
        parser.error("per-site must be at least 2 to include age extremes")
    review, failed = select_review(args.shards, args.per_site)
    args.output.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.output / "qc_review_index.csv", index=False)
    failed.to_csv(args.output / "qc_automatic_failures.csv", index=False)
    save_sheets(review, args.output)
    print(f"QC sheets: {len(review)} reviewed candidates, {len(failed)} automatic failures; "
          f"output={args.output}")


if __name__ == "__main__":
    main()
