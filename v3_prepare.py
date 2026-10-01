"""Prepare V3 ResNet and SFCN inputs from the locked public-data catalogue."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
import requests

from brainage import bootstrap, check_volume, preprocess_one, sha256, write_json


SFCN_SHAPE = (160, 192, 160)
SYNTHSTRIP_URL = "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt"
SYNTHSTRIP_SHA256 = "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"


def ensure_fast(cache: Path):
    fsl = cache / "fsl"
    if (fsl / "bin/fast").exists():
        return
    mamba = cache / "bin/micromamba"
    if not mamba.exists():
        raise FileNotFoundError("micromamba missing after FSL bootstrap")
    env = os.environ.copy()
    env["MAMBA_ROOT_PREFIX"] = str(cache / "mamba-root")
    subprocess.run([str(mamba), "install", "-y", "-p", str(fsl),
                    "-c", "https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/",
                    "-c", "conda-forge", "fsl-fast4"], check=True, env=env)
    if not (fsl / "bin/fast").exists():
        raise FileNotFoundError("FSL FAST installation did not produce fast")


def ensure_synthstrip(cache: Path):
    model = cache / "synthstrip.1.pt"
    if model.exists() and sha256(model) == SYNTHSTRIP_SHA256:
        return model
    model.parent.mkdir(parents=True, exist_ok=True)
    temporary = model.with_suffix(".pt.part")
    with requests.get(SYNTHSTRIP_URL, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    if sha256(temporary) != SYNTHSTRIP_SHA256:
        raise ValueError("SynthStrip weights checksum mismatch")
    temporary.replace(model)
    return model


def download_mri(url: str, target: Path, expected_etag: str):
    if target.exists() and target.stat().st_size > 100_000:
        return sha256(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                if response.headers.get("ETag", "").strip('"') != expected_etag:
                    raise ValueError(f"Source MRI ETag changed: {url}")
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if temporary.stat().st_size <= 100_000:
                raise ValueError("Downloaded MRI is unexpectedly small")
            temporary.replace(target)
            return sha256(target)
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def resolve_ixi(row, ixi_images: Path):
    matches = list(ixi_images.rglob(row["source_name"]))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one IXI image {row['source_name']}; got {len(matches)}")
    return matches[0]


def ixi_sfcn_input(raw: Path, folder: Path, fsl: Path, subject_id: str,
                   ixi_prepared: Path):
    """Use the label-blind V2 rigid transform to limit SynthStrip's FOV/RAM.

    The released CNN1 preprocessing reorients the native scan before fitting
    its FLIRT matrix. Applying that matrix directly to the raw scan is wrong.
    """
    folder.mkdir(parents=True, exist_ok=True)
    reoriented = folder / "reoriented.nii.gz"
    aligned = folder / "initial_mni.nii.gz"
    cropped = folder / "initial_mni_crop.nii.gz"
    matrix = ixi_prepared / "registration" / subject_id / f"{subject_id}_mni_dof_6.mat"
    reference = fsl / "data/standard/MNI152_T1_1mm.nii.gz"
    if not matrix.exists():
        raise FileNotFoundError(f"IXI rigid registration matrix missing: {matrix}")
    with (folder / "initial_alignment.log").open("w") as log:
        subprocess.run([str(fsl / "bin/fslreorient2std"), str(raw), str(reoriented)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        subprocess.run([str(fsl / "bin/flirt"), "-in", str(reoriented),
                        "-ref", str(reference), "-applyxfm", "-init", str(matrix),
                        "-out", str(aligned)], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=1200)
    image = nib.load(str(aligned))
    starts = np.array([(size - target) // 2 for size, target in zip(image.shape, SFCN_SHAPE)])
    if np.any(starts < 0):
        raise ValueError(f"IXI initially aligned image too small: {image.shape}")
    values = np.asarray(image.dataobj)[tuple(slice(int(i), int(i + size)) for i, size
                                                in zip(starts, SFCN_SHAPE))]
    affine = image.affine.copy()
    affine[:3, 3] = image.affine[:3, :3] @ starts + image.affine[:3, 3]
    nib.save(nib.Nifti1Image(values, affine, image.header), str(cropped))
    return cropped


def sfcn_array(path: Path, folder: Path, fsl: Path):
    """SFCN UKB-style 1 mm, brain-extracted, bias-corrected linear MNI input."""
    folder.mkdir(parents=True, exist_ok=True)
    fast = fsl / "bin/fast"
    flirt = fsl / "bin/flirt"
    template = fsl / "data/standard/MNI152_T1_1mm_brain.nii.gz"
    if not template.exists():
        raise FileNotFoundError(template)
    brain = folder / "brain.nii.gz"
    bias = folder / "bias"
    aligned = folder / "aligned.nii.gz"
    commands = [
        [sys.executable, "-m", "nipreps.synthstrip", "-i", str(path), "-o", str(brain),
         "--model", str(ensure_synthstrip(fsl.parent)), "-n", "1"],
        [str(fast), "-B", "-o", str(bias), str(brain)],
        [str(flirt), "-in", str(folder / "bias_restore.nii.gz"), "-ref", str(template),
         "-dof", "12", "-omat", str(folder / "to_mni.mat"), "-out", str(aligned)],
    ]
    with (folder / "sfcn.log").open("w") as log:
        for command in commands:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=1200)
    return aligned_to_array(aligned, fsl)


def aligned_to_array(aligned: Path, fsl: Path):
    """Crop and quantitatively reject gross misregistration, without rerunning FSL."""
    volume = nib.load(str(aligned)).get_fdata(dtype=np.float32)
    if any(size < crop for size, crop in zip(volume.shape, SFCN_SHAPE)):
        raise ValueError(f"Aligned image too small: {volume.shape}")
    starts = [(a - b) // 2 for a, b in zip(volume.shape, SFCN_SHAPE)]
    x = volume[tuple(slice(start, start + size) for start, size in zip(starts, SFCN_SHAPE))]
    positive = x[x > 0]
    if len(positive) < x.size * .05 or not np.isfinite(x).all():
        raise ValueError("Invalid SFCN brain mask or non-finite voxels")
    scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid SFCN intensity normalization")
    mask_image = nib.load(str(fsl / "data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz"))
    mask = mask_image.get_fdata(dtype=np.float32)
    mask = mask[tuple(slice(start, start + size) for start, size in zip(starts, SFCN_SHAPE))] > .5
    foreground = x > 0
    outside = float((foreground & ~mask).sum() / foreground.sum())
    if outside > .05:
        raise ValueError(f"SFCN registration/skull strip QC failed: outside-MNI-brain={outside:.3f}")
    x = (x / scale).astype(np.float16)
    return x


def save_qc(array: np.ndarray, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, plane in zip(axes, [array[array.shape[0] // 2, :, :],
                                array[:, array.shape[1] // 2, :],
                                array[:, :, array.shape[2] // 2]]):
        ax.imshow(np.rot90(plane), cmap="gray")
        ax.axis("off")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def valid_array(path: Path, shape):
    if not path.exists():
        return False
    try:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                data = loaded["x"]
        else:
            data = loaded
        return data.shape == shape and np.isfinite(data).all()
    except (OSError, ValueError, KeyError, EOFError):
        return False


def resnet_center_occupancy(path: Path) -> float:
    """A near-empty central crop indicates a failed CNN1 registration."""
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != (80, 96, 80) or not np.isfinite(x).all():
        raise ValueError(f"Invalid ResNet array: {path}")
    return float((x[20:60, 24:72, 20:60] > -0.9).mean())


def prepare_one(row: dict, output: str, cache: str, ixi_images: str, ixi_prepared: str):
    start = time.time()
    folder = Path(output)
    cache = Path(cache)
    subject = row["subject_id"].replace(":", "_")
    raw = folder / "raw" / row["dataset"] / row["source_name"]
    try:
        if row["dataset"] == "IXI":
            raw = resolve_ixi(row, Path(ixi_images))
            source_digest = sha256(raw)
        else:
            source_digest = download_mri(row["source_url"], raw, row["source_etag"])
        check_volume(raw)
        result = {**row, "source_sha256": source_digest,
                  "resnet_array": f"resnet/{subject}.npy", "sfcn_array": f"sfcn/{subject}.npz"}
        resnet_path = folder / result["resnet_array"]
        if not valid_array(resnet_path, (80, 96, 80)):
            if row["dataset"] == "IXI":
                src = Path(ixi_prepared) / "arrays" / f"{row['subject_id'].split(':')[1]}.npy"
                if not src.exists():
                    raise FileNotFoundError(src)
                resnet_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = resnet_path.with_suffix(".npy.part")
                shutil.copy2(src, temporary)
                temporary.replace(resnet_path)
            else:
                item = {"subject_id": subject, "source_path": str(raw),
                        "age": row["age"], "site": row["site"]}
                prep = preprocess_one(item, folder / "resnet_work")
                if prep["qc_status"] == "failed":
                    raise RuntimeError(prep["error"])
                src = folder / "resnet_work" / prep["array_file"]
                resnet_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src, resnet_path)
        occupancy = resnet_center_occupancy(resnet_path)
        if occupancy < 0.10:
            raise ValueError(f"ResNet central anatomy absent: occupancy={occupancy:.4f}")
        sfcn_path = folder / result["sfcn_array"]
        if not valid_array(sfcn_path, SFCN_SHAPE):
            sfcn_work = folder / "sfcn_work" / subject
            sfcn_input = (ixi_sfcn_input(raw, sfcn_work / "ixi_input", cache / "fsl",
                                         row["subject_id"].split(":", 1)[1], Path(ixi_prepared))
                          if row["dataset"] == "IXI" else raw)
            x = sfcn_array(sfcn_input, sfcn_work, cache / "fsl")
            sfcn_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = sfcn_path.with_suffix(".part.npz")
            np.savez_compressed(temporary, x=x)
            temporary.replace(sfcn_path)
            save_qc(x, folder / "qc" / f"{subject}.png", subject)
        original_qc = folder / "resnet_work" / "qc" / f"{subject}.png"
        if original_qc.exists():
            destination = folder / "qc_resnet" / f"{subject}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(original_qc, destination)
        # Source digests are retained in the manifest; bulky generated staging
        # files are unnecessary after both arrays and montages are committed.
        shutil.rmtree(folder / "resnet_work" / "registration" / subject, ignore_errors=True)
        shutil.rmtree(folder / "sfcn_work" / subject, ignore_errors=True)
        if row["dataset"] != "IXI" and raw.exists():
            raw.unlink()
        result.update(qc_status="automatic_checks_passed", error="",
                      preprocess_seconds=time.time() - start)
    except Exception as exc:
        result = {**row, "source_sha256": "", "resnet_array": "", "sfcn_array": "",
                  "qc_status": "failed", "error": repr(exc),
                  "preprocess_seconds": time.time() - start}
    return result


def prepare(catalog: Path, source: str, output: Path, cache: Path,
            ixi_images: Path, ixi_prepared: Path, workers=1,
            shard_index=0, shard_count=1):
    if source not in {"IXI", "SALD", "NIMH", "DLBS"}:
        raise ValueError(source)
    catalog = Path(catalog)
    output = Path(output)
    cache = Path(cache).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bootstrap(cache, need_fsl=True)
    ensure_fast(cache)
    ensure_synthstrip(cache)
    df = pd.read_csv(catalog)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard specification")
    rows = df[df.dataset == source].sort_values("subject_id").to_dict("records")
    rows = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    if not rows:
        raise ValueError(f"No {source} records in catalogue")
    previous = output / "manifest.csv"
    done = {}
    if previous.exists():
        for row in pd.read_csv(previous).to_dict("records"):
            if row.get("qc_status") == "automatic_checks_passed":
                a = output / row["resnet_array"]
                b = output / row["sfcn_array"]
                if valid_array(a, (80, 96, 80)) and valid_array(b, SFCN_SHAPE):
                    done[row["subject_id"]] = row
    pending = [row for row in rows if row["subject_id"] not in done]
    results = list(done.values())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare_one, row, str(output), str(cache),
                               str(ixi_images), str(ixi_prepared)) for row in pending]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            pd.DataFrame(results).sort_values("subject_id").to_csv(previous, index=False)
            print("V3_PREPARE", source, len(results), "/", len(rows), result["subject_id"],
                  result["qc_status"], result.get("error", "")[:300], flush=True)
    passed = sum(x["qc_status"] == "automatic_checks_passed" for x in results)
    summary = {"source": source, "shard_index": shard_index, "shard_count": shard_count,
               "total": len(rows), "passed": passed,
               "failed": len(rows) - passed, "catalog_sha256": sha256(catalog),
               "synthstrip_sha256": SYNTHSTRIP_SHA256,
               "sfcn_preprocessing": "SynthStrip+FSL_FAST_bias+FLIRT_12dof_MNI1mm",
               "complete": len(results) == len(rows), "manual_qc": "pending"}
    write_json(output / "prepare_summary.json", summary)
    if passed / len(rows) < .90:
        raise RuntimeError(f"{source} passed only {passed}/{len(rows)} registration QC")
    return summary
