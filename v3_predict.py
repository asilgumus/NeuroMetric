"""Research-only V3 inference on one new, de-identified T1-weighted NIfTI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from brainage import bootstrap, check_volume, load_model, preprocess_one, sha256, write_json
from v3_model import SFCN, expected_age
from v3_prepare import ensure_fast, ensure_synthstrip, ixi_sfcn_input, save_qc, sfcn_array
from v3_train import normalize_sfcn_array


COMPONENTS = {"v2", "sfcn", "resnet"}


def selected_components(selection: dict) -> tuple[str, ...]:
    names = tuple(selection.get("candidate", "").split("+"))
    if not names or len(set(names)) != len(names) or not set(names) <= COMPONENTS:
        raise ValueError("Invalid validation-selected V3 candidate")
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("Selection must be validation-only")
    return names


def checked_checkpoint(path: Path | None, expected_hash: str, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Missing {label} checkpoint: {path}")
    if sha256(path) != expected_hash:
        raise ValueError(f"{label} checkpoint differs from locked selection")
    return path


def validate_checkpoint_set(names: tuple[str, ...], selection: dict, args):
    expected = selection["checkpoints"]
    if "sfcn" in names:
        checked_checkpoint(args.sfcn_checkpoint, expected["sfcn"], "SFCN")
    if "resnet" in names:
        checked_checkpoint(args.resnet_checkpoint, expected["resnet"], "ResNet")
    if "v2" in names:
        if args.v2_dir is None or len(expected["v2"]) != 4:
            raise ValueError("Four validation-selected V2 members are required")
        for index in range(1, 5):
            checked_checkpoint(args.v2_dir / f"member_{index}.pt",
                               expected["v2"][index - 1], f"V2 member {index}")


def prepare_mri(mri: Path, cache: Path, work: Path, qc_output: Path):
    check_volume(mri)
    bootstrap(cache, need_fsl=True, member=1)
    ensure_fast(cache)
    ensure_synthstrip(cache)
    resnet_root = work / "resnet"
    result = preprocess_one({"subject_id": "prediction", "source_path": str(mri)},
                            resnet_root)
    if result["qc_status"] != "automatic_checks_passed":
        raise RuntimeError(f"ResNet preprocessing failed: {result['error']}")
    resnet = np.load(resnet_root / result["array_file"], allow_pickle=False)
    initial = ixi_sfcn_input(mri, work / "sfcn_initial", cache / "fsl",
                             "prediction", resnet_root)
    sfcn = sfcn_array(initial, work / "sfcn", cache / "fsl")
    qc_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resnet_root / "qc/prediction.png", qc_output / "qc_resnet.png")
    save_qc(sfcn, qc_output / "qc_sfcn.png", "prediction | SFCN registration QC")
    return np.asarray(resnet, dtype=np.float32), normalize_sfcn_array(sfcn)


def infer_components(names: tuple[str, ...], selection: dict, resnet: np.ndarray,
                     sfcn: np.ndarray, args, device: torch.device) -> dict[str, float]:
    expected = selection["checkpoints"]
    predictions = {}
    x_resnet = torch.from_numpy(resnet[None].copy()).to(device)
    with torch.inference_mode():
        if "resnet" in names:
            checkpoint = checked_checkpoint(args.resnet_checkpoint, expected["resnet"], "ResNet")
            model = load_model(checkpoint, device)
            predictions["resnet"] = float(model(x_resnet)[0].flatten()[0].float().cpu())
            del model
        if "v2" in names:
            paths = [args.v2_dir / f"member_{index}.pt" for index in (1, 2, 3, 4)] \
                    if args.v2_dir else []
            if len(paths) != 4 or len(expected["v2"]) != 4:
                raise ValueError("Four validation-selected V2 members are required")
            members = []
            for index, path in enumerate(paths):
                checkpoint = checked_checkpoint(path, expected["v2"][index], f"V2 member {index + 1}")
                model = load_model(checkpoint, device)
                members.append(float(model(x_resnet)[0].flatten()[0].float().cpu()))
                del model
            predictions["v2"] = float(np.mean(members))
        if "sfcn" in names:
            checkpoint = checked_checkpoint(args.sfcn_checkpoint, expected["sfcn"], "SFCN")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state.get("model") != "sfcn":
                raise ValueError("Wrong SFCN checkpoint architecture")
            model = SFCN().to(device).eval()
            model.load_state_dict(state["model_state"], strict=True)
            x_sfcn = torch.from_numpy(sfcn[None, None].copy()).to(device)
            predictions["sfcn"] = float(expected_age(model(x_sfcn)).float().cpu()[0])
    return predictions


def predict(args):
    selection = json.loads(args.selection.read_text())
    names = selected_components(selection)
    if not np.isfinite(args.age) or not 18 <= args.age <= 100:
        raise ValueError("Chronological age must be finite and between 18 and 100")
    validate_checkpoint_set(names, selection, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.cache.resolve()
    with tempfile.TemporaryDirectory(prefix="brainage-v3-") as folder:
        resnet, sfcn = prepare_mri(args.mri.resolve(), cache, Path(folder), args.output.parent)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        values = infer_components(names, selection, resnet, sfcn, args, device)
    prediction = float(np.mean([values[name] for name in names]))
    if not np.isfinite(prediction) or not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite model prediction")
    report = {"research_only": True, "clinical_diagnosis": False,
              "chronological_age": float(args.age), "predicted_brain_age": prediction,
              "brain_age_gap": prediction - float(args.age),
              "selected_model": selection["candidate"], "component_predictions": values,
              "input_sha256": sha256(args.mri), "selection_sha256": sha256(args.selection),
              "checkpoint_hashes": {name: selection["checkpoints"][name] for name in names},
              "qc_images": ["qc_resnet.png", "qc_sfcn.png"]}
    write_json(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mri", type=Path, required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sfcn-checkpoint", type=Path)
    parser.add_argument("--resnet-checkpoint", type=Path)
    parser.add_argument("--v2-dir", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".cache/brainage"))
    parser.add_argument("--output", type=Path, required=True)
    report = predict(parser.parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
