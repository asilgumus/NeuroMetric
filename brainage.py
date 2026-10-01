"""IXI adaptation of the released Westman CNN1; research use only.

Commands: prepare, train, predict. See docs/MODEL.md for the research log
and Kaggle orchestration.
The original model and transforms are downloaded from a fixed release tag.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile

TAG = 'Brainage-release-v1.0'
REPO = 'westman-neuroimaging-group/brainage-prediction-mri'
DEMO_URL = 'https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls'
DEMO_MIRROR = 'https://raw.githubusercontent.com/dfsp-spirit/brainnet/e60cef5bc5a77a5abcc8be6c9a9d6a7ec3c976de/inst/extdata/IXI.xls'
DEMO_SHA256 = '5b974cf58e9fe5101c91f594473fcd2b4c76ba2cf98fbd02d88174bd20a21c2b'
WEIGHTS_URL = f'https://github.com/{REPO}/releases/download/{TAG}/CNN1-weights-v1.0.zip'
SHAPE = (80, 96, 80)
SEED = 42
MEMBER_SHA256 = {
    0: '1cedb9dbe5e053dd0b6afdae793c96cacb4e5729d5edba95e1fd26b8091df5e9',
    1: '7af6af95f1e87489a11845736918416f28781d0807b7522f0ba9c7acce9ad2da',
    2: 'e18fb16937e6993d89cbb7d66659b75d54f7053379b08de4e0a0ab6e47beba4b',
    3: '49f57fd14f5dae17aa80cf49ca40b30469c9cc589115f123add1cde89174e1c7',
    4: 'a9ee60c76316f653e58d46645b57ebb67a1f27770a71bd2e0257d6f6e92ce193',
}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def run(cmd):
    print('RUN', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_suffix(path.suffix + '.part')
        run(['curl', '-fL', '--retry', '3', '--connect-timeout', '30', '-o', part, url])
        part.replace(path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def get_demographics(output):
    path = Path(output) / 'IXI.xls'
    try:
        download(DEMO_URL, path)
        source = DEMO_URL
    except subprocess.CalledProcessError:
        print('Official IXI endpoint unavailable; using pinned copy of original spreadsheet', flush=True)
        download(DEMO_MIRROR, path)
        source = DEMO_MIRROR
        if sha256(path) != DEMO_SHA256:
            raise ValueError('Demographics mirror checksum mismatch')
    return path, source


def clean_demographics(demo):
    """Collapse equal-age duplicates; exclude IDs with conflicting or missing ages."""
    import numpy as np
    import pandas as pd
    records, excluded = [], []
    for uid, group in demo.groupby('IXI_ID'):
        ages = pd.to_numeric(group['AGE'], errors='coerce')
        finite = ages[np.isfinite(ages)].unique()
        if len(finite) != 1:
            excluded.append({'subject_id': f'IXI{int(uid):03d}',
                             'reason': 'conflicting_ages' if len(finite) > 1 else 'missing_age'})
            continue
        row = group.iloc[0].copy()
        row['AGE'] = float(finite[0])
        records.append(row)
    return pd.DataFrame(records), excluded


def bootstrap(cache, need_fsl=False, member=0):
    """Fetch attributable upstream code and install the original FSL dependencies."""
    cache = Path(cache).resolve()
    source = cache / 'upstream'
    manifest = {'upstream_repo': REPO, 'tag': TAG, 'files': {}}
    for name in ['model/model.py', 'model/modules.py', 'transforms/transforms.py',
                 'transforms/load_transform.py', 'utils/misc.py', 'LICENSE']:
        path = download(f'https://raw.githubusercontent.com/{REPO}/{TAG}/{name}', source / name)
        manifest['files'][name] = sha256(path)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    archive = download(WEIGHTS_URL, cache / 'CNN1.zip')
    weights_dir = cache / 'weights'
    if not list(weights_dir.rglob('*.pth')):
        with zipfile.ZipFile(archive) as z:
            for archive_member in z.infolist():
                if not (weights_dir / archive_member.filename).resolve().is_relative_to(weights_dir.resolve()):
                    raise ValueError('Unsafe archive path')
            z.extractall(weights_dir)
    weights = sorted(weights_dir.rglob('*.pth'))
    if member not in MEMBER_SHA256:
        raise ValueError(f'CNN1 member must be one of {sorted(MEMBER_SHA256)}')
    # CNN1 release bundles five independently trained members. The selected
    # member is configuration, never chosen using test performance.
    member_name = f'ResNet3D_3x_{member}.pth'
    weights = [p for p in weights if p.name == member_name]
    if len(weights) != 1:
        raise ValueError(f'Expected CNN1 member {member_name}')
    manifest['ensemble_member'] = member_name
    manifest['weights_sha256'] = sha256(weights[0])
    if manifest['weights_sha256'] != MEMBER_SHA256[member]:
        raise ValueError('Released CNN1 member checksum mismatch')
    write_json(cache / 'provenance.json', manifest)
    if need_fsl:
        fsl = cache / 'fsl'
        os.environ['MAMBA_ROOT_PREFIX'] = str(cache / 'mamba-root')
        system_dc = shutil.which('dc')
        if not (fsl / 'bin/flirt').exists() or (not (fsl / 'bin/dc').exists() and not system_dc):
            archive = download('https://micro.mamba.pm/api/micromamba/linux-64/latest', cache / 'micromamba.tar.bz2')
            run(['tar', '-xjf', archive, '-C', cache, 'bin/micromamba'])
            run([cache / 'bin/micromamba', 'create', '-y', '-p', fsl,
                 '-c', 'https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/',
                 '-c', 'conda-forge', 'fsl-flirt', 'fsl-bet2', 'fsl-avwutils', 'fsl-data_standard', 'bc'])
        os.environ['FSLDIR'] = str(fsl)
        os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
        os.environ['PATH'] = str(fsl / 'bin') + ':' + os.environ['PATH']
        if not (fsl / 'data/standard/MNI152_T1_1mm.nii.gz').exists():
            raise FileNotFoundError('FSL MNI152 template missing')
        if not shutil.which('dc'):
            raise FileNotFoundError('FSL BET dependency dc is missing')
    return weights[0], manifest


def seed_all():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(weights, device):
    import torch
    from model.model import ResNet3D
    model = ResNet3D(SHAPE, width_f=3)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    state = state.get('model_state', state)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    return model.to(device).eval()


def check_volume(path):
    import nibabel as nib
    import numpy as np
    image = nib.load(str(path))
    if len(image.shape) != 3 or min(image.shape) < 32:
        raise ValueError(f'Expected a 3D T1 volume: {image.shape}')
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all() or float(np.ptp(data)) <= 0:
        raise ValueError('Non-finite or constant MRI intensity')
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError('Invalid NIfTI affine')


def transform_registered(path):
    import torch
    from transforms.load_transform import load_transforms
    check_volume(path)
    x = load_transforms({'img_dim': [160, 192, 160]}, random_chance=0)(str(path))
    if tuple(x.shape) != SHAPE or not torch.isfinite(x).all():
        raise ValueError(f'Invalid transformed MRI: {tuple(x.shape)}')
    return x


def make_manifest(images_root, demographics):
    import numpy as np
    import pandas as pd
    demo = pd.read_excel(demographics)
    if not {'IXI_ID', 'AGE'} <= set(demo.columns):
        raise ValueError('Official IXI_ID and AGE columns required')
    demo, excluded = clean_demographics(demo)
    records, seen = [], set()
    for path in sorted(Path(images_root).rglob('*.nii*')):
        if not path.is_file():
            continue
        match = re.search(r'IXI(\d+)-(Guys|HH|IOP)', str(path))
        if not match:
            excluded.append({'path': str(path), 'reason': 'unrecognized_id_or_site'})
            continue
        number, site = int(match[1]), match[2]
        uid = f'IXI{number:03d}'
        if uid in seen:
            raise ValueError(f'Duplicate MRI for participant {uid}')
        seen.add(uid)
        rows = demo.loc[demo['IXI_ID'] == number]
        if len(rows) != 1 or not np.isfinite(rows.iloc[0]['AGE']):
            excluded.append({'path': str(path), 'reason': 'missing_age'})
            continue
        age = float(rows.iloc[0]['AGE'])
        if not 18 <= age <= 100:
            excluded.append({'path': str(path), 'reason': 'outside_adult_scope'})
            continue
        records.append(dict(subject_id=uid, age=age, site=site, source_path=str(path)))
    if len(records) < 40:
        raise ValueError(f'Too few usable participants: {len(records)}')
    return pd.DataFrame(records), excluded


def split_participants(df):
    """Split once, before fitting; never split slices or visits independently."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    if df.subject_id.duplicated().any():
        raise ValueError('Repeated participant')
    df = df.sort_values('subject_id').reset_index(drop=True).copy()
    age_group = pd.qcut(df.age, q=3, labels=False, duplicates='drop').astype(str)
    strata = df.site.astype(str) + ':' + age_group
    strategy = 'site_and_age_tertile'
    if strata.value_counts().min() < 8:
        strata, strategy = df.site, 'site_only_due_to_sparse_cells'
    if strata.value_counts().min() < 8:
        raise ValueError('Too few participants per site for stratified split')
    train_idx, other = train_test_split(df.index, test_size=.30, random_state=SEED, stratify=strata)
    val_idx, test_idx = train_test_split(other, test_size=.50, random_state=SEED, stratify=strata.loc[other])
    df['split'] = ''
    for label, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        df.loc[idx, 'split'] = label
    return df, strategy


def preprocess_one(record, output):
    import numpy as np
    from utils.misc import native_to_tal_fsl
    output = Path(output)
    uid = record['subject_id']
    folder = output / 'registration' / uid
    folder.mkdir(parents=True, exist_ok=True)
    before = time.time()
    try:
        check_volume(record['source_path'])
        with open(folder / 'fsl.log', 'w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            native_to_tal_fsl(record['source_path'], output_folder=str(folder), guid=uid)
        registered = folder / f'{uid}_mni_dof_6.nii'
        x = transform_registered(registered).numpy()
        array = output / 'arrays' / f'{uid}.npy'
        array.parent.mkdir(exist_ok=True)
        np.save(array, x.astype(np.float32))
        # Keep a compact QC montage for every participant, in canonical orientation.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(8, 3))
        for ax, plane in zip(axes, [x[40, :, :], x[:, 48, :], x[:, :, 40]]):
            ax.imshow(np.rot90(plane), cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
        fig.suptitle(f'{uid} | registration QC (not an attention map)')
        (output / 'qc').mkdir(exist_ok=True)
        fig.savefig(output / 'qc' / f'{uid}.png', dpi=90)
        plt.close(fig)
        # This is a generated intermediate; the source MRI remains untouched.
        registered.unlink()
        return {**record, 'array_file': f'arrays/{uid}.npy', 'qc_status': 'automatic_checks_passed',
                'preprocess_seconds': time.time()-before, 'error': ''}
    except Exception as exc:
        return {**record, 'array_file': '', 'qc_status': 'failed',
                'preprocess_seconds': time.time()-before, 'error': repr(exc)}


def prepare(args):
    import pandas as pd
    seed_all()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _, provenance = bootstrap(args.cache, need_fsl=True)
    demographics, demographics_source = get_demographics(output)
    df, excluded = make_manifest(args.images, demographics)
    df, strategy = split_participants(df)
    df.to_csv(output / 'splits_before_qc.csv', index=False)
    write_json(output / 'exclusions.json', excluded)
    write_json(output / 'provenance.json', {**provenance, 'demographics_url': demographics_source,
        'official_demographics_url': DEMO_URL,
        'demographics_sha256': sha256(demographics), 'image_source': args.images,
        'split_strategy': strategy, 'seed': SEED})
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(preprocess_one, row, str(output)) for row in df.to_dict('records')]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            pd.DataFrame(results).sort_values('subject_id').to_csv(output / 'manifest.csv', index=False)
            print('PREPARE', len(results), '/', len(df), result['subject_id'], result['qc_status'], result['error'], flush=True)
    passed = sum(r['qc_status'] != 'failed' for r in results)
    write_json(output / 'prepare_summary.json', {'total': len(results), 'passed': passed,
        'failed': len(results)-passed, 'manual_qc': 'pending', 'complete': True})
    if passed / len(results) < .95:
        raise RuntimeError('More than 5% preprocessing failures; inspect before training')


def metrics(actual, predicted):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if len(actual) == 0 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Empty or non-finite evaluation')
    return {'n': len(actual), 'mae': float(mean_absolute_error(actual, predicted)),
            'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
            'r2': float(r2_score(actual, predicted)) if len(actual) > 1 and np.ptp(actual) > 0 else None,
            'mean_gap': float(np.mean(predicted-actual))}


def make_loader(frame, prepared, batch=2, shuffle=False):
    import numpy as np
    import torch
    class MRIs(torch.utils.data.Dataset):
        def __len__(self):
            return len(frame)
        def __getitem__(self, idx):
            row = frame.iloc[idx]
            x = np.load(Path(prepared) / row.array_file, allow_pickle=False)
            if x.shape != SHAPE or not np.isfinite(x).all():
                raise ValueError(f'Invalid array for {row.subject_id}')
            return torch.from_numpy(x.astype(np.float32)), torch.tensor(row.age, dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    return torch.utils.data.DataLoader(MRIs(), batch_size=batch, shuffle=shuffle,
                                      num_workers=0, generator=generator)


def evaluate(model, loader, device):
    import torch
    actual, predicted = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))[0].flatten().cpu()
            actual.extend(y.tolist())
            predicted.extend(pred.tolist())
    return metrics(actual, predicted), predicted


def freeze_for_stage(model, stage):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.fc1.parameters():
        p.requires_grad_(True)
    if stage == 'last_block':
        for p in model.features.features[7].parameters():
            p.requires_grad_(True)
    # eval keeps BatchNorm statistics fixed; gradients still flow.
    model.eval()


def train(args):
    import numpy as np
    import pandas as pd
    import torch
    seed_all()
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU unavailable; refusing an unintended full CPU training run')
    device = torch.device('cuda')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    weights, provenance = bootstrap(args.cache, member=args.member)
    prepared = Path(args.prepared)
    if not json.loads((prepared / 'prepare_summary.json').read_text())['complete']:
        raise ValueError('Preprocessing incomplete')
    df = pd.read_csv(prepared / 'manifest.csv').sort_values('subject_id')
    if df.subject_id.duplicated().any() or set(df.split) != {'train', 'val', 'test'}:
        raise ValueError('Invalid participant split')
    df = df[df.qc_status != 'failed'].reset_index(drop=True)
    frames = {s: df[df.split == s].reset_index(drop=True) for s in ['train', 'val', 'test']}
    loaders = {s: make_loader(f, prepared, args.batch) for s, f in frames.items()}
    training_loader = make_loader(frames['train'], prepared, args.batch, shuffle=True)
    model = load_model(weights, device)
    baseline_val, _ = evaluate(model, loaders['val'], device)
    print('BASELINE_VALIDATION', baseline_val, flush=True)
    write_json(output / 'config.json', {**vars(args), 'device': torch.cuda.get_device_name(),
        'seed': SEED, 'provenance': provenance, 'torch': str(torch.__version__),
        'prepared_manifest_sha256': sha256(prepared / 'manifest.csv')})
    (output / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    best_loss = baseline_val['mae']
    best_path = output / 'best_model.pt'
    torch.save({'model_state': model.state_dict(), 'stage': 'pretrained', 'epoch': 0,
                'validation_mae': best_loss, 'provenance': provenance}, best_path)
    history = []
    adapted_loss = float('inf')
    resume = None
    if args.resume:
        resume_path = Path(args.resume)
        resume = torch.load(resume_path, map_location=device, weights_only=True)
        if resume['manifest_sha256'] != sha256(prepared / 'manifest.csv'):
            raise ValueError('Resume checkpoint belongs to a different participant manifest')
        for filename in ['best_model.pt', 'best_finetuned.pt']:
            src = resume_path.parent / filename
            if src.exists() and src.resolve() != (output / filename).resolve():
                shutil.copy2(src, output / filename)
        model.load_state_dict(resume['model_state'])
        best_loss, adapted_loss = resume['best_loss'], resume['adapted_loss']
        history = resume['history']
        torch.set_rng_state(resume['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in resume['cuda_rng']])
        training_loader.generator.set_state(resume['loader_rng'].cpu())
    # Stage 2 is run only if head adaptation improves validation by >= 0.1 years.
    for stage, epochs in [('head', args.head_epochs), ('last_block', args.block_epochs)]:
        if resume and resume['stage'] == 'last_block' and stage == 'head':
            continue
        if stage == 'last_block':
            if baseline_val['mae'] - best_loss < .1:
                print('SKIP_LAST_BLOCK: head improvement below 0.1 years', flush=True)
                break
            if not (resume and resume['stage'] == stage):
                model = load_model(best_path, device)
        freeze_for_stage(model, stage)
        groups = [{'params': list(model.fc1.parameters()), 'lr': 1e-4}]
        if stage == 'last_block':
            groups.append({'params': list(model.features.features[7].parameters()), 'lr': 1e-5})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        stale = 0
        start_epoch = 1
        if resume and resume['stage'] == stage:
            optimizer.load_state_dict(resume['optimizer_state'])
            scaler.load_state_dict(resume['scaler_state'])
            start_epoch, stale = resume['epoch'] + 1, resume['stale']
        for epoch in range(start_epoch, epochs+1):
            if stale >= 5:
                break
            model.eval()
            optimizer.zero_grad(set_to_none=True)
            running, count = 0., 0
            for step, (x, y) in enumerate(training_loader):
                x, y = x.to(device), y.to(device)
                window_start = (step // args.accumulation) * args.accumulation
                window_size = min(args.accumulation, len(training_loader)-window_start)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(x)[0].flatten()
                    loss = (pred-y).abs().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                scaler.scale(loss / window_size).backward()
                if (step+1) % args.accumulation == 0 or step+1 == len(training_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running += loss.item()*len(y)
                count += len(y)
            val, _ = evaluate(model, loaders['val'], device)
            history.append({'stage': stage, 'epoch': epoch, 'train_mae': running/count, 'val_mae': val['mae']})
            pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
            improved = val['mae'] < best_loss
            adapted_improved = val['mae'] < adapted_loss
            if improved:
                best_loss, stale = val['mae'], 0
            else:
                stale += 1
            if adapted_improved:
                adapted_loss = val['mae']
            state = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                     'scaler_state': scaler.state_dict(), 'stage': stage, 'epoch': epoch,
                     'validation_mae': val['mae'], 'provenance': provenance,
                     'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                     'loader_rng': training_loader.generator.get_state(), 'history': history,
                     'best_loss': best_loss, 'adapted_loss': adapted_loss, 'stale': stale,
                     'manifest_sha256': sha256(prepared / 'manifest.csv')}
            torch.save(state, output / 'last_checkpoint.pt')
            print('EPOCH', history[-1], flush=True)
            if improved:
                torch.save(state, best_path)
            if adapted_improved:
                torch.save(state, output / 'best_finetuned.pt')
            if stale >= 5:
                break
    # Lock selection before opening test data for inference.
    selection = torch.load(best_path, map_location='cpu', weights_only=True)
    write_json(output / 'selection.json', {'stage': selection['stage'], 'epoch': selection['epoch'],
        'validation_mae': selection['validation_mae'], 'test_used_for_selection': False})
    candidates = {'pretrained': load_model(weights, device),
                  'finetuned': load_model(output / 'best_finetuned.pt', device),
                  'selected': load_model(best_path, device)}
    report, predictions = {}, []
    reference_age = float(frames['train'].age.mean())
    for split in ['val', 'test']:
        frame = frames[split].copy()
        frame['mean_age_reference'] = reference_age
        report[split] = {'mean_age_reference': metrics(frame.age, np.repeat(reference_age, len(frame)))}
        for name, candidate in candidates.items():
            scores, values = evaluate(candidate, loaders[split], device)
            report[split][name] = scores
            frame[name] = values
        frame['brain_age_gap'] = frame.selected-frame.age
        predictions.append(frame)
    all_predictions = pd.concat(predictions)
    all_predictions.to_csv(output / 'predictions.csv', index=False)
    groups = []
    test = all_predictions[all_predictions.split == 'test'].copy()
    test['age_group'] = pd.cut(test.age, [18, 30, 40, 50, 60, 70, 80, 101], right=False).astype(str)
    for column in ['site', 'age_group']:
        for group, rows in test.groupby(column):
            for name in candidates:
                groups.append({'grouping': column, 'group': group, 'model': name, **metrics(rows.age, rows[name])})
    pd.DataFrame(groups).to_csv(output / 'subgroup_metrics.csv', index=False)
    write_json(output / 'metrics.json', report)
    plots_and_report(output, test, report, selection, len(frames['train']))
    print('TRAINING_COMPLETE', report, flush=True)


def plots_and_report(output, test, report, selection, n_train):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ['pretrained', 'finetuned', 'selected']:
        axes[0].scatter(test.age, test[name], s=12, alpha=.6, label=name)
        axes[1].scatter(test.age, test[name]-test.age, s=12, alpha=.6, label=name)
    limits = [test.age.min(), test.age.max()]
    axes[0].plot(limits, limits, 'k--')
    axes[0].set(xlabel='Chronological age', ylabel='Predicted age')
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set(xlabel='Chronological age', ylabel='Brain age gap (years)')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'test_evaluation.png', dpi=160)
    plt.close(fig)
    before, after = report['test']['pretrained']['mae'], report['test']['selected']['mae']
    content = f'''# BrainAGE IXI research experiment

Training: {n_train} participants. Test: {len(test)} participants.
Pretrained CNN1 member 0 test MAE: {before:.3f} years (five-member ensemble not used).
Validation-selected model test MAE: {after:.3f} years.
Test MAE change (positive = improvement): {before-after:.3f} years.
Selected stage: {selection['stage']}, epoch: {selection['epoch']}.
Fine-tuned model test MAE: {report['test']['finetuned']['mae']:.3f} years.

The model was selected using the validation set only. Test results did not
influence model selection. If fine-tuning did not improve on validation, the
selected model is the original pretrained model. The result belongs to a
participant split inside IXI; it is not an independent hospital validation.
Image quality control is limited to automatic checks and sampled visual
inspection. The age gap is raw prediction minus chronological age; no age-bias
correction was applied. Regional age, healthy-control percentile, disease
diagnosis or clinical recommendation were not produced.

Sources: https://brain-development.org/ixi-dataset/ (IXI, CC BY-SA 3.0)
and https://github.com/{REPO} (model code, MIT).
The CNN1 model training report lists ADNI, AIBL, GENIC and UK Biobank; IXI is
not in that list. We have no access to per-participant original training records.
'''
    (output / 'REPORT.md').write_text(content)


def predict(args):
    import torch
    seed_all()
    if not math.isfinite(args.age) or not 18 <= args.age <= 100:
        raise ValueError('Chronological age must be a finite adult age (18–100)')
    _, provenance = bootstrap(args.cache, need_fsl=not args.registered)
    path = Path(args.input).resolve()
    check_volume(path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.registered:
        from utils.misc import native_to_tal_fsl
        native_to_tal_fsl(str(path), output_folder=str(output), guid='input', force_new_transform=True)
        path = output / 'input_mni_dof_6.nii'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = transform_registered(path)
    checkpoints = [Path(p) for p in args.checkpoint]
    values = []
    with torch.no_grad():
        for checkpoint in checkpoints:
            model = load_model(checkpoint, device)
            values.append(float(model(x.unsqueeze(0).to(device))[0].item()))
    value = sum(values) / len(values)
    if not math.isfinite(value):
        raise ValueError('Non-finite prediction')
    result = {'subject_id': args.subject_id, 'chronological_age': args.age,
              'predicted_brain_age': value, 'brain_age_gap': value-args.age,
              'model_version': hashlib.sha256(''.join(sha256(p) for p in checkpoints).encode()).hexdigest(),
              'ensemble_size': len(checkpoints), 'member_predictions': values,
              'qc_status': 'automatic_checks_passed',
              'research_only': True}
    write_json(output / 'prediction.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--images', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    p = sub.add_parser('train')
    p.add_argument('--prepared', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--head-epochs', type=int, default=5)
    p.add_argument('--block-epochs', type=int, default=25)
    p.add_argument('--resume', help='last_checkpoint.pt; best checkpoints must be beside it')
    p.add_argument('--member', type=int, choices=range(5), default=0,
                   help='CNN1 pretrained ensemble member to fine-tune')
    p = sub.add_parser('predict')
    p.add_argument('--input', required=True)
    p.add_argument('--checkpoint', required=True, nargs='+',
                   help='One checkpoint, or several checkpoints averaged as an ensemble')
    p.add_argument('--age', type=float, required=True)
    p.add_argument('--subject-id', default='anonymous')
    p.add_argument('--output', required=True)
    p.add_argument('--registered', action='store_true', help='Input already registered by the original FSL pipeline')
    for p in sub.choices.values():
        p.add_argument('--cache', default='.cache/brainage')
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
