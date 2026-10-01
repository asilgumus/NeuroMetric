"""Real-MRI smoke test; no test-set performance claims from this pilot."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path('/kaggle/working')
CACHE = Path('/tmp/brainage')
CACHE.mkdir(exist_ok=True)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'


def run(command):
    print('RUN', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print('DOWNLOAD', url, flush=True)
    run(['curl', '-fL', '--retry', '3', '--connect-timeout', '30', '-o', path, url])
    return path


def main():
    if not os.environ.get('BRAINAGE_BOOTSTRAPPED'):
        run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy==1.26.4',
             'pandas==2.2.3', 'scipy==1.14.1', 'scikit-learn==1.5.2',
             'nibabel==5.3.2', 'nipype==1.10.0', 'xlrd==2.0.1', 'matplotlib==3.9.2'])
        os.environ['BRAINAGE_BOOTSTRAPPED'] = '1'
        os.execv(sys.executable, [sys.executable, __file__])
    import torch
    print('ENV', sys.version, torch.__version__, torch.cuda.is_available(), flush=True)
    run(['nvidia-smi'])
    import numpy as np
    import pandas as pd
    import nibabel as nib
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    assert torch.cuda.is_available(), 'GPU is required for this pilot'

    # Freeze upstream source to the release tag; retain its license and checksum.
    source = CACHE / 'upstream'
    tag = 'Brainage-release-v1.0'
    base = f'https://raw.githubusercontent.com/westman-neuroimaging-group/brainage-prediction-mri/{tag}'
    for name in ['model/model.py', 'model/modules.py', 'transforms/transforms.py',
                 'transforms/load_transform.py', 'utils/misc.py', 'LICENSE']:
        download(f'{base}/{name}', source / name)
    sys.path.insert(0, str(source))
    weights_zip = download('https://github.com/westman-neuroimaging-group/brainage-prediction-mri/releases/download/Brainage-release-v1.0/CNN1-weights-v1.0.zip', CACHE / 'weights.zip')
    with zipfile.ZipFile(weights_zip) as z:
        z.extractall(CACHE / 'weights')
    weights = sorted((CACHE / 'weights').rglob('ResNet3D_3x_0.pth'))
    assert len(weights) == 1, weights
    print('WEIGHTS', weights, flush=True)

    # Install only the FSL components used by the original preprocessing.
    archive = download('https://micro.mamba.pm/api/micromamba/linux-64/latest', CACHE / 'micromamba.tar.bz2')
    run(['tar', '-xjf', archive, '-C', CACHE, 'bin/micromamba'])
    fsl = CACHE / 'fsl'
    os.environ['MAMBA_ROOT_PREFIX'] = str(CACHE / 'mamba-root')
    run([CACHE / 'bin/micromamba', 'create', '-y', '-p', fsl,
         '-c', 'https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/',
         '-c', 'conda-forge', 'fsl-flirt', 'fsl-bet2', 'fsl-avwutils', 'fsl-data_standard', 'bc'])
    os.environ['FSLDIR'] = str(fsl)
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
    os.environ['PATH'] = str(fsl / 'bin') + ':' + os.environ['PATH']
    assert (fsl / 'data/standard/MNI152_T1_1mm.nii.gz').exists() and (fsl / 'bin/dc').exists()

    demographics = download('https://raw.githubusercontent.com/dfsp-spirit/brainnet/e60cef5bc5a77a5abcc8be6c9a9d6a7ec3c976de/inst/extdata/IXI.xls', ROOT / 'IXI.xls')
    assert hashlib.sha256(demographics.read_bytes()).hexdigest() == '5b974cf58e9fe5101c91f594473fcd2b4c76ba2cf98fbd02d88174bd20a21c2b'
    demo = pd.read_excel(demographics)
    print('DEMOGRAPHICS', demo.columns.tolist(), demo.shape, flush=True)
    images = sorted(p for p in Path('/kaggle/input').rglob('*.nii*') if p.is_file())
    print('IMAGES', len(images), [str(p) for p in images[:2]], flush=True)
    from model.model import ResNet3D
    from transforms.load_transform import load_transforms
    from utils.misc import native_to_tal_fsl
    model = ResNet3D(np.array([160, 192, 160]) // 2, width_f=3)
    state = torch.load(weights[0], map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    transform = load_transforms({'img_dim': [160, 192, 160]}, random_chance=0)
    out = ROOT / 'pilot'
    out.mkdir(exist_ok=True)
    records, tensors = [], []
    for path in images:
        import re
        match = re.search(r'IXI(\d+)', str(path))
        if not match:
            continue
        number = int(match[1])
        row = demo.loc[demo['IXI_ID'] == number]
        if len(row) != 1 or not np.isfinite(row.iloc[0]['AGE']):
            continue
        uid = f'IXI{number:03d}'
        age = float(row.iloc[0]['AGE'])
        before = time.time()
        native_to_tal_fsl(str(path), output_folder=str(out), guid=uid)
        registered = out / f'{uid}_mni_dof_6.nii'
        x = transform(str(registered))
        assert x.shape == (80, 96, 80) and torch.isfinite(x).all()
        with torch.no_grad():
            prediction = float(model(x.unsqueeze(0).cuda())[0].item())
        assert np.isfinite(prediction)
        records.append(dict(subject_id=uid, chronological_age=age,
                            predicted_brain_age=prediction, brain_age_gap=prediction-age,
                            seconds=time.time()-before))
        tensors.append(x)
        print('PREDICTION', records[-1], flush=True)
        if len(records) == 8:
            break
    assert len(records) == 8
    pd.DataFrame(records).to_csv(ROOT / 'pilot_predictions.csv', index=False)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.fc1.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.fc1.parameters(), lr=1e-4)
    batch = torch.stack(tensors[:2]).cuda()
    ages = torch.tensor([r['chronological_age'] for r in records[:2]], device='cuda')
    old = model.fc1[0].weight.detach().clone()
    loss = (model(batch)[0].flatten()-ages).abs().mean()
    loss.backward()
    opt.step()
    assert not torch.equal(old, model.fc1[0].weight)
    checkpoint = ROOT / 'pilot_checkpoint.pt'
    torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    summary = {'status': 'passed', 'n': len(records), 'loss_before_step': float(loss.item()),
               'weights_sha256': hashlib.sha256(weights[0].read_bytes()).hexdigest(),
               'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
               'note': 'Smoke test only; these predictions are not a held-out evaluation.'}
    (ROOT / 'pilot_summary.json').write_text(json.dumps(summary, indent=2))
    print('SUCCESS', summary, flush=True)


if __name__ == '__main__':
    main()
