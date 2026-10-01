# BrainAge — MRI-based brain age prediction (research prototype)

A 3D CNN pipeline that predicts brain age from T1 MR images, plus the public
clinical-reading preview site (**NeuroMetric**). This is a research prototype,
not a clinical device or diagnostic system.

Two components:

1. **Model pipeline** — transfer-learning CNNs trained on IXI + SALD + NIMH
   healthy-control T1 scans and tested independently on DLBS wave-1
   (Westman *CNN1* 3D ResNet and a fine-tuned UK Biobank-pretrained 3D SFCN).
2. **NeuroMetric preview site** — a static clinical-reading dashboard built
   from real model outputs and Grad-CAM explanations. It does not upload the
   visitor's MRI, and it runs no analysis (`DATA_SOURCES.md`).

## Repository layout

```
.
├── brainage.py            # CLI: prepare / train / predict
├── v3_*.py                # V3 pipeline (prepare, merge, train, evaluate, test, predict)
├── oasis_test.py          # OASIS-side test / Grad-CAM generation
├── kaggle/                # generated Kaggle kernel sources (kernel-metadata.json + run.py)
├── tools/                 # kernel and site builders, QC tooling
├── tests/                 # pytest (pipeline + site content tests)
├── systemd/               # local Kaggle queue service (optional)
├── docs/                  # MODEL.md (experiment log), DELIVERY.md, working notes
├── index.html             # NeuroMetric landing page (static)
├── report.html            # assessment dashboard (?case=ixi361 …)
├── longitudinal.html      # OAS2_0001 follow-up example
├── resources.html         # model card / release notes / data policy
├── assets/                # site visuals, sample report PDF, Grad-CAM PNGs
├── DATA_SOURCES.md        # data attribution and demo scope
└── robots.txt
```

`data/` (2.6 GB images) and `artifacts/` (4.4 GB checkpoints/results) are not
committed; they can be regenerated with the catalogue and prepare commands.
The only exception is `data/dkt_lobe_groups.json` (FastSurfer DKT→lobe grouping
configuration, used by tests and tooling).

## Setup and running

### Model

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Prediction for a new T1 NIfTI with a trained checkpoint
python brainage.py predict \
  --input /path/to/scan.nii.gz \
  --checkpoint <checkpoint.pt> \
  --age 57 --subject-id anonymous \
  --output artifacts/prediction

python -m pytest tests/ -q   # some tests expect local data/artifacts
```

Training runs on Kaggle GPU kernels: the `kaggle/<job>/kernel-metadata.json`
+ `run.py` pairs are uploaded to Kaggle; `tools/v3_queue.py` manages the
prepare/training/merge queue (`docs/MODEL.md`).

### Site

```bash
python3 -m http.server 8123   # http://localhost:8123
```

Fully static — no MRI upload, no analysis, no server-side anything.

## Data sources and licenses

| Source | Publication | License |
|---|---|---|
| IXI | [brain-development.org](https://brain-development.org/ixi-dataset/) | CC BY-SA 3.0 |
| SALD | 2018 public release | CC BY-NC (check the intended use) |
| NIMH | [OpenNeuro ds005752](https://openneuro.org/datasets/ds005752/versions/2.0.0) | CC0 |
| DLBS | [OpenNeuro ds004856](https://openneuro.org/datasets/ds004856/versions/1.2.0) | CC0 |
| OASIS-2 | [sites.wustl.edu/oasisbrains](https://sites.wustl.edu/oasisbrains/home/oasis-2/) | its own usage terms |

Citations, versions and SHA-256 provenance: `DATA_SOURCES.md` and `v3_data.py`.

## Disclaimers

- Proprietary research software — see `LICENSE` (© Asil Doğan Gümüş, all rights
  reserved). Third-party components and datasets keep their own terms.
- A brain age gap is not, by itself, evidence of pathology; regional gaps do
  not sum to the whole-brain gap.
- Dashboard case cards are selected research examples and not an unbiased
  performance summary; four cases contain invented values, while IXI170,
  IXI361, IXI566, DLBS sub-3898 and OAS2_0007 carry real model predictions.
- The site neither uploads nor analyzes data; every figure is precomputed.
- `technical_review` and QC flags are technical-control information, not
  clinical validation.