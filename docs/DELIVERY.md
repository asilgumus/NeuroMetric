# BrainAGE delivery summary

> V2 update: the ensemble of four validation-selected members lowered test MAE
> from 6.757 to 6.490 years. The current delivery lives in `artifacts/ensemble-v2`.

## Result

The Westman CNN1 3D ResNet model was fine-tuned on IXI T1 MRI data with a Kaggle
Tesla T4. Participants were split 70/15/15 by center and age triplets with
seed 42 before model training.

| Model | Test MAE | Test RMSE | Test R² | Mean gap |
|---|---:|---:|---:|---:|
| Training-age mean | 14.690 | 17.033 | -0.001 | -0.547 |
| Pretrained CNN1 member 0 | 13.046 | 16.638 | 0.045 | +12.109 |
| Fine-tuned model | **6.757** | **8.414** | **0.756** | **+0.222** |

Fine-tuning lowered test MAE by 6.289 years. Model selection used validation
MAE only; the selected checkpoint is epoch 25 of the final ResNet block and
its validation MAE is 6.309.

## Data and validation

- 561 usable participants: 392 train, 84 validation, 85 test.
- No duplicate participants; all 561 images passed automatic image and
  registration checks.
- IXI002, IXI300 and IXI600 QC montages were manually sampled and judged
  reasonable in alignment. This is not a full expert manual QC of the cohort.
- The official IXI demographics endpoint currently returns HTTP 403, so a
  pinned copy of the same `IXI.xls` file was used and verified with SHA-256.
- Repeated rows with identical age values were merged; contradictory or
  missing age records were excluded during data preparation.

## Working prediction

Final test on a raw NIfTI T1 MRI:

```json
{
  "subject_id": "IXI002-raw-final-check",
  "chronological_age": 35.800137,
  "predicted_brain_age": 36.29789733886719,
  "brain_age_gap": 0.49776033886718807,
  "qc_status": "automatic_checks_passed",
  "research_only": true
}
```

For a new raw T1 NIfTI:

```bash
.venv/bin/python brainage.py predict \
  --input /path/to/scan.nii.gz \
  --checkpoint artifacts/training-complete/training/best_model.pt \
  --age 57 \
  --subject-id anonymous \
  --output artifacts/prediction
```

The command performs FSL rigid MNI alignment and produces `prediction.json`.

## Delivery artifacts

- Model: `artifacts/training-complete/training/best_model.pt`
- Metrics: `artifacts/training-complete/training/metrics.json`
- Test predictions: `artifacts/training-complete/training/predictions.csv`
- Subgroup results: `artifacts/training-complete/training/subgroup_metrics.csv`
- Training history: `artifacts/training-complete/training/history.csv`
- Evaluation chart: `artifacts/training-complete/training/test_evaluation.png`
- Research report: `artifacts/training-complete/training/REPORT.md`
- Fixed data split and QC: `artifacts/prepare-complete/prepared/manifest.csv`
- Raw MRI final test: `artifacts/raw-final-inference/prediction.json`

Checkpoint SHA-256:
`a1184304fd94bb5a018962c31efe5b026e2b7f00ffeaad94de7b026ead6edbfa`

This research work is a research prototype. It does not produce regional brain
age, clinical percentile, disease diagnosis or clinical recommendations.