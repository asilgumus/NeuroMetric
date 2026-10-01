# BrainAGE research prototype

## SFCN refinement experiment (2026-10-01)

The extended SFCN run completed all 150 additional epochs. Its best checkpoint
was at additional epoch 144: validation MAE 4.4553 years (218 participants).
This is not an independent-test result. The separate `brainage-v3-train-sfcn-refine`
experiment warm-starts that checkpoint, pinned to SHA-256
`a3e5b8da3ecec9b634f697b3820ec027a9dd8250e4f93e91be30d89a0d9e06a3`.
It trains only `conv_4`, `conv_5` and the age head with learning rates
3e-6, 1e-5 and 1e-4. BatchNorm running statistics remain frozen.
Training augmentation adds zero-padded shifts of up to two voxels and 50%
left/right mirroring on axis 0 (verified against the MNI template's LAS axes).
Validation is never augmented. AdamW weight decay 1e-4, dropout 0.5 and the
existing KL + 0.05 MAE loss are unchanged. Effective batch size is eight.
ReduceLROnPlateau uses patience 5, factor 0.5, minimum LR 1e-7 and zero
improvement threshold. The run allows 80 additional epochs with early-stopping
patience 20. Parent weights are retained if no improvement occurs; test and
external data are not mounted. These settings are experimental, not a proven
optimum. The original and extended jobs remain unchanged.

This project adapts Westman et al.'s released CNN1 3D ResNet to the IXI
healthy-control T1 MRI data set. It is a research prototype, not a clinical
device or diagnostic system.

## Pipeline

1. `prepare` matches each T1 scan to an IXI age label, removes uncertain
   demographics, makes one participant-level split, performs the original
   FSL rigid registration, and saves a normalized `80×96×80` array plus QC
   image.
2. `train` measures the fixed pretrained model, fine-tunes its regression
   head, optionally opens the final ResNet block, and chooses the deployed
   checkpoint using validation MAE only.
3. `predict` emits `chronological_age`, `predicted_brain_age`,
   `brain_age_gap`, model hash, and automatic QC status.

The first experiment runs CNN1 member 0. The V2 improvement separately
fine-tunes the five pretrained members and averages the validation-selected
ones with equal weights.

## Reproducibility and evaluation

- Seed: `42`; split: 70/15/15 by participant, stratified by site and age.
- `provenance.json` records upstream source, model checksum, and demographic
  table checksum.
- Training selects by validation MAE, then reports test MAE/RMSE/R² and
  site/age-group metrics. It never selects a checkpoint with test results.
- The official IXI demographic endpoint currently returns HTTP 403. The code
  uses a pinned copy and verifies its SHA-256 checksum, while retaining the
  official IXI URL in provenance.

## Kaggle runs

`kaggle/prepare` is the private preprocessing job. When it succeeds, push
`kaggle/train`, which reads its prepared output as a Kaggle kernel source.
The short `kaggle/pilot` run exists only to test the complete MRI path before
the full cohort is processed.

For the V2 ensemble prediction, run `brainage.py predict --checkpoint` and
supply the selected checkpoints in `artifacts/ensemble-v2/member_*.pt` together.
The output records each member's prediction and the ensemble average.

## Limits

Brain-age gap is raw predicted age minus chronological age. This version does
not calculate regional brain ages, clinical percentiles, disease labels,
longitudinal trajectories, attention maps, or clinical recommendations.

## V3 research pipeline (in progress)

V3 extends the locked IXI split with SALD and NIMH training/validation T1 MRI,
then reserves DLBS wave-1 for an independent external test. The generated
catalogue is `artifacts/v3/catalog/catalog.csv` (1,691 distinct people: 924
training, 218 validation, 85 historical IXI test, 464 external DLBS). Public
source metadata, licensing, and URL/checksum provenance are in `v3_data.py`.
SALD is CC BY-NC; confirm competition use complies with that license.

`v3_prepare.py` makes the original CNN1 2 mm inputs and independent 1 mm
SynthStrip → FAST bias correction → FLIRT 12-DOF MNI inputs for SFCN. It writes
losslessly compressed `.npz` arrays, QC montages, and rejects non-finite
volumes or gross registration errors. The ResNet path additionally rejects a
near-empty central anatomy crop; the merge stage applies the same guard to
shards prepared before that check existed. A
pilot MRI from every dataset and visual QC are required before a full run.
Locally, use only one worker on a 16 GB RAM machine; the simultaneous pilot
run previously exhausted memory. Kaggle source jobs are split into shards and
the source dataset is only marked complete when every shard finishes.
For IXI specifically, a label-blind V2 rigid transform is applied after the
same FSL reorientation used to fit it. The initial MNI image is cropped around
the brain before SynthStrip, reducing pilot peak RAM from about 8 GB to about
3.5 GB; the later 12-DOF SFCN registration remains independent.

`v3_model.py` reproduces the released SFCN feature extractor and loads its
pretrained UK Biobank weights with SHA-256 verification. `v3_train.py` adapts
SFCN age-distribution and CNN1 regression heads on training data, selects
checkpoints on validation MAE, and never opens held-out splits. `v3_merge.py`
combines completed preparation shards. In a GPU job it can use lightweight
links; the source-level Kaggle merge jobs materialize portable copies plus QC
images so training needs only three cohort inputs and evaluation needs four.
The SFCN loader converts stored positive-voxel-normalized arrays to the
released checkpoint's whole-MNI-volume-mean convention before center-crop
inference; otherwise its transfer-learning starting point is badly distorted.
Before optimization, training saves `baseline_metrics.json` and
`baseline_validation_predictions.csv` for the exact initialization. The ResNet
baseline uses released weights; the SFCN initialization includes the expanded
18–100 head, so it must not be described as the untouched UKB model.
`v3_evaluate.py` chooses among V2, SFCN, ResNet and equal-weight ensembles on
IXI validation MAE (±1 year tie-break); only then does it compute IXI test and
external-DLBS MAE, RMSE, R², ±1/5/10-year rates, rounded-year agreement and
bootstrap confidence intervals, plus paired MAE difference versus V2 and
age-band/site breakdowns. No V3 accuracy claim is valid until this
evaluation has actually completed.

After the selected checkpoints and `selection.json` exist, `v3_predict.py`
preprocesses one new de-identified T1 NIfTI with the same ResNet and SFCN
paths, verifies the selected checkpoint hashes, and writes a research-only
JSON with predicted brain age, age gap, provenance hashes and QC montages.
It does not emit a diagnosis, percentile or regional claim.
The raw-MRI preprocessing path passed a local IXI pilot on 2026-09-26;
ResNet/SFCN montages are in `artifacts/v3/pilot_predict_qc`. This is an
integration and visual-QC check, not a held-out accuracy result.

Generated Kaggle jobs come from `tools/build_v3_kaggle.py` and
`tools/build_v3_merge_kaggle.py` and `tools/build_v3_train_kaggle.py`.
These builders embed the exact source and
locked catalogue into private Kaggle kernels. Source kernels must not be run
until the pilot QC passes for SALD, NIMH, DLBS and IXI.
After pilots pass, `KAGGLE_CONFIG_DIR=/path/to/kaggle-config python tools/v3_queue.py`
fills at most five CPU preparation slots and launches the 8 IXI, 8 SALD,
4 NIMH and 8 DLBS shards as slots free up. Once every shard of a source has
finished, it also launches that source's portable merge job. It stops on a
failed job so its logs can be inspected; it does not launch training or claim
that QC is done.
The source-level merge jobs render a ResNet-input montage from each copied
array, including IXI arrays reused from V2. Their age/site-stratified review
sheets show both SFCN and ResNet inputs for every selected participant, plus
an index and automatic-failure list. Generating sheets is not approval.
`tools/v3_qc_sheets.py --output <review-dir> <merged-cohort-dir> ...` can
regenerate the same two-modality sheets from downloaded merged outputs. After
visual inspection,
`tools/v3_qc_fingerprint.py <merged-output>/prepared/<source>` prints the
catalogue, manifest, summary, failure-list, index and montage hashes needed for
`artifacts/v3/qc/approval.json`. The SFCN/ResNet training and final evaluation
jobs refuse to run until an explicit reviewer-approved record matches the
exact mounted cohort output. The project owner approved the current cohort
QC on 2026-09-30; `artifacts/v3/qc/approval.json` records that decision and
the exact fingerprints. Both V3 training kernels were submitted afterward.

`tools/v3_training_watch.py --hours 2` checks Kaggle training readiness and
active jobs at 3, 8, then 10 minute intervals. Transient API failures reset
the interval to 3 minutes. It submits only missing jobs after the exact
downloaded cohort QC fingerprints match human approval; existing running
jobs are never restarted. Status/history and terminal failure logs are saved
under `artifacts/v3/training_watch`. Completed training outputs are downloaded
for the subsequent locked evaluation. It does not fabricate QC approval or
automatically retry a terminal training error.

`tools/build_v3_baseline_kaggle.py` builds a separate provisional inference
job over validation subjects only (218 before manual review), using all five
released CNN1 members and the untouched 40-bin SFCN head with its original
42.5–81.5 age centers. It performs no fitting, opens no held-out MRI and
does not select a model. Results carry `human_review_pending` and remain
provisional until QC is approved. The training watcher also monitors this
job and downloads its metrics/predictions when it completes.

The project owner requested a longer ResNet run on 2026-09-30. The separate
`brainage-v3-train-resnet-extended` kernel starts from the original selected
checkpoint (SHA-256 `f76b2c43fd389b5b1acf1fd85dadd87ac7914674d5245935608867d80cbadf01`),
with 0 head-only epochs and at most 60 additional last-block/head epochs,
patience 15. Optimizer state is reinitialized; this is warm-start fine-tuning.
The initialization remains a candidate, so no improvement retains the parent
model. Original run outputs are preserved. A separate adaptive watcher stores
extended results under `artifacts/v3/trained_resnet_extended` and watches both
this run and the ongoing SFCN job. Final selection must compare extended and
original validation outcomes before any held-out evaluation is run.
