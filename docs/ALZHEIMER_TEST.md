# OASIS-1 Alzheimer cohort research test

This test does not train the model, does not diagnose Alzheimer's disease and
does not assume a true biological brain-age label. The question: in the
dementia/Alzheimer cohort of OASIS-1 separated by CDR, is the brain age gap
higher than in age/sex-matched controls from the same source? CDR is dementia
severity and is not a substitute for a biomarker-confirmed Alzheimer diagnosis.
The source describes the older dementia group as clinical AD:
https://sites.wustl.edu/oasisbrains/home/oasis-1/ .

## Locked model

`artifacts/alzheimer/model_lock.json` pins the completed agebalance-cont2 SFCN
checkpoint with SHA-256. The checkpoint exists on disk and a loading test is
performed. New training does not change this lock automatically. OASIS results
may not be used for model selection, training or calibration. Overlap between
OASIS and the pretraining set has not been verified; results may not be
presented as independent clinical validation.

## Data and running

Read the DUA first: https://www.nitrc.org/frs/download.php/6348/oasis_cross-sectional.csv .
The terms require maintaining de-identification, citation of OASIS and the
required grant acknowledgments. The MRIs are not uploaded publicly anywhere
through this pipeline. The active pilot includes 25 dementia/Alzheimer cohort
cases and 25 age/sex-matched controls. Only the first raw T1 img/hdr files of
these 50 subjects are stored. Because the source serves a compressed archive,
non-selected data may still pass over the network; this does not guarantee
that the method reduces network traffic to 50 MRI files. Full archives can
total about 18 GB; duration depends on network and CPU.

When you accept the terms:

```sh
.venv/bin/python tools/run_oasis_test.py --accept-dua --cases 25
```

This command runs download, matched manifest and prediction with the locked
model in sequence. The selection is recorded in `data/oasis1/pilot_clinical.csv`.

`--discs 1` is only for a technical pilot download and does not guarantee
enough matched pairs. The full cohort defaults to 12 discs.
`--allow-exploratory-overlap` permits exploratory-only analysis while
explicitly accepting the pretraining-overlap uncertainty; it does not replace
the data use agreement. Inputs are not PNG/JPG: the first T1 Analyze img/hdr
pair in the RAW folder. MR2 and repeated acquisitions are not counted as
independent subjects. T88-processed files are not used in place of MNI
alignment. The raw T1 goes through our SynthStrip/FAST/FLIRT SFCN pipeline;
processing takes minutes on CPU.

## Protocol and outputs

Ages 60–100, valid CDR 0/0.5/1/2, MR1; exact sex match, age difference at most
3 years, deterministic greedy matching with non-reused controls. This is not a
global-optimum match. Missing images and failed jobs are reported; the
counterpart of a missing pair is not entered into the group comparison.

`predictions.csv`: chronological age, prediction, age gap, subject/pair id.
`failures.json`: failed preprocessing jobs.
`qc/`: registration/alignment visuals, not attention maps.
`report.json`: dementia-vs-control mean age gap over complete pairs with a
95% confidence interval from 5,000 bootstrap draws. At least 5 pairs is a
technical minimum and not sufficient scientific power. Statistics remain
provisional until visual QC approval.

A positive age gap in the Alzheimer cohort is not, by itself, a diagnosis or a
success criterion. Matching within the same source does not fully remove
acquisition differences. Generalization to the sparse upper ends of the
training age distribution in older OASIS participants is additionally limited.
If the results do not turn out well, do not modify the model using patient labels.