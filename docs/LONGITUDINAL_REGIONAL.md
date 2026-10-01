# Real follow-up and regional measurement pipeline

`data/oasis30_visits.csv` contains the current real prediction. Source:
`artifacts/alzheimer/predictions_contrast_cont4/all_predictions.csv`.
The scan date is left empty because it is unknown; the date of a prediction run
is not the scan date. Preprocessing comparability and visual QC are not yet approved.

Follow-up summary:

```sh
.venv/bin/python tools/longitudinal_measures.py --input data/oasis30_visits.csv --output artifacts/alzheimer/longitudinal_OAS1_0030.json
```

To refresh the dashboard data as well, add
`--javascript-output assets/oasis30-longitudinal.js` to the same command. The
dashboard loads this static data; model training or MRI prediction never runs
in the browser.

New visits are appended for the same subject with the real scan date,
chronological age, model output, checkpoint SHA256, preprocessing signature and
visual QC status. `visual_qc=approved` is written only after the image has
really been reviewed. No change rate is produced for a single scan, across
different models, or for unreviewed images. With two or more comparable visits,
the first-to-last visit difference and the annualized model difference are
computed; this measure is not a clinically validated ageing rate.

Regional volumes:

```sh
.venv/bin/python tools/regional_measures.py --mri native.nii.gz --segmentation native_labels.nii.gz --labels label_mapping.json --output regional.json --qc-approved
```

`label_mapping.json`: a JSON that maps region names to label numbers according
to the label dictionary of the anatomical segmentation tool. Do not use the
example numbers without checking the dictionary of the real tool. The MRI and
the label map must be in the same native coordinates, and spatial units must be
explicitly mm. The output measurement is mL; it is not a healthy-control
percentile and not a regional age. A fixed atlas template volume must not be
presented as a patient's anatomical volume.

Sources evaluated for anatomical segmentation:
[SynthSeg official repository](https://github.com/BBillot/SynthSeg),
[FreeSurfer SynthSeg documentation](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg).
Downloading the repository or passing unit tests does not mean real patient
segmentation succeeded; the real output and image QC must be verified separately.

Local SynthSeg research setup: `.cache/SynthSeg`, separate Python 3.8
environment `.cache/synthseg-env`; the current training environment is unchanged.
Downloaded code commit: `2a2aa3bbfccb83f8253a51ca8b329b9938a2646d`.
SynthSeg 1.0 weight in the repo, SHA256:
`7d2e32d298fe38dc51ea38e6a8e8fc5c665d56b63cddb409e8b199535f8b5298`.
The 1.0 model alone does not provide cortical lobe parcellation; start with
labeled subcortical structures such as the hippocampus and complete cortical
parcellation separately.

## OAS30 experimental result

SynthSeg could not find the hippocampus with the generic affine in the original
Analyze conversion. `OAS1_0030_MR1.xml` in the official archive confirms that
the acquisition is sagittal and that the voxel sizes are 1 × 1 × 1.25 mm.
A separate ASR orientation hypothesis was produced without modifying the
original MRI; right/left orientation is still unverified. In this experimental
segmentation, bilateral hippocampus 7.939 mL, amygdala 3.091 mL, thalamus
11.832 mL and cerebral cortex 505.623 mL hard-label volumes were obtained.
Because SynthSeg's own CSV contains soft posterior volume, the numbers are not
required to match.

`tools/export_oasis30_candidate_regions.py` exports these results to
`assets/oasis30-regional-candidate.js` strictly as experimental, unapproved
data. The approved regional volume export pipeline is separate. The dashboard
shows these numbers as "candidate" in a separate section. They are not age or
disease risk. Temporal/frontal/parietal lobe parcellation and orientation
verification are incomplete.

## Cortical parcellation experiment

FastSurfer v2.4.2 (commit `7e5334356e5abc0b600c11d6e9890e386d176433`) was run
on CPU over the experimental ASR image with three VINN checkpoints from the
official Zenodo record. Source configuration:
`.cache/FastSurfer/FastSurferCNN/config/checkpoint_paths.yaml`.
Batch size 1, two threads; the existing training environment was not modified.
A separate environment `.cache/fastsurfer-env` uses the packages of the current
PyTorch environment read-only; it has its own NumPy and imaging dependencies.
Single source change: the one-element `Compose([ToTensorTest()])` was replaced
with a direct `ToTensorTest()` call doing the same work, so torchvision is no
longer required just for Compose. Modified inference.py SHA256:
`36067c70ecb7eb1195aea40b1da9f6afc2fc5bf67dd989c10f30ea8bda541e6a`.

`data/dkt_lobe_groups.json` aggregates DKT labels into explicit, two-sided,
disjoint gray-matter groups for this project. These are not whole-lobe volumes
including white matter; the cingulate cortex and insula are kept separate and
do not contribute to the lobe sums in this file. This grouping is not a
clinical norm. Patient volumes are not produced from these groups before real
parcellation is complete.

The FastSurfer experiment completed; the 15-slice overlay control visual
`artifacts/alzheimer/fastsurfer_OAS1_0030/cortical_qc.png` was reviewed.
All required group labels are present in the output; experimental bilateral
hard-label volumes: hippocampus 7.557 mL; temporal cortical gray matter
99.239 mL; frontal 143.460 mL; parietal 102.706 mL; occipital 55.145 mL.
The dashboard's experimental table now shows this FastSurfer output; previous
SynthSeg results remain in their own artifact directories. Orientation and
clinical validation are incomplete; no result is presented as an approved
measurement. The orientation issue in the first raw MRI input additionally
requires corrected registration and re-evaluation for the current brain-age
output and Grad-CAM.