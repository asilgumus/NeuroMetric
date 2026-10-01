# Real MRI demo features

## Completed

- OASIS-30: new NIfTI with documented ASL geometry from the preserved original
  Analyze file; SynthStrip, FAST and 12-DOF FLIRT; technical registration check
  in nine planes.
- Locked contrast-cont4 SFCN prediction: age 62.344738; chronological age 65;
  gap -2.655262 years. A new signed Grad-CAM and separate three-plane views are
  wired into the dashboard.
- FastSurfer v2.4.2 VINN segmentation was recomputed; five bilateral volumes
  were published after technical visual control in 15 planes and a full
  anatomical label audit.
- Volumes (mL): hippocampus 7.504; temporal cortical gray matter 99.570;
  frontal 142.800; parietal 102.996; occipital 54.759. Cortical groups are not
  whole-lobe volumes; white matter, cingulate and insula are excluded.
- OASIS-2 OAS2_0001 MR1/MR2: two real MRIs, 457 days between visits, ages 87/88.
  Predictions with the same preprocessing and model: 70.542854/73.303589.
  Gap change +1.760735 years; this may not be read as a biological ageing rate.
- `longitudinal.html`: real dataset nWBV values kept separate from model
  predictions; no invented calendar dates. The clearly low prediction in this
  example is explicitly flagged.

## Limits and publication

Technical visual control is not clinician approval or clinical validation.
A healthy brain, Alzheimer risk, normative percentile or regional age is not
computed. The local static demo shows completed measurements; it is not a live
MRI upload/remote inference service. Public distribution has not started.
OASIS terms of use and citations must be maintained.

Sources:
- https://sites.wustl.edu/oasisbrains/home/oasis-2/
- https://doi.org/10.1162/jocn.2009.21407
- https://brainder.org/2011/08/13/converting-oasis-brains-to-nifti/
- https://4dfp.readthedocs.io/en/latest/format.html

Reproduction: `tools/fetch_oasis2_demo.py`, `tools/prepare_oasis2_demo.py`,
after visual control `tools/record_technical_qc.py`,
`tools/finish_demo_measurements.py --longitudinal --regional-reviewed`.
QC control flags must not be issued for unreviewed images.