# Site content alignment

Original static site preserved in `backups/site-before-content-k7jdzLey/`,
including HTML, JavaScript, CSS, images and the old technical-overview PDF.
The active stylesheet, layout and site interactions are unchanged.

Active pages describe fine-tuned SFCN, 923 usable training participants and
the fixed 218-person validation set. Reported deeper-cont3 metrics come from
saved best-checkpoint predictions, not independent test or clinical results.
The static dashboard contains invented example values, never real predictions.
Regional gaps are shown as separate estimates, not an additive decomposition.
Actual regional inference/validation and model-derived attention generation
are not claimed as completed. The schematic heatmap remains in the dashboard.

Slope stays as an explicitly planned, illustrative research view. For the
illustrated endpoints, (69.6 - 58.9) / (61 - 54) ≈ 1.53; it is not a validated
biological ageing rate. No ICC, confidence interval, reference percentile,
clinical endorsement, approved protocol or live upload service is claimed.

The historical PDF remains preserved but is no longer linked from active pages;
the technical overview link now leads to the current model card. Download and
preprocessing services are outside this content-only change. No age prediction
was run for this website update.

## Verification status

- Backup source HTML, CSS, JavaScript and assets exist. Active CSS and app.js
  match the original backup byte-for-byte.
- Four content tests cover local links/anchors, unique IDs, demo disclosures,
  removal of unsupported claims and recorded validation counts.
- Node executes the actual dashboard script for all four cases and invalid-case
  fallback; age units are retained after script rendering.
- Browser visual verification completed through the user's authorized Brave
  CDP connection. Checked 1440px desktop and 390px mobile layouts for overview,
  dashboard and model-card pages. No horizontal overflow; team photos load
  after lazy-loading. All four dashboard cases render correct demo values,
  unavailable reference percentile and schematic heatmap captions.
- Shortened two diagram labels after spotting overlap, without changing CSS
  or geometry. Screenshots are saved under `artifacts/site_review/`.
