# MSKCC diagnostic error-floor audit — fixed scope

Owner instruction: diagnostic analysis only. No model or quality-predictor fitting,
no model search, no new source acquisition beyond restoration of the original
MSKCC TRAIN material. C and the face inference are not changed by this branch.

Base: `9cad271aad159d73083259967b4107b2ce828eb1`.

## Cohort and provenance

Use the original `skin_mskcc_data.patient_roles` implementation and historical
acquisition/split receipts. Restore source files only on exact SHA256 matches.
Expected scope: 966 images / 24 patients / 248 exact measurement sites.
No GroupKFold regeneration and no reserved numeric endpoint analysis. This
descriptive audit does not require refitting the original OOF fold models.

Site identity is patient plus source `tag_id`, not an anatomical category.
One recorded three-assessment S4 row is a reference acquisition; photographs
sharing that row are not independent reference repetitions.

## Measurements

- Instrument-to-instrument CIEDE2000: three unordered assessment pairs per site;
  coordinate differences and sample SD; patient-cluster bootstrap of equal-patient
  means, seed73019, 10,000 draws.
- Target sensitivity: triplet mean versus each two-reading mean. This is a
  reference perturbation, not latent-truth estimation or a known error floor.
- Capture: EXIF + embedded ICC to sRGB; unprofiled RGB assumes sRGB; central
  square `int(.8*min(width,height))`, original-pixel mean/median, and Lanczos128
  color36. Crop median is not a verified skin-only mask. No face parser on
  dermoscopic images. Exact identity with the unavailable historical C cache is
  not asserted.
- Observed image Lab uses D65/2 degrees; instrument provenance specifies
  D65/10 degrees. No physical cross-observer DeltaE is computed.
- All exact-site image pairs; separate capture-mode/type/device/anatomy groups;
  unique-site and equal-patient weighting as descriptive summaries.
- All label-informed oracles explicitly `LEAKY_DIAGNOSTIC_ONLY`; leave-site-out
  patient oracle weights other unique sites equally.

## Counterfactual masks fixed before C residual access

Original C per-image OOF predictions are unavailable in the restored workspace.
The following diagnostics are fixed without them:

1. Reference-stable sites: mean of three instrument pairwise DeltaE values no
   greater than p80 over all248 unique TRAIN sites.
2. Capture-stable sites: mean same-site pairwise original median-ROI observed
   DeltaE no greater than p80 over247 sites with multiple photographs. A singleton
   site is unknown, not assumed stable.
3. Clipping: original ROI fraction of pixels with any channel>=253 plus fraction
   with any channel<=2, no greater than p80 over966 TRAIN photographs. The sum
   is a proxy and can double count a pixel.
4. Reference/capture intersection and reference/capture/clipping intersection.
5. Keep every tie; report actual image/person/site coverage. Main full-cohort
   metrics remain unchanged. These are not deployment quality gates.

No subtraction of instrument DeltaE from model DeltaE, no best-reading selection
against predictions, no causal percentage attribution from these observations.

## Required C input and validation

Need `LUMA_TRAIN_OOF_C.zip`, specifically
`experiment/private_review/oof_anonymized.npz`. No training to reconstruct it.
Require explicit prediction/target/index/group keys, complete original row-index
permutation, target equality, patient-label bijection, matching capture metadata
and exact frozen-mask equality. Original archive provenance remains necessary:
identical targets cannot detect falsely asserted index swaps within one site.

The continuation analyzer computes grouped C errors, global-tail contributions,
Pearson/Spearman with patient-cluster intervals, anonymous top50, reference
perturbations and fixed-mask diagnostics. Synthetic tests of this analyzer are
not evidence about real C errors. Final MODEL-LIMITED / DATA-CAPTURE-LIMITED /
MIXED attribution is deferred until the paired residual audit is possible, and
unidentifiable causal components must remain explicitly unidentifiable.
