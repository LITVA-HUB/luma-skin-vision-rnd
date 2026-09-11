# Fresh source-method check for the skin correspondence experiment

Checked 2026-09-11 while source experiments ran. No external images, weights,
or numeric participant endpoints were acquired in this pass.

- [Original MSKCC paper, Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12749783/):
  triplicate SkinColorCatch measurements, gentle contact to avoid blanching;
  four imaging modes; initial white balance calibration; contact photographs
  involved alcohol and a glass lens. Camera assignment was randomized across
  people, not paired across cameras within the same person. These details
  limit ordinary-phone and sensor-only interpretations. Published high ITA
  ICC is different from locally calculated three-dimensional DeltaE00 spread.
- [Colorimeter-Supervised Skin Tone Estimation, February 2026](https://arxiv.org/abs/2602.10265):
  rechecked the existing prior-art record. Its targets include ITA via color
  regression, with extensive pretraining. Neural instrument-supervised skin
  estimation is therefore not our novelty. No author weights were adopted.
- [DAST original repository](https://github.com/dasec/DAST-SkinTone-database):
  full release still requests author contact; public examples do not establish
  a general dataset license. Existing LICENSE UNCLEAR status unchanged.
- A July 2026 search lead, DOI `10.64898/2026.07.27.26358883`, surfaced through
  the [medRxiv index](https://www.medrxiv.org/content/early/recent/1000?page=150).
  Direct primary abstract/full-text fetching failed (403 / unavailable).
  Original dataset release and license have NOT been verified; do not use this
  lead as evidence of an available phone/reference training dataset. No download
  or author contact followed. An open paper would not itself clear photo rights.

The new paired-mean objective is an algebraic ablation of standard squared
regression and view consistency. No new-invention claim follows from changing
their relative weights; actual single-image skin accuracy is the deciding test.
