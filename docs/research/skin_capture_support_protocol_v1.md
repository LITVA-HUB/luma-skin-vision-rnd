# Observed same-site patch-bag support: frozen real-image screen

TRAIN diagnostic:966images,24people,248sites,1421unordered pairs. All same-site
native Lab targets agree exactly. No same-site pair spans cameras. Coordinate
pattern cosine is not a registration test; about42% are below0.5. Do not infer
that corresponding pixels depict the same physical point or that a camera
intervention is measured. One site has only one capture and is retained.

Hypothesis: training on bags of observed patches from multiple actual captures
of one skin site can reduce reliance on capture appearance. Every patch token
must exactly equal an original observed token from that same TRAIN site. No
RGB interpolation, invented spectrum, synthetic target or pixel correspondence
is used. A mixed bag is a derived training input, not a real single photograph
or a new independent observation. Test inputs remain one original photograph.

## Five matched arms

Use original CaptureColor mixture,929297parameters, same seeds17/29/43 and data.
Three protocols mixed/SLR-to-iPod/iPod-to-SLR,5arms =45fits. No foundation model,
camera ID input or inference-time adaptation. Same80epochs,AdamW0.001,wd0.01,
cosine to0.00001,16sitepairs/32images per step,64patches per image.

- baseline: original bags, standardized Lab MSE +0.1hard capture-mode CE.
- self_bootstrap: with probability0.5, resample64patches with replacement from
  that same image; mode and Lab unchanged. Controls resampling regularization.
- soft_mode_control: leave image patches unchanged but use exactly the paired
  arms' softened auxiliary mode target. This is an auxiliary-label regularizer,
  not a claim that the original image has a different measured capture mode.
  Its comparison with paired_union isolates the actual input mixing effect.
- paired_union: with probability0.5, build64patches from the paired same-site
  images. Draw partner fraction uniformly0..1, independent Bernoulli source
  choice per patch, and independent within-image patch indices with replacement.
- paired_stratified: same probability/source choices, but each of64grid indices
  is used once, taking either image's observed token at that index. The model
  has no positional input; this is stratified bag sampling, not pixel registration.

All arms receive the exact same epoch pair draws. Augmentation uses a separate
fixed NumPy stream, so it cannot change original pair selection or model init.
In paired arms, Lab remains the verified common reference. Auxiliary mode CE
uses actual patch-origin proportions (multiples of1/64); this describes source
composition, not an independently measured mode label for a fictional photo.
No extra forward passes or additional source images per optimization step.

Only TRAIN/sourceVALIDATION caches. Select smallest same-camera validation
patient-mean DeltaE00; evaluate other camera after checkpoint selection. All45fits
run regardless of mixed results. The independent TEST/CAL archive is untouched.
All source cohorts are repeatedly explored, not fresh confirmatory experiments.

## Evidence

Verify exact patch provenance, unit mode mass and same-site target equality.
Archive aggregate augmentation counts and a deterministic plan digest. Compare
full/100,95,90,80,70,60%skin DeltaE00, mean/median/p95/tails and complete curves.
Risk ranking remains uncalibrated four-hypothesis dispersion. No C+ or error-bound
claim without later person-held-out residual fitting/calibration.

Report same-shape/initialization, exact inference replays, independent scalar
color checks, fit VRAM, parameter count and checkpoint bytes. No deployment
optimization before positive evidence. Mixing/patch augmentation and MixStyle,
AugMix, CutMix/MIL bag mixing are prior art; this combination is unverified.

The key failure mode is that within-camera capture variation does not cover
new-camera processing. A gain in mixed validation alone cannot establish the
requested universal phone facial accuracy. Preserve negative outcomes.
