# TRAIN-only measured spectral representation and ambiguity probe

Frozen before numerical extraction of the selected skin regions. This is a
mechanism falsifier, not a new RGB camera or independent color-accuracy benchmark.
No MSKCC TEST/CAL and no UMINHO VALIDATION/TEST cubes may be loaded.

## Data and region selection

Use the three smallest original TRAIN cubes by bytes from the already frozen
UMINHO manifest. Original CC BY4.0; verify source hashes before every decode.
The three faces were inspected locally at550nm with a coordinate grid. Select
one50x50forehead region and two50x50cheek regions per face, avoiding visible hair,
eyelids, lips, nostrils and boundaries. Coordinates and original names stay local;
commit their hash. This is manual source-only skin selection, not an independently
validated segmentation or guaranteed pure diffuse reflectance. No clipping or
numeric-based pixel deletion; report all nonpositive/nonfinite/above-one values.
No masks, spectra, faces or participant identifiers are published or committed.

## Representation screen

Three leave-one-face-out folds WITHIN TRAIN, equal pixels and regions per face.
Fit a mean and PCA basis using both remaining faces, compare linear-reflectance
and log-reflectance PCA at ranks1/2/3/5/8. Report mean-spectrum and nearest-source-
spectrum controls. All projected query coefficients use the full33-band measured
spectrum: this is an ORACLE COMPRESSION/REPRESENTATION ceiling, not RGB inference.
No such oracle may appear as a deployable-model benchmark. Report reflectance
RMSE, per-spectrum relativeL2 error and p95, plus unphysical reconstructed values.
33bands400–720nm; no invented full-visible CIEDE2000 or instrument Lab labels.
Pixels are correlated; three faces cannot establish population generalization.

## Regional ambiguity stress

Average each of nine regions, enumerate all36unordered pairs. For each pair,
use the first measured reflectance under unit illuminant and solve for the
positive smooth illuminant on the second region by least squares on log-ratio.
Compare degree0(exposure only),1,2,3 polynomial log-SPDs on400–720nm. This family
is a DECLARED MATHEMATICAL STRESS FAMILY, not measured household illuminants.
Report radiance relativeL2 mismatch, illuminant max/min and absolute gains,
and original reflectance separation. Keep all pairs, do not present a selected
counterexample as prevalence. Local spectra only: spatial context may disambiguate.
Spectral product model excludes geometry, fluorescence and nonlinear phone ISP.

If low-rank material constraints fail between these faces, expand the source
material family before a learned inverse. If very different measured spectra
can be made similar under smooth spectral scaling, material plausibility alone
is insufficient: retain capture/context evidence and explicit ambiguity output.
Neither outcome ends the real-photo R&D objective. Follow with a matched real-photo
experiment; do not replace it with a large derived-image optimization campaign.
