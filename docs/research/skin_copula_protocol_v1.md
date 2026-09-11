# Marginal/dependence context with absolute color: source screen

Frozen before model fitting. Independent alternative to the graph hypothesis;
SOURCE exploration only, not a fresh independent test.36fits: none/rgb_hist/
copula/rank_only x seeds17/29/43 x mixed/from_SLR/from_ipod. All rows reported.

Input: original source128x128JPEG crops and64x18patch tokens. Main copula arm
retains these absolute-color tokens unchanged and adds an8x8x8histogram of RGB
channel midranks, computed separately per image with average ranks for ties.
This represents dependence rather than marginal brightness. Strict independent
monotonic channel maps preserve ranks; clipping, channel mixing and local ISP
may not.16TRAIN-image deterministic transform probes test this limited property;
they create no accuracy targets and are NOT training augmentation.

Matched rgb_hist uses the ordinary8x8x8RGB histogram. None uses a zero histogram
(its branch biases can still learn constants; not equal informative capacity).
Rank_only replaces the absolute tokens with tokens computed from rank images;
it is an explicit destructive-information ablation, never a skin-color guarantee.

Architecture: same256wide patch encoder,512wide global context and patch-color/
confidence head, with an additional512->128->128histogram branch and context
input expanded by128. Square-root histograms have equalL2mass for nonempty data.
All arms have identical stored parameter tensors and initialization. Nominal
capacity is approximately1.07M; none has a constant input branch. The primary
mechanism comparison is copula versus rgb_hist, with historical compact ordinary
and mixture controls retained. No graph, foundation model, camera ID or new data.

80epochs/site-balanced real-view batches/AdamW/schedule/standardizedLabMSE and
fit-only target scaling unchanged. Best checkpoint selected using same-camera
source VAL patient-balancedDeltaE00; opposite camera evaluated only afterward.
No source error-head fitting or calibrated coverage claim. Same100/95/90/80/70/60%
uncalibrated patch-dispersion diagnostics. No TEST/CAL or spectral targets.
Data/cache/code hashes and exact best/final/evaluation replay are required.

Copulas and color texture dependence are existing prior art. A useful narrow
result would be improved native skin-color error under source capture changes
from dependence context while absolute color remains available. Failure against
the matched raw-histogram branch rejects this candidate, even if some larger
context network beats the smaller historical patch core.
