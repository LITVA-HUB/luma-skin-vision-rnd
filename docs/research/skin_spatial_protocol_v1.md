# Real-photo source spatial mechanism screen

Frozen before any training. Target is original MSKCC instrument nativeLab;
evaluation CIEDE2000. Same verified128x128JPEG crop and64row-major patch tokens
as the original source cache. No new pixels, spectral targets, camera ID,
capture-mode labels, test-time adaptation, pretrained weights or extra data.
Only original TRAIN/VALIDATION; never load exposed TEST or CALIBRATION.

## Mechanism and controls

Invert exchangeable-patch aggregation by exchanging information on the8x8grid.
Six arms: plain, conv1, conv3, graph1, graph3, graph3_scrambled. Three seeds17/29/43,
three protocols mixed/from_SLR/from_ipod:54fits, all reported regardless of wins.
No cross-camera endpoint selects an epoch or arm for another fitting run.

Common core:18->256->256local encoder, pooled mean/max/std->512->512context,
local+context->256->four patch outputs (three color values and confidence).
All models initialize this core identically, then initialize identical auxiliary
modules. Stored and actually active parameter counts are reported separately.
Plain reproduces the original patch-voting mechanism without extra processing.

Graph: four-neighbor undirected spatial lattice, symmetric nonnegative affinity
from squared learned-feature distance and a learned positive temperature/coupling;
positive per-node anchors. Starting from local featuresh, take1or3Jacobi updates
for (diag(a)+L_w)z=a*h. Inject the latent residual through a common256x256linear
map and .5*tanh bounded residual before nonlinear context/color readout. This
is latent regularization, NOT calibrated spectral inverse rendering or a new
physical law. Constant latent fields are preserved by the solver. A purely
linear confidence-weighted pooled solution would conserve its weighted mean,
so it is not used as a supposedly improved color estimate.

Ordinary controls: a learned depthwise3x3residual convolution, repeated1or3times
with shared weights, followed by the same residual projection/readout. Parameter
counts closely match graph arms; measure rather than assume equal latency/VRAM.
Graph3scrambled uses one pre-fixed random node permutation before diffusion and
its inverse afterward, at training and evaluation, retaining identical content
but replacing true spatial adjacency. Same random permutation for all seeds.
Zero graph steps equals plain exactly; verify numerically rather than retrain.

These mechanisms have graph diffusion and recurrent-network prior art. No new
method claim follows from implementing them. A graph model must beat plain AND
the matched ordinary spatial control consistently, with intact geometry helping
over scrambled geometry, to justify a subsequent stronger experiment.

## Fitting and selection

80epochs AdamWlr.001wd.01 cosine end.00001. StandardizedLabMSE using only fitting
target mean/std. Same site-balanced batches as the previous source experiments:
16sites with two actual views,ceil(ntrain/32)updates per epoch. No paired-output
penalty, mode auxiliary loss or augmentation. Identical batch schedules per seed.
Select best epoch by original same-camera VALIDATION patient-balanced meanDeltaE00;
retain final epoch too. Mixed:24TRAINpeople966images/6VALpeople264images.
SLR-only fitting:8people323images,3SLRVALpeople132images, evaluate other3iPodVAL
people132images after selection. iPod-only fitting:16people643images with inverse
camera roles. Both camera families were examined in prior source research, so
this is EXPLORATORY SOURCE CAMERA TRANSFER, not independent unseen-camera proof.

Report mean/median/p90/p95 and tails>5/>10, repeatability, runtime/VRAM and per-arm
seed results. As diagnostics only, rank using uncalibrated nativeLab patch-vote
dispersion at100/95/90/80/70/60%coverage; no expected-error or coverage guarantee.
No calibrated selective claim until a new source-OOF protocol is justified.
Save checkpoint hashes, source/code bindings and predictions. Replay best/final
selection and evaluation outputs; independently recompute scalarDeltaE00.
