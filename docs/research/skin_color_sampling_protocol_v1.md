# Frozen TRAIN-color support allocation screen

Actual original MSKCC native skin Lab / DeltaE00. Use only original TRAIN cache.
The prior support-curve fixed six-person internal holdout and remaining18 TRAIN
people are unchanged. Seeds17/29/43; five arms,15fits. Original VALIDATION/CAL/TEST
are not loaded. Internal holdout people were TRAIN in older experiments; this
is exploratory and cannot be called new independent or unseen-camera validation.

All models are original SkinRepresentation('baseline') / CaptureColor mixture:
929,297 parameters, no adapter, camera or mode input. Mode CE0.1 remains training
auxiliary. Target normalization uses all selected TRAIN images and is identical
across arms. Exactly30 rounds x31 batches x32 draws =930 updates/29,760 draws.
AdamW.001 wd.01, cosine after each round to.00001. Native standardized Lab MSE
plus modeCE. Primary final checkpoint only; no holdout-driven selection.

Sampling arms:
1. image: uniform photos with the exact previous integer RNG draw stream;
   must exactly reproduce the three prior n18 baseline final states.
2. person_site: uniform people, uniform sites within person, uniform photos/site.
3. site: uniform distinct sites, uniform photos/site.
4. color: fit only selected TRAIN unique-site native Lab. Let rho_i be the mean
   of exp(-0.5*(DeltaE00_ij/5)^2) across all TRAIN sites, including self. Set
   w_i=clip(median(rho)/rho_i,1/3,3). Site proposal is0.5/S+0.5*w_i/sum(w).
   Then uniform photos within site. Bandwidth5/caps/mixing are frozen constants,
   not fitted to holdout. This kernel is a positive similarity, not a claim of
   exact physical measurement noise or a Gaussian process covariance.
5. color_ipw: exactly the same color proposal/draws, but multiply each per-image
   total loss by site-uniform probability / proposal probability. No batch
   self-normalization. Expected gradient targets site-uniform objective and
   correction weight is bounded above by2 due to the uniform mixture floor.

All non-image proposals use fixed CDF inverse sampling with common uniforms from
seed*1000+round. Camera labels do not enter sampling distributions or loss weights.
No image synthesis, additional weights, paid resources or new data. Original
CC-BY applies. Store probabilities/counts privately, only aggregate diagnostics
and hashes in Git. Check all same-site native Lab equality and person membership.

Evaluate ordinary image-weighted and patient/site-balanced errors, p95 and tails
above5/10, plus fixed100/95/90/80/70/60% and full risk-coverage curves under the
same uncalibrated input-novelty ranking as prior support curves. This is not C+.
Record effective sample mass, min/max proposal ratios, unique sampled sites and
people, and per-round draw digests. Scalar-color and probability audits required.

Hypothesis: allocating updates by measured color support reduces skin tail errors
without new image representations. Main failure: noisy rare sites are amplified
and common-color accuracy deteriorates. color_ipw distinguishes a changed target
objective from a different allocation of stochastic optimization effort; it is
not expected to reproduce a site sampler trajectory sample-for-sample. Balancing
and importance sampling are prior art, not an asserted invention.
