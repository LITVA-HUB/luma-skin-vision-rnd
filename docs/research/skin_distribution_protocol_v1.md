# Frozen conditional native-skin-color screen

Question: can retaining an image-conditioned distribution improve actual skin
DeltaE00 versus a matched point regressor, and can a loss-aware decision improve
the same density's mean prediction? Established MDN and Bayes decisions are
prior art. This is an exploratory mechanism screen, not a novelty claim.

Only existing MSKCC TRAIN and source VALIDATION caches are permitted. No CAL or
exposed TEST, no ISSA/UMINHO endpoints. Native instrument Lab is unchanged. Source
validation is heavily reused; camera cohorts differ in people and capture modes,
so transfer is not an isolated camera intervention or fresh phone validation.

Four arms, three seeds 17/29/43, mixed and both source camera directions =36fits.
All share original CaptureColor mixture backbone plus a 512-to-12 scale head,
identical initialization/parameter shapes and fit-only target standardization.
Scale is 0.05+softplus(raw), initialized to 1 standardized unit. Camera and capture
mode are not inputs. No pretrained weights or new data.

- mse_mode: standardized point MSE +0.1 capture-mode CE; historical strong control.
- mse: same MSE without mode CE, separating removal of mode supervision.
- gaussian: 2/3 times single diagonal Gaussian NLL; mean is gated point output,
  standard deviations average the four scale heads. No mode CE.
- mdn4: 2/3 times four-component diagonal Gaussian mixture NLL. No mode CE.

Same 80epochs AdamW lr0.001, wd0.01, cosine to0.00001, batch16sitepairs/32images.
Selection for every arm: smallest same-camera validation patient-mean DeltaE00
of the ordinary point/mixture-mean prediction. This isolates post-fit inference
changes; it does not optimize the density decision's validation score separately.
Other-camera evaluation follows selection. All 36fits run regardless of mixed
outcome; no adaptive hyperparameter search within this screen.

Final density inference evaluates the mean and a finite minimum expected-DeltaE00
decision. Candidate set: mean, component centers, +/-0.25 and +/-0.5 marginal SD
along each Lab axis. Normal-component integration uses order3 tensor Gauss-Hermite
(27nodes/component), with order2 a fixed sensitivity check, never selected.
This is neither the exact continuous Bayes optimum nor a physical skin spectrum.
Unbounded Gaussian Lab support can include implausible colors; report rather
than silently clip. For mse arms, only ordinary prediction and historical
four-hypothesis dispersion ranking are meaningful; unfitted scales are not risk.

Measure full DeltaE00, patient means, median, p95, fractions above5/10 and fixed
100/95/90/80/70/60% coverage plus full curves. Density risk is predicted expected
DeltaE00, not calibrated error or an upper bound. Calibrated C+ requires a later
person-held-out residual experiment, only if prediction/ranking shows promise.

Record all 36negative or positive fits, weights, histories, exact code/data hashes,
stored/active parameters, checkpoint size and fit-process peak allocation.
No latency/export optimization before positive evidence. Verify density against
torch.distributions, quadrature moments, target-free decision behavior, model
replay and independent scalar CIEDE2000 aggregation before conclusions.

Go/no-go: a direction-specific gain is insufficient for universality. Compare
three-seed matched means and descriptive patient uncertainty against strongest
historical controls. Stronger independent evidence requires a new untouched
compatible dataset, not reusing the exposed MSKCC test.
