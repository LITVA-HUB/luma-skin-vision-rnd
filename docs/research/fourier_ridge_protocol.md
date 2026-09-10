# Spatial-CNN removal spike: Fourier ridge correlation filter

A source-development control, not a claimed new FFCC algorithm/reproduction.
Uses already audited FFCC-inspired two log-chroma histograms (color and local
absolute deviation),64x64 periodic grid; replace learned spatial CNN and
iterative nonlinear optimizer with independent complex ridge normal equations
per frequency. Three regressors are two histogram channels and a bias.
Targets are circular Gaussian heatmaps around true illuminant log chroma.
This is a convex squared-heatmap objective, not original FFCC CE/Gaussian NLL.

Same1126 real SimpleCube++ train and119 reused validation, never phone/test GT.
Use existing128px thumbnails scaled by per-image maximum to explicit[0,1],
without changing chroma. This is not author full-resolution preprocessing.
Fixed sweep: Gaussian sigma1 or2bins; mean-loss ridge lambda .001,.01,.1,1;
softmax temperature200; both gray_light and gray_world torus unwrapping.
All16 outcomes retained; select by validation mean reproduction only. Report
raw1-confidence risk80 separately. No calibrated probability/coverage claim.
Preserve numerical source SHA, data identities, each model, every validation
prediction and elapsed fit time.12,288 real spatial filter/bias coefficients;
complex storage is a representation, not a free doubling of hypothesis capacity.

Falsifier: if no candidate is competitive with existing cheap statistics/compact
CNN source validation, this exact linear least-squares variant is not sufficient.
That does not disprove original FFCC, other likelihood objectives or better
semantic representations. Useful advantage would be comparable error at orders
of magnitude fewer parameters; any transfer claim needs separately frozen data.
