# Fourier histogram alternative:34 development outcomes

The spatial CNN can be removed with a substantial size reduction, but the tested replacement does not beat the strongest CNN. This is a known-family correlation-filter control, not a new FFCC reproduction or a novelty claim. Real SimpleCube++ train1126/validation119; validation has been reused throughout research. No phone/test images were scored by these models.

Phase1:16 fixed cases. Best12,288-coefficient sigma2/ridge.001/gray-world-unwrapped model gives mean2.7468 degrees, median1.3577, raw risk80 2.0714, p95 8.4429, max18.4504. At80% acceptance1/95 accepted images remains above10degrees. All16 saved filters were replayed against reconstructed validation histograms and independently rescored.

Phase2:18 additional cases, selected after inspecting phase1. Lower ridge only marginally changes the mean2.7468 to2.7446. Removing the input-independent learned color prior reduces active coefficients to8192 but worsens the best mean to3.3792. Smoothing the target more broadly also fails to improve the mean. Preserve the entire34-case adaptive search; do not promote its chosen validation minimum to independent accuracy.

|Phase|Candidate|Active real coefficients|Mean reproduction°|Raw risk80°|Fit+validation seconds|
|---|---|---:|---:|---:|---:|
|1|sigma1.0_ridge0.001_gray_light|12288|6.8843|3.6743|0.515|
|1|sigma1.0_ridge0.001_gray_world|12288|5.9448|3.3655|0.515|
|1|sigma1.0_ridge0.01_gray_light|12288|6.9308|3.3882|0.512|
|1|sigma1.0_ridge0.01_gray_world|12288|6.0306|3.0812|0.512|
|1|sigma1.0_ridge0.1_gray_light|12288|7.2242|3.2708|0.499|
|1|sigma1.0_ridge0.1_gray_world|12288|6.4660|3.2708|0.499|
|1|sigma1.0_ridge1.0_gray_light|12288|6.8464|2.9578|0.473|
|1|sigma1.0_ridge1.0_gray_world|12288|6.0778|2.9578|0.473|
|1|sigma2.0_ridge0.001_gray_light|12288|4.0918|3.2096|0.528|
|1|sigma2.0_ridge0.001_gray_world|12288|2.7468|2.0714|0.528|
|1|sigma2.0_ridge0.01_gray_light|12288|4.1040|3.2336|0.461|
|1|sigma2.0_ridge0.01_gray_world|12288|2.7655|2.1005|0.461|
|1|sigma2.0_ridge0.1_gray_light|12288|4.2702|2.7312|0.439|
|1|sigma2.0_ridge0.1_gray_world|12288|2.9693|2.1399|0.439|
|1|sigma2.0_ridge1.0_gray_light|12288|4.5370|2.3331|0.443|
|1|sigma2.0_ridge1.0_gray_world|12288|3.2636|2.3331|0.443|
|2|sigma2.0_ridge1e-05_gray_world_biasTrue|12288|2.7446|2.0674|0.718|
|2|sigma2.0_ridge0.0001_gray_world_biasTrue|12288|2.7448|2.0678|0.747|
|2|sigma2.0_ridge0.001_gray_world_biasTrue|12288|2.7468|2.0714|0.585|
|2|sigma2.0_ridge1e-05_gray_world_biasFalse|8192|3.3792|3.0981|0.489|
|2|sigma2.0_ridge0.0001_gray_world_biasFalse|8192|3.3792|3.0983|0.438|
|2|sigma2.0_ridge0.001_gray_world_biasFalse|8192|3.3794|3.0999|0.440|
|2|sigma4.0_ridge1e-05_gray_world_biasTrue|12288|2.8727|2.7325|0.447|
|2|sigma4.0_ridge0.0001_gray_world_biasTrue|12288|2.8726|2.7326|0.420|
|2|sigma4.0_ridge0.001_gray_world_biasTrue|12288|2.8720|2.7337|0.431|
|2|sigma4.0_ridge1e-05_gray_world_biasFalse|8192|3.3954|3.1634|0.434|
|2|sigma4.0_ridge0.0001_gray_world_biasFalse|8192|3.3956|3.1638|0.434|
|2|sigma4.0_ridge0.001_gray_world_biasFalse|8192|3.3976|3.1674|0.433|
|2|sigma8.0_ridge1e-05_gray_world_biasTrue|12288|3.1365|2.9817|0.809|
|2|sigma8.0_ridge0.0001_gray_world_biasTrue|12288|3.1364|2.9818|0.768|
|2|sigma8.0_ridge0.001_gray_world_biasTrue|12288|3.1353|2.9821|0.426|
|2|sigma8.0_ridge1e-05_gray_world_biasFalse|8192|3.4523|3.2233|0.768|
|2|sigma8.0_ridge0.0001_gray_world_biasFalse|8192|3.4525|3.2235|0.435|
|2|sigma8.0_ridge0.001_gray_world_biasFalse|8192|3.4541|3.2254|0.434|

All34 outcomes have119/119 valid predictions. Complex Fourier storage retains conjugate symmetry;12288 is the corresponding real spatial parameter count, not the number of stored complex scalar components. No optimized inference latency/VRAM/export is measured. Fitting is CPU ridge linear algebra; these timings are not deployment timings. Fourier feature formulas come from the previously audited Apache2.0 FFCC-inspired implementation. Ground-truth angles, not skin DeltaE00, supervise heatmaps.

Mechanistic conclusion: on this source distribution the learned global prior helps. That does not show it transfers to unknown cameras. Removing spatial semantics does not make the problem disappear, but12k coefficients deserve a fair future transfer comparison and a stronger likelihood objective. Next independent alternatives: correction-set coverage, robust regional influence diagnostics and training-only semantic teacher, each with explicit matched ablations.
