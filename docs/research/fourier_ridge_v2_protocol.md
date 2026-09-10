# Fourier spike phase2: remove the learned global color prior

Adaptive DEVELOPMENT search after phase1's16 candidates, not a fresh benchmark.
Phase1 winner sigma2/ridge.001/GW had2.747 mean and2.071 raw risk80 on119.
The best ridge lies on the phase1 boundary; test lower regularization and
broader heatmap targets. Fixed phase2 matrix: sigma2/4/8; ridge1e-5/1e-4/.001;
bias on/off; fixed temperature200 and gray_world unwrapping.18 candidates.
Same real1126/119, unchanged preprocessing, no external/phone data.

Bias removal inverts the assumption that a fixed global illuminant prior helps.
Mechanism: retain only correlation with input color statistics, so inference
cannot emit a camera-specific learned heatmap independent of the image.
Failure mode: loss of useful source illumination prior, worse error and weak
identifiability. Cheapest falsifier: paired on/off source development results.
8192 active real coefficients without bias versus12288 with bias. Do not call
this a novel algorithm or use selection-adjusted validation as fresh evidence.
Retain both phases, all34 candidate outcomes, worst tails and negatives.
