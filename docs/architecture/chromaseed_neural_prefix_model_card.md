# Luma ChromaSeed-NP model card

Exact fixed-prefix exports of existing ND networks; no new primary training. One finite prepared color36 vector -> native instrument D65/10-degree Lab3. Standalone NumPy Predictor in scripts/chromaseed_neural_prefix_numpy.py takes one (36,) input; module predict takes nonempty Nx36. FP32 normalization, cached FP64 arithmetic. Input images, segmentation/detection, feature extraction and runtime/code overhead are not included in model size or latency.

First state-input rows are pruned because inference starts at zero. Blind4 exports only its selected head, since none consumes state. Preserve original schedule denominator K and clean-head semantics for other prefixes. The final unused state update is removed. All original training is still required and charged in this study, including e2e gradients from later heads. No backprop-free or training acceleration claim.

Single h16 head:643 parameters /2,886 numeric B /5,456 cached-array B /~6.3us one-thread response. Two metadata uint8 scalars count in payload; NPZ archives and family strings are separate. Blind compact/quality policies choose heads2/3/2 inside the inner folds; errors mixed5.77054 /forward8.29435 /reverse8.58953. FG5.43865/8.59700/8.70502. Mixed is worse; transfer uncertainty does not establish superiority. Complete original blind4 fits plus prefix export~304–309ms on RTX4060. Full controls and all positive/negative prefixes are retained.

Conditional inner selection follows unchanged ND rate/checkpoint decisions. Compact tolerance+0.10 applies to measured inner error only and is not a cosmetic requirement. Reused original TRAIN roles overlap, and people/camera are confounded. No ordinary-phone facial accuracy, cosmetic shade-match success, clinical use, novelty or Skolkovo eligibility established by these measurements. Existing upstream data/code licenses apply; no new assets or external source implementation adopted.

All153 exports/references,405 inner exports and2,016,234 actual final consumer calls independently verified.90 full reconstructions yield bitwise identical original and prefix weights;20 behavioral tests pass. Source and evidence are sealed; report script re-verifies read-only.

[Evidence](../benchmarks/chromaseed_neural_prefix_v1/report.md) · [Protocol](../research/chromaseed_neural_prefix_v1_protocol.md).
