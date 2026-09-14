# ChromaSeed WE model card

Unchanged WIDE patch/color regression architecture and consumer. WE changes only optimization rate pairs and candidate checkpoints. Model capacities30915/60611/110979/832259, FP32 exports and cached FP64 CPU arithmetic. No ensemble inference. Inputs are color36 plus64x18 skin-patch descriptors; output instrument-native D65/10-degree Lab.

Original TRAIN966rows24people only; person-disjoint inner folds, historically reused outer roles. Tiny remains a fixed old control. See ../benchmarks/chromaseed_widen_early_v1/report.md and summary.json for the full selected outcomes, adverse cases and measured construction/response costs. Synthetic affine stress is target-invariance by assumption. No ordinary-phone face/end-to-end quality or patent/Skolkovo claim.
