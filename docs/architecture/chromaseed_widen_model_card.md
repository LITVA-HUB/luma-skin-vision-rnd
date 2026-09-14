# Luma ChromaSeed WIDE

Research skin-region Lab regression, not face identification. One color36 vector plus64x18 existing patch descriptors. Shared local ReLU encoder, mean/spread/max pooling, wider ReLU color head. FP32 payload, cached FP64 NumPy consumer. Capacities1179/30915/60611/110979/832259; initial function matches the corresponding NP head. No ensemble inference.

Only original TRAIN966rows24people; three historically reused person-disjoint role splits, inner selection. Instrument-native D65/10-degree Lab. Synthetic affine stress retains original target by assumption. No new ordinary-phone face/end-to-end validation or Skolkovo eligibility claim.

See ../benchmarks/chromaseed_widen_v1/report.md and summary.json for all selected capacities, adverse transfer outcomes, actual CPU response and full warm-training costs. Do not export a favorable outer checkpoint in place of the frozen selection. Previous pixel/R/pooling architectures already exist; this is a matched capacity study, not a novelty proof.
