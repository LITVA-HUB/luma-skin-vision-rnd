# Native-color sampling provenance

Reuse only verified original MSKCC TRAIN cache, CC-BY, and native instrument Lab.
No new data or pretrained weights adopted. Sampling probabilities use selected
TRAIN targets and people/sites only. Neither camera metadata nor any evaluation
target enters the probability construction. No images/identifiers/checkpoints
are included in Git; aggregate metrics and exact-byte bindings are retained.

Protocols frozen before their fits:
checkpoint/skin-color-sampling-pretrain-2026-09-11 and
checkpoint/skin-color-mass-pretrain-2026-09-11. The second is explicitly a
post-screen falsifier. Its training recipe was copied from our local frozen
first recipe with only sampler/protocol/output wiring changes; no external
implementation was copied. Exact full refits and probability identities pass.

Label-density balancing, importance sampling and regression reweighting are
established methods. Our partial result does not establish novelty or patentability.
Current paper review adds conceptual prior art only, not copied code or weights.
