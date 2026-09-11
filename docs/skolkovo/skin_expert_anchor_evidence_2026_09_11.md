# Expert supervision versus removal: skin-color evidence

Implemented: a genuine single-head compact model, uniform and mode-conditional
native-Lab expert supervision, observed-patch training support and a common
input-only acceptance rule. No camera identity is required at inference.

Measured on real original MSKCC photographs with actual site instrument Lab:
72 locally reproduced source-development fits. Removing the decomposition and
adding paired patch support improves the plain model's transfer error in both
directions (5.8301 to 5.4150; 5.5087 to 5.3047), with worse mixed-source accuracy.
It does not beat the strongest historical controls in both directions. Explicit
mode supervision improves internal head/label diagnostics but not overall skin
accuracy. Selective acceptance using input novelty fails in reverse transfer.

[Report](../benchmarks/skin_expert_anchor_v1/report.md) and
[decision](../research/skin_expert_anchor_next_decision.md) retain all outcomes.
216 color-array replays, 12,672 independent scalar color cases and 337 passing
tests provide implementation evidence, not proof of scientific superiority.

Still unvalidated: ordinary phone facial color precision, robust unseen-device
generalization, calibrated per-image guarantees, cosmetics matching and novelty.
The exposed independent MSKCC result remains unchanged; ordinary fusion remains
stronger than Proposed. Original MSKCC CC-BY; no new external weights/data or
publication. No approved legal classification, patentability or product-readiness
claim follows from this experimental implementation.
