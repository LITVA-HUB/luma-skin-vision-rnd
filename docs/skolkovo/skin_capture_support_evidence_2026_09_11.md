# Skin color support experiment evidence

Implemented: exact observed-patch training augmentation, matched auxiliary-label
control, source provenance replay and a frozen-expert routing diagnostic.

Measured: 45 fits on real MSKCC photographs and actual instrument-native Lab.
SLR-to-iPod source mean improves from 5.0193 to 4.8328 DeltaE00 for stratified
sampling, but mixed-source error worsens and reverse transfer is not competitive
with the strongest earlier control. No general improvement is demonstrated.
Known source labels do not rescue the frozen experts in a TRAIN-only diagnostic.
[Results](../benchmarks/skin_capture_support_v1/report.md).

Still unvalidated: ordinary phone facial color accuracy, unseen modern devices,
reliable product acceptance thresholds, cosmetics matching and novelty. Angle
of illuminant estimation is not the product endpoint. The independent MSKCC
test is unchanged and Proposed has not beaten the strongest ordinary ensemble.

Evidence is reproducible implementation and an explicitly preserved negative
research result. It is not proof of patentability, an approved Skolkovo legal
classification or permission to claim product readiness. No external publication
or participant-data release occurred. No new external data or weights adopted.
