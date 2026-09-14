# Stability addendum for ChromaSeed-G

The [G model card](chromaseed_gated_model_card.md) is preserved with its original verification. GS adds synthetic sensitivity evidence and changes no payload or predictor.

Use the soft gate as the continuous research option; retain hard for comparison. The original mixed person means are soft5.295118 and hard5.285104 ΔE00. On deliberately constructed legal gate-boundary pairs, mean jumps are soft0.010986 and hard1.927893, with hard maximum7.817570. Those pairs can be far from the original input: median15.69% color mixture, while their mutual separation is≤0.0002 encoded RGB. This is not a real-photo failure-rate estimate.

Soft does not eliminate the shared model's broader color sensitivity. At16/255 mixture, worst-of-eight reference errors are approximately9.80 for the base, soft and hard models alike. A constant answer could have perfect stability while being inaccurate, so both original target accuracy and sensitivity controls are required. Neither variant has validated ordinary-phone face/shade-matching quality.

Representative unchanged [soft seed17 weights](../../experiments/runs/chromaseed_gated_v1/selected/mixed/perceptual_soft_s17.npz) can be consumed by the same [NumPy predictor](../../scripts/chromaseed_gated_numpy.py). The model still stores21,973 numeric bytes; previously measured soft inference11.8µs and full734-row fit34.51ms exclude image processing and process/library memory. GS does not repeat those timings or select a production model from new independent people.

[All GS doses and controls](../benchmarks/chromaseed_gate_stability_v1/report.md) · [Verification](../benchmarks/chromaseed_gate_stability_v1/verification.json) · [Next learning experiment](../research/chromaseed_gate_stability_next_decision.md).
