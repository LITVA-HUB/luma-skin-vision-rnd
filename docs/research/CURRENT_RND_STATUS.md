# CURRENT R&D STATUS — 2026-09-10

**Runnable research infrastructure exists. Real skin-color accuracy and proposed technological advantage are NOT MEASURED.**

## COMPLETED

- Isolated Git repository, root AGENTS, README, locked dependencies and versioned schema. Evidence: source tree, uv.lock, [plan](implementation_plan.md).
- Archive context imported without execution and nine file hashes verified. Evidence: [provenance](../context/provenance.json). No original Luma source was independently verified.
- sRGB/D65/2° conversions and ΔE00 validated on all 34 Sharma reference pairs; schema/leakage/EXIF/mirroring/ROI/CCM/selection/calibration/model-loading/ONNX/memory/artifact-tamper tests. Evidence: **48 passing tests**, [verification.json](verification.json).
- A0/A1/A2 and compact C/C+/proposed train → OOF error model → calibration → test evaluation execute on deterministic synthetic data. Evidence: [registry](experiment_registry.md), [baseline report](../benchmarks/baseline_report.md).
- FP32 ONNX parity and batch-1 CPU/RTX4060/ORT timing measured for three learned models. Evidence: [deployment benchmark](../benchmarks/deployment_benchmark.md). No production validity follows.
- Data acquisition/repeatability/privacy protocols, real-training gate, independent review fixes and conservative API contract. Evidence: [DATA_REQUIRED](../data/DATA_REQUIRED.md), [review](review_record.md), [contract](../architecture/api_contract.md).
- Primary-source research: 13 close analogues, narrowed novelty hypothesis, separate code/weights/data licensing and technical Skolkovo/IP evidence map. Evidence: [prior_art](prior_art.md), [licensing](../ip/licensing_inventory.md), [evidence map](../skolkovo/evidence_map.md).

## IN PROGRESS

Scientific definition of measurement-aware reliability and strong C+ tuning protocol. The current attention is weakly supervised through Lab loss; it is not an independently validated measurement mask. Some recent prior-art full texts/quantitative tables and exact third-party artifact rights still need additional review before adoption/public claims.

## BLOCKED

G1 instrument validity/repeatability, G2 actual cross-camera/light problem, G3 real ML value, G4 proposed versus strong C+, G5 useful selective risk and G6 faithful compact deployment require a physical dataset. No real protocol-approval file, participant images or measured targets were fabricated. All six gates remain NOT MEASURED.

## FAILED / NEGATIVE OBSERVATIONS

No final engineering check failed. During development, tests exposed cheek inversion, EXIF mismatch, nonfinite acceptance, ambiguous bootstrap units, excessive cache memory and stale ONNX attribution; these are fixed. A Windows encoding failure in the verification transcript writer was also fixed by explicit UTF-8 child-process settings.

The synthetic two-epoch neural prototype is worse than classical A2 on the toy generator. This is an engineering observation, not a real-world negative result and not evidence for commercial deployment. See [negative results](negative_results.md).

## NOT STARTED

Instrument-target dataset collection, real segmentation/reliability supervision, tuned strong C+, rigorous unseen-device/light tests, full real ablations, paired-consistency objectives, risk-control guarantees under a preregistered protocol, teacher/student, production FP16/INT8/TensorRT, product-shade matching validation and live Luma integration.

## NEXT BEST ACTION

Run the 10–15 participant physical pilot with three camera pipelines, four lighting conditions, two capture repeats and at least three instrument readings per anatomical cheek per session. Lock canonical mirroring/registration and the repeatability threshold first. Run validation and repeatability before granting G1; then estimate study variance/sample size. Do not buy more compute or tune on a test set to compensate for missing measurement evidence.
