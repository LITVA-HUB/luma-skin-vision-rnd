# Negative-results register

Status: **NO REAL EXPERIMENTS RUN; results NOT MEASURED.** Add one immutable entry for every failed gate, null result, harmful ablation, protocol failure, or unsupported domain.

## ENG-SYN-2026-09-10 — completed engineering observation

Evidence: canonical six runs in [experiment_registry.md](experiment_registry.md), source SHA256 `b12faad7b06b77262cc7c69fb8a3ad469d8c3e8035a64f4b2406060e92fe3335`, dataset SHA256 `d5fb130c739634a24772245f62234b502be72bd3899d4f525fc1e86fb13b9f42`.

On 96 held-out SYNTHETIC patch images from 12 toy subjects, all methods selected 76 images (requested 80%, realized 79.1667%). A2 mean error was 2.0711 ΔE00 at that coverage; proposed_v1 was 22.1875. C+ was 24.3790. Two epochs with random initialization did not produce a neural model competitive with a train-fitted color matrix on the generator. This is not a real-data gate failure, statistical efficacy claim or rationale for abandoning the hypothesis; the smoke was designed to exercise code and was not tuned for accuracy. No stronger real-world interpretation is permitted.

Gray World also worsened toy error relative to no correction. The generator contains skin-colored fields, violating its neutral-scene-average premise. Preserve this failure rather than presenting automatic white balance as universally beneficial. Next action is physical target/protocol validation, not synthetic metric optimization. Full curves, paired subject bootstrap and exact run artifacts remain in [synthetic evidence](../benchmarks/synthetic_smoke_evidence.json).

## Entry template

- Entry ID/date/status: [value]
- Research question/gate: [RQ/G]
- Frozen dataset, split, code, config, and artifact hashes: [value]
- Planned hypothesis and decision criterion: [value]
- Method and strong comparator: [value]
- Sample/subject/domain counts and exclusions: [value]
- Result with paired subject-cluster interval and full risk–coverage reference: [value]
- Diagnostics checked without changing the test result: [value]
- Interpretation and limits: [value]
- Decision: stop / repair protocol / gather data / smallest pivot
- Follow-up registered before new evaluation: [value]

Do not erase or relabel a negative result after changing the method. Create a new experiment and link it. Do not tune a threshold on test data, substitute a weaker baseline, select a favorable seed, or narrow the population after seeing results.

Candidate evidence-driven pivots include calibrated-device mode, few-shot device calibration, two-image guided capture, a simpler classical-plus-ML hybrid, or active recapture. A pivot requires its own target, domain, baseline, and pre-registration; it does not retroactively pass the original gate.
