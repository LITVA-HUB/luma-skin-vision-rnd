# Negative-results register

Status: **NO REAL EXPERIMENTS RUN; results NOT MEASURED.** Add one immutable entry for every failed gate, null result, harmful ablation, protocol failure, or unsupported domain.

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

