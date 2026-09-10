# V4 final integration review

Review date: 2026-09-11. Immutable candidate: **`c99ca7baa897cbae31c65f18d56b740ecb5b29cc`**, compared with base `75ee907`. Verdict: **PASS — no new load-bearing scientific, evaluation, provenance, or implementation finding identified.** This passes the integration review for an explicitly negative source-development screen; it does not promote the proposed model or certify novelty, independent generalization, calibration, or product readiness.

## Scope and independent verification

Reviewed the committed change inventory, model/geometry and runner integration, benchmark/replay/report code, prior review closures, protocol, selected receipts, report, current research status, next decision, IP provenance and Skolkovo addendum. Avoided dumping the 22,595-line summary; inspected the actual committed prediction/metric/history/timing artifacts instead. Read code and receipts, without fitting a model or decoding real images/GT. No code or data edits, checkpoint changes, downloads, commits, or subagents.

Observed on the candidate checkout:

```text
git rev-parse HEAD
c99ca7baa897cbae31c65f18d56b740ecb5b29cc

.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
185 passed, 14 warnings in 29.85s

.venv/Scripts/python.exe scripts/cc_v4_report.py --self-check
passed: atan2 exact null; action-grid oracle; stable risk coverage
```

Warnings came from existing ONNX export deprecation/tracing paths, not failing V4 checks. The working tree was clean when the test run began. These tests and the report self-check are constructed verification, not additional real accuracy evidence.

A separate read-only probe used `git show <full-commit>:<path>` to load committed artifacts. It independently selected the earliest minimum eligible step-2 checkpoint from each full history, recomputed mean and 95-of-119 raw risk80 from saved per-image prediction errors and stable risk ordering, and checked all nine timing receipts' mode/status, 100 samples per stage, median arithmetic, and observed 25/51/103 query counts:

| Mode | Selected epoch | Reproduction mean | Raw risk80 |
|---|---:|---:|---:|
| posterior | 77 | 2.1808844108 | 1.8945554081 |
| action | 87 | 2.3169029856 | 2.0088624182 |
| transport | 63 | 2.3888876014 | 1.9998393957 |

These agree with the committed report. This probe recomputes summaries from saved errors; the separate recorded NumPy geometry audit and six CPU checkpoint replays supply prediction/GT and checkpoint-to-prediction correspondence. I reviewed their code/receipts without claiming to have rerun the real-data replay in this review. Root independently verifies the raw Git-blob archive bindings and original milestone tags; that complementary audit is not represented as my own execution.

## Load-bearing conclusions

**The reported negative result follows from the evidence.** Transport loses primary mean to both the coherent posterior and generic action router, and loses raw risk80 to posterior. The report does not suppress action's fewer >10-degree outliers or infer tail-uniform posterior superiority. It keeps separately selected point-readout checkpoints distinct from a separately trained conventional point model. Historical V2/V3 values are context, not matched V4 causal controls.

**The original mechanism objection is resolved in the implemented experiment.** The committed model uses a nonlinear corrected-simplex transform and includes the action-conditioned null with fixed simplex evidence plus relative action. Thus the first-layer affine-reparameterization criticism of the rejected log-shift draft is not falsely presented as a solved architectural innovation. The remaining claim is a finite-capacity physical inductive bias; the result supplies no measured advantage for it. Action-dependent weights are correctly called a decision field rather than one posterior.

**Training and evaluation boundaries are consistent.** The runner freezes the source identity, numerically materializes only fitting/development-validation roles, and samples actions without a GT argument. GT enters supervised reproduction targets. Checkpoint selection is fixed to step 2; step 4 is diagnostic. The aggregate report checks role membership and group disjointness, exact selected history entries, saved arrays, oracles, risk curves, and benchmark provenance. Invalid source inputs were separately audited; there is no evidence of an invalid-row omission inflating the current result.

**Refinement is described honestly.** The encoder is cached and the action is updated through bounded search. This is not a recurrent hidden-state network or proof of LLM-like reasoning. Retaining candidates makes predicted risk nonincreasing, while the report explicitly exposes true-error worsening. Candidate-oracle performance is label-dependent coverage evidence, not an attainable learning bound. Equal-query nonadaptive comparison remains a future requirement before attributing a gain specifically to iterative adaptation; no such gain is claimed here.

**Compute and reliability claims remain appropriately bounded.** All timing rounds are retained, including anomalous desktop timings. The report gives synchronized device-resident batch-one scope and excludes decode/upload/rendering; it does not infer search overhead without direct timing or claim a V4 export result. Raw validation ranking is repeatedly identified as uncalibrated. The one-seed, reused-validation and pre-field warmup divergence caveats are visible in the report and status, preventing a stable architecture-effect claim.

**The next decision is a hypothesis, not a retrospective cure.** Retaining posterior/direct as the current development reference is consistent with the means. Policy-selected-action training, paired initialization, locked randomness, multiple seeds, teacher controls, and independent reliability/calibration roles are reasonable proposed tests. The oracle gap does not identify its cause uniquely or prove richer context will solve it; the next-decision document explicitly says the labels cannot deploy and ambiguity may remain. This screen does not justify consuming the unseen population for repeated architecture selection.

**Product/IP positioning respects the evidence boundary.** Current status and the Skolkovo/IP documents preserve the negative V4 result, earlier negatives and bounded V2 positives. They distinguish public linear-RGB illuminant reproduction from skin Lab/DeltaE, arbitrary ISP behavior, cosmetics utility, independent validation, patentability and freedom to operate. Teacher sources are studied separately from adopted assets; no external pretrained assets are represented as used. Assistant reviews are described as development assistance, not external peer review or inventorship evidence.

## Remaining limitations, not new blockers

Existing review limitations remain: angular epsilon smoothing provides finite autodiff behavior but does not establish literal global C2 smoothness at every max tie; the exact sine-squared derivative path is separate. The source screen does not exercise later Sobolev/teacher/calibration stages. Runner/benchmark rejection-path tests can be broadened before reuse, and the report generator's current scoped containment/rollback protections are not a general transaction guarantee against arbitrary process termination. None changes the frozen result or its stated conclusions.

This verdict applies to the exact candidate commit above. Any later documentation-only archive/lineage additions require their own identity record; they were not silently included in this immutable review. PASS concerns integration quality, not advancement through the research promotion gates.
