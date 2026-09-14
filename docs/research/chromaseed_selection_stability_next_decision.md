# ChromaSeed-S: decision after fixed-OOF selection diagnosis

2026-09-13. **Progress; broad goal active.** No new model, new training gain or independent phone-face accuracy claim. The W verification/progress was checked and finalized before this diagnostic. The original repository and application ZIP remain unchanged.

## What the result resolves

All 891 W family-candidate scores reproduce from the original nine OOF archives. In the registered 210 score-deletion choices, static methods switch 4/126 times and iterative families 34/84 times. Their exact selected correction configurations recur in only 17.09–38.88% of camera-stratified bootstrap draws, compared with 46.48–73.02% for static methods. All 15 fixed runner-up paired ranges include zero. These are conditional, overlapping, retrospective diagnostics, not significance tests or new held-person validation.

At the same time, shared/local static SLR choices survive all single-person deletions and select alpha below 0.1 in approximately 99.99–100% of the resamples. Instability of a few fit-side people therefore does not explain away the weak-alpha preference. The prior outer regressions cannot be repaired just by assuming a stability rule will return the old settings. Camera changes are confounded with person and target-color changes; the cause remains unresolved.

The primary diagnostic took 1.585 seconds excluding interpreter import, and did no training. Eight tests passed. The final independent audit reconstructed all 12,474 candidate/person errors, 891 means, 15 original choices, 60,000 stratified draws, 300,000 bootstrap choices/ranks, 210 deletion choices and 30 paired distributions. Person-score drift was zero; paired-gap drift at most 3.55e-15. The earlier 20,284-byte compact model and roughly 0.017 ms prepared-feature response are prior measured results; this diagnostic did not improve them.

## Next useful work — planned, not launched

Before another accuracy search, register an input/target support diagnosis on original TRAIN. Quantify how camera groups differ in color-feature support and native-Lab coverage; assess person-held-out camera predictability, including controls matched on target-color support. Fit any normalizers/classifiers within their fit folds only. Report matching coverage and unmatched people rather than extrapolating to them. Never turn camera predictability or target overlap into a claim of camera causality: people and acquisition remain confounded.

This should determine whether a camera-aware calibration or color-normalization experiment has a concrete rationale. It is not approval to select one from exposed outer errors. Do not run another broad penalty/step sweep or add query-time recursion merely to produce more configurations. Preserve stronger previous candidates and all negative results. Real ordinary-phone facial quality remains unvalidated; no new facial dataset is available here. Useful local diagnosis remains, so the broad goal is neither achieved nor blocked.

## Resume facts

Run `experiments/runs/chromaseed_selection_stability_v1`. Primary PID13944 returned exit 0 directly; initial audit56446 and final tightened audit30639 returned exit 0. Report generator returned exit 0. No S or W job remains. Source SHA256 `88189161a7cee0c87d8ed5da2695083374564984c01ad77e3711f5fd49b3efea`; results SHA256 `2490104c6919306d197c46f5b876f03319d01d306ed93cd06dcb0864b4a8243a`.

Preserve S core/runner/tests/protocol and all 35 source bindings, 23 input bindings including the now-frozen W verification receipt. No delegation, publication or data acquisition. Only TRAIN target/patient/device and inner OOF loaded; no colors, images, tokens or old validation/calibration/test.

[Report](../benchmarks/chromaseed_selection_stability_v1/report.md) · [Verification](../benchmarks/chromaseed_selection_stability_v1/verification.json) · [Goal ledger](chromaseed_active_goal.md).
