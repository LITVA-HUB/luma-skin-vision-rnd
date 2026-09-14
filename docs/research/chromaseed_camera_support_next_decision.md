# ChromaSeed-D: acquisition-conditioned readout is now a concrete hypothesis

2026-09-13. **Verified progress, full compact/fast/high-quality goal remains active.** The S turn was rechecked against immutable source/input/artifact hashes and live process state. D's primary/final audit/report are now terminal; no background job remains. D diagnoses the representation, not skin-color accuracy.

## Evidence

Three person-held-out folds,24 people,966 original TRAIN records,24 fixed diagnostic classifier fits. Linear color36 camera classifier AUC1.000 and24/24 people correct; RBF AUC1.000 but23/24 at its fixed zero threshold. Native-Lab-only AUC0.586/0.578. After fit-only linear regression of color36 on native Lab, residual camera AUC remains1.000. Thus a strong camera-group signal remains beyond that simple target-color relationship; this does not isolate camera causality.

Target matching within held folds gives0/2/2/4/8 pairs at calipers1/2/3/5/10DeltaE00. Color36 ranking remains perfect on represented subsets, but at caliper3 only4 of24 people are represented. No caliper is promoted as a validation threshold.

Color-feature query mass outside a fit-side95th-percentile nearest-neighbor radius: mixed1.52%,SLR→iPod28.31%,iPod→SLR54.20%. Native-Lab analogues10.61%,3.98%,9.34%. Weighted target lightness medians differ50 versus60, and acquisition remains confounded with people. The two distance spaces have separate radii/geometry. These numbers are neither calibrated OOD rates nor color-error estimates.

Eleven numerical tests pass. Independent SVD/metric/exhaustive-matching audit covers24 fits/7728 query predictions,8 pooled and24 fold metrics,15 matching problems,40 matched metric groups,6 support cases/5796 distances and2 camera target summaries. Maximum prediction drift4.53e-14, support-distance drift8.88e-16. Primary diagnostic0.6885s excludes import/audit and is not a production training benchmark.

## Next series, planned and not launched: ChromaSeed-G

Test a compact shared128-center kernel basis with an acquisition-conditioned residual readout. A small gate trained on fit-side acquisition labels selects or blends correction coefficients from color features; no measured Lab or query camera label is required at inference. Reuse successful exact kernel algebra and the shared predictor as a control. Shared centers avoid a second kernel/feature-extraction pass; report actual storage, fitting time, batch-one inference and all quality roles.

Before fitting, register a bounded set of controls and penalties, the gate training/calibration procedure, and selection order. Use person-disjoint inner folds for gate/readout tuning; freeze choices before outer scoring. If a fit split has only one camera group, use exact shared fallback rather than importing held-camera labels or statistics. That means this experiment can test adaptation among known acquisition groups; it must not be advertised as a demonstrated cure for unknown-camera transfer.

Compare the gated residual to an ungated residual of the same capacity, the frozen strong shared kernel/perceptual controls and any relevant oracle route only as a separately labeled diagnostic. Include zero residual so the model can retain its shared answer. Avoid claiming that high camera-classification AUC predicts lower skin-color error. Do not select a gate threshold/caliper from D or earlier outer outcomes. Ordinary-phone facial validation is still unavailable, but this small concrete model experiment is useful local work toward the user's requested dynamic connections. The goal is neither achieved nor blocked.

## Frozen resume state

Run `experiments/runs/chromaseed_camera_support_v1`. Primary PID37356 returned exit0 directly. First audit returned exit1 only at JSON serialization of a NumPy integer; the writer was corrected, all numerical checks rerun, final audit returned exit0 directly. Report returned exit0. No D/S/W process remains.

Source SHA256 `59ca583d3fad34e16d0ae003b2d9d89df453bc730b0040d557407c86f2cb10a5`; results SHA256 `79317e6bf3b76539004bf4a700be410d4bf9b7f0aad839da080df11349d83ded`. Preserve41 source bindings and2 parent bindings, including S verification, plus all prior locks. Only original TRAIN color/target/patient/site/device loaded. No old held-out partitions, raw images/tokens, downloads, delegation or publication. Original repository and Luma ZIP unchanged.

[Report](../benchmarks/chromaseed_camera_support_v1/report.md) · [Verification](../benchmarks/chromaseed_camera_support_v1/verification.json) · [Goal ledger](chromaseed_active_goal.md).
