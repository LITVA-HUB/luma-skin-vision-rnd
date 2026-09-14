# ChromaSeed-GS decision: continuity verified, common sensitivity remains

2026-09-13. This continuation is **progress** toward the full compact/fast/high-quality user goal. No new skin-color model was fitted, and the goal remains active.

All72 frozen G models were evaluated under33 registered bounded encoded-RGB contractions, with all2,376 transform cases/948,816 predictions and288 dose summaries independently checked. At constructed legal boundary pairs, perceptual hard jumps1.927893 ΔE00 by person mean versus soft0.010986, maximum hard7.817570. The534 unique query/anchor pairs involve185 original rows/six people and repeat across six model/loss/seed cases, producing3,204 pair evaluations. All actual FP32 signs cross;25,632 matched-control predictions and1,392 unconstrained projections independently agree.

The tiny pair separation≤0.0002 RGB is not its distance from an original image. Roots range from0.258% to24.969% mixture, median15.694%; only2 roots lie below1/255 and5 below4/255. At the registered1/255 and4/255 doses, any-anchor sign flips occur on1/232 and2/232 source rows. This proves the constructed discontinuity, not frequent real camera failures. Unconstrained nearest-feature projections are also not certified image perturbations.

Smooth routing removes that jump, but its broader color sensitivity remains: at16/255 mixture, worst-of-eight person-mean reference errors are base9.8045/soft9.8064/hard9.8081. Hard remains slightly better on worst-case means at the two smallest doses. Do not present soft as universally more accurate or infer a robustness guarantee from continuity. Retain soft as the continuous candidate and hard as a comparison; preserve the shared base and every stronger historical control.

Primary PID39468 returned exit0; independent audit session1166 returned exit0. Eight tests passed in1.51s before execution. Diagnostic5.474s and independent audit14.338s are not training/inference benchmarks. Source5102e3ba875d0a21f71d13dd538edf55b04b1efc0deb4110d19d66522c935b26, results41e8a9060ebf12eed131534e5aced4ac61605688537c27f0a0ea9050ef9a4bc0. Preserve50 source/148 G-input bindings, including G verification and every selected model/prediction. No GS job remains.

## Next actual learning experiment: controlled mild color augmentation

Planned, not launched at this decision snapshot. Study a compact analytical readout trained on original fit-side rows plus bounded mild affine RGB variations. Start with the already defined1/255 and4/255 doses, not the extreme stress dose. Reuse the128-center basis learned only from original fit rows, keeping exported storage and prepared-feature response compact. Use continuous conditional and static controls. This addresses the user's fast local learning objective directly without backpropagation or a new topology search.

Before fitting, specify the exact grids, budget, model families, analytic objectives, normalizers and selection rules. If a joint shared/conditional readout is introduced, include its identical no-augmentation control so the augmentation effect is not confused with the change from G's frozen-base residual. Retain exact original G/P references and compare with a constant Lab fit-side baseline. Add ordinary stronger-ridge controls; smoothing the predictor toward a constant must not count as solving color estimation.

Augmented examples reuse their source person's target and must remain inside that person's fit fold. Normalize the total augmentation weight per original row so extra copies do not overweight people/sites or silently change the regularization scale. Fit acquisition gates only on fit-side data, with explicit single-camera fallback. Use all three person-disjoint inner folds and freeze every choice before final-role scoring. Report original target error and fixed synthetic sensitivity together, including clean-error regressions; avoid choosing thresholds from GS's exposed outer outputs.

The output remains an estimate from prepared skin statistics. Neither analytic fitting nor synthetic augmentation can establish unknown-phone face accuracy. No new-data acquisition, delegation, publication, image/token access or legacy validation/calibration/test is authorized for this planned series. A real independent acquisition benchmark remains necessary for the full user goal.

[Report](../benchmarks/chromaseed_gate_stability_v1/report.md) · [Verification](../benchmarks/chromaseed_gate_stability_v1/verification.json) · [Active goal](chromaseed_active_goal.md).
