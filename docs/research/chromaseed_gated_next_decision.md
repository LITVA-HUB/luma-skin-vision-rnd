# ChromaSeed-G decision and next experiment

2026-09-13. This series is **progress** toward the active compact/fast/high-quality goal. It is not broad-goal completion or a universal model promotion.

G executed D's proposed acquisition-conditioned residual. Exact shared 128-center bases plus a small gate/residual give mixed perceptual hard5.2851 and soft5.2951 versus base5.3901 and uniform5.3617 ΔE00. Hard adds1,689 numeric bytes for21,973 total, answers in11.7µs using the standalone NumPy consumer, and fits734 prepared examples in35.53ms. The NumPy path also speeds the unchanged base to9.4µs, so acceleration is not attributed to gating. Cached active arrays occupy44,640B, excluding runtime overhead.

The mixed held set has six people and has been repeatedly explored. Hard improves four of six versus base; the descriptive fixed-prediction range includes zero. Both routed transfer directions are exact base fallback because their fit side has only one acquisition group. Uniform residual correction worsens both directions. No unseen-phone accuracy or causal camera correction is established.

Completed:2,016 readouts in12 shared banks, including864 fallback copies/72 exact P bases;360 residual coefficient solutions,36 perceptual base solves and4 gate fits;24 choices frozen before72 final evaluations. Primary workflow6.671s. All72 selected models independently refitted,12 positive route probes,285,600 OOF and28,752 selected query rows verified.23 primary tests plus3 separate portable tests pass.112 full timing fits including warmups exactly reproduce stored arrays. Primary PID36812, audit27067 and runtime/report terminal exit0. Preserve45 primary source bindings,27 P-parent artifacts,2 D input bindings and all older locks; final verification also binds the NumPy consumer and postprocessing.

## Next bounded question: gate stability

Does the hard gate's small ordinary-input advantage survive plausible small changes in prepared color statistics, or does its discontinuity create large output jumps? Use frozen selected models, not a newly tuned family or gate threshold. The follow-up is planned, not launched at this decision snapshot.

1. Register a limited deterministic grid of positive per-channel affine transformations of encoded RGB statistics. Update quantiles/means/std consistently; correlations remain unchanged. Do not clip individual quantiles and pretend the resulting mean/std are pixel-exact. Transformations that leave the legal encoded-RGB range require an explicit coverage rule or a bounded contraction construction chosen before inspecting outcomes.
2. Compare all four base/uniform/soft/hard families for both losses on the same original TRAIN role queries, all three seeds. Record output ΔE00 drift from original predictions, change in error against the unchanged instrument target, gate flips, gate margins and group/person distributions. Keep original controls and zero perturbation identity. Label all transformations synthetic; no claim they reproduce an actual camera or lighting distribution.
3. Independently verify affine feature algebra against small synthetic pixel arrays and direct kernels; separately probe the analytic hard-gate boundary, whose nearest unconstrained feature perturbation may not correspond to a realizable skin image. Report that distinction and all registered strengths.
4. Freeze protocol/code/model references before the diagnostic. No correction fitting, outer-selected threshold, worst-case tuning, new data, delegation or publication. If hard gating is fragile, retain the matched soft and shared controls before proposing a later remedy with its own protocol.

This is useful local progress while genuine phone-face acquisition remains unavailable. It cannot replace an independent person/device/lighting benchmark or prove shade matching. Preserve the active goal and classify each continuation from the measured result.

[Report](../benchmarks/chromaseed_gated_v1/report.md) · [Verification](../benchmarks/chromaseed_gated_v1/verification.json) · [Model card](../architecture/chromaseed_gated_model_card.md) · [Active goal](chromaseed_active_goal.md).
