# Independent manual review: V4 model core

Reviewed 2026-09-11. **PASS for the amended Task 1 model-core contract; no blocking finding.** This is a manual scientific/code review with constructed CPU verification, not an external review service or an accuracy evaluation. No source/test edits, data access, GPU execution, model downloads or uploads were performed by this reviewer.

## Reviewed identities

| File | SHA256 |
|---|---|
| scripts/cc_v4_model.py | 1c7c4b54c3b07bfbaf9e26a689e08ca0e511532b62403421664ed7c18054ae68 |
| tests/test_cc_v4_model.py | f0ec240c9354a39a154785fd50f3a838c2ef6c120daa5cefe81614f52153261d |
| docs/research/cc_v4_spec.md | 2ae0c34bfd6d2c4898ea1e916510912ce0dbc96b2ddc8563cbd62841b17ae53a |

The reviewed spec includes nonlinear corrected-simplex transport and the additional generic action-conditioned null. It supersedes the affine log-transport draft; this review does not rehabilitate that earlier design.

## Load-bearing checks

**Physics and gradients.** Residual RGB is exp([bR−aR,0,bB−aB]) up to a positive common factor. Pairwise squared difference D divided by 3 sum(r²) is algebraically the specified sin² reproduction cost. atan2(sqrt(D),sum(r)) is the reproduction angle; only the angular norm receives the explicitly declared tiny smoothing. Weighted angles and weighted sin² remain separate, and selection uses weighted angle. This avoids interpreting arcsin(sqrt(mean sin²)) as a mean angle. Bounded positive local hypotheses and actions keep the internal evaluation numerically well behaved. Tests cover first/second derivatives and the exact sin² null; the small angular smoothing qualification below remains applicable.

**Evidence transport.** Mean and RMS are computed in linear RGB before conversion to log ratios. A positive channelwise divisor commutes with these statistics, apart from the declared floor, so corrected_simplex(mean,a) and corrected_simplex(rms,a) represent the stated physical operation. The nonlinear simplex map is present. The router receives 48 local + 64 global + 2 corrected mean + 2 corrected RMS + 2 relative-action coordinates, totaling 118.

**Null fairness.** Transport uses candidate-corrected color; action mode fixes that color at the point while retaining the action displacement; posterior additionally zeros the displacement. The action null therefore controls generic action dependence, rather than merely comparing against an action-independent posterior. All four module graphs have **3,097,189 parameters**. The 128 inactive first-layer router weights in posterior/direct are accurately disclosed. Equal stored size does not establish identical active capacity or FLOPs. Direct inference returns the point and its shared diagnostics; it is a policy/readout control, not an independently identical inference graph.

**Initialization and dimensions.** The encoder is scratch MobileNetV3-large features, weights=None. The point head starts at zero. The 16 deterministic proposal offsets are distinct, with initial maximum Euclidean norm .06; they are correctly described as a prior. Learned proposals are bounded within .5 per coordinate around the point. Cache dimensions, 4×4 pooling, xy ordering, and the 118-input local-vote head agree with the spec.

**Input handling.** Structural invalidity raises. Nonfinite/negative input, black images or a globally missing channel produce valid=false and finite neutral selection output. Common maximum scaling precedes moments and conversion of double input to default float32. Empty local cells use the declared finite floor. The single 32×32 training boundary explicitly reports the BatchNorm limitation; singleton evaluation works. Input tensors are not mutated. Autocast is disabled in the core as specified; caller double is supported, while half parameter conversion is explicitly rejected.

**Selection and budgets.** Stages use the specified radii .24/.06/.03/.015 and clamp to [−2,2]. The preceding decision remains the center candidate; later stages append the original point. Total query counts are 25/51/103, or one diagnostic point for direct. All three learned policies use the same search. Selection reuses the cache without another encoder call and has no image-dependent Python loop. Predicted risk can decrease while true error increases; the implementation/spec do not claim otherwise. A finite search is not a global optimum of the decision field.

**Training and leakage.** Encode/query/select accept no GT, label, identity or external state, and perform no cross-image adaptation. Query does not detach hypotheses or parameters. Constructed value-plus-action-gradient losses reach model parameters with finite backward results. Action-dependent weights define a decision field, not one coherent posterior, as stated in the spec. The runner must independently enforce GT-independent action sampling, source-only selection and held-out residual calibration; these protocol properties cannot be certified from the model file alone.

## Independent observed verification

Fresh command: .venv/Scripts/python.exe -m pytest tests/test_cc_v4_model.py -q

Result: **39 passed in 4.46 seconds**. The larger count than the original 33-test notification reflects the current parametrized refinement checks.

Additional reviewer-authored inline CPU probes, without importing the repository geometry oracle:

- NumPy cross-product/dot-product angle oracle for 3 batches × 16 hypotheses × 11 random actions: maximum angular discrepancy **1.4210854715202004e−14 degrees** away from exact null. Independent sin² identity discrepancy **4.440892098500626e−16**.
- Double-precision query followed by angular value plus squared action-gradient penalty: finite losses and all populated parameter gradients in transport, action, posterior and direct diagnostic modes.
- All four modes with K=1/2/4: finite outputs and nonincreasing stored predicted-risk trajectories within the numerical tolerance.
- Default float32 network with finite double inputs multiplied by 1e−250 and 1e250: valid=true, and **zero observed maximum difference** across context, local features, point/proposal actions and mean/RMS log chroma after common scaling for this constructed input.
- Entirely invalid all-black training batch: both rows refused, finite diagnostic loss and finite populated parameter gradients. This checks finiteness, not suitability of invalid rows as training examples.

The ordinary test suite additionally checks independent action-gradient formulas, finite differences, exact-null search, physical statistics, nonlinear transport derivatives, posterior invariance, generic-action responsiveness, an action-gradient penalty alone reaching the backbone/router/vote head, query domain assertions, repeatability and input immutability. These constructed checks establish numerical behavior, not benchmark accuracy.

## Nonblocking qualifications

1. **Angular smoothing is not globally C² in the literal mathematical sense.** The fixed epsilon is added after maximum normalization of residual RGB. This slightly breaks the common-scale cancellation, retaining a tiny cusp when the maximum changes, although computed autograd second derivatives are finite. At the null along aR=t, the limiting one-sided slopes are approximately −2 epsilon/9 and +epsilon/9 radians. The sin² cost used for the planned Sobolev target is unaffected. Keep the claim as finite autograd derivatives, rather than a global smoothness theorem. If a globally smooth angular objective becomes necessary, a homogeneous smoothing such as atan2(sqrt(D+epsilon² S²),S), S=sum(r), removes this dependence; that would change the declared tiny smoothing and requires verification. This is not a blocker for the present value-only screen or exact sin² derivative loss.

2. **Invalid rows must remain excluded by the integration protocol.** Validity is explicit and refusal is correct, but sanitized invalid rows still pass through standard BatchNorm during training. Therefore their presence can affect training statistics of other rows; the core does not promise that refusing a row makes it invisible to the batch. Drop invalid training examples upstream and do not score the diagnostic 90-degree constant as a physical target.

3. **No generalization/calibration guarantee was audited.** Source validation, measured selected-action regret, held-out populations, confidence calibration, actual runtime/VRAM and teacher/weight licensing belong to later integration and evidence. No model-core test justifies universal surface-color accuracy, a coherent posterior for action-dependent routing, or novelty from the nonlinear map alone.

No blocking correction is requested at the reviewed hashes. Full training remains a parent protocol decision after its integration pilot and runner checks.
