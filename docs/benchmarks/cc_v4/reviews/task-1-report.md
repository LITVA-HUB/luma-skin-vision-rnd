# Task 1 model core report

Status: implemented and CPU-verified. All numerical scenes below are **SYNTHETIC** correctness probes. There is no measured public-benchmark accuracy, training outcome, real-image performance, surface DeltaE result, or novelty claim in this report.

## Scope and authorized refinements

Owned implementation files: `scripts/cc_v4_model.py`, `tests/test_cc_v4_model.py`. This report is the requested task artifact. No old source, lock, dataset, or experiment script was edited. No GPU training, download, commit, or subagent was used.

The original brief described a linear log-color transport router using `[mean-a, rms-a, a-point]`. The first implementation passed 29 CPU checks, but a parent pretraining design review correctly identified that this input transformation can be absorbed into the first linear layer. **That draft is superseded and is not a distinct transport contribution. No training ran on it.**

The parent explicitly authorized replacing routed color with physically corrected, centered simplex RGB and adding an `action` comparison mode before training. For each mean/RMS log chroma `z` and action `a`, `corrected_simplex(z,a)` computes `p=softmax([zR-aR,0,zB-aB])` and returns `(3*pR-1,3*pB-1)`. This bounded nonlinear representation uses actual image color and cannot be removed through that simple first-layer affine reparameterization. This observation establishes neither novelty nor empirical superiority.

The parent also accepted preserving standard MobileNet BatchNorm. A single 32x32 image works in evaluation; training this exact singleton boundary explicitly raises actionable guidance because the backbone reaches one value per channel. Larger training batches or image sizes avoid that BatchNorm limitation.

## Delivered interface and computation

- `CorrectionEvidenceNet(mode='transport'|'action'|'posterior'|'direct')` uses torchvision MobileNetV3-Large `.features`, `weights=None`, with unchanged standard BatchNorm. No pretrained weight adoption.
- `encode(image)` returns the specified point, 16 local hypotheses, 48-dimensional local features, 64-dimensional global context, raw-color mean/RMS log chroma, and row validity. The point head starts at zero and clamps each coordinate to [-2,2]. The local vote head sees original log color, context, features, and xy.
- Local hypotheses are point plus `.5*tanh(vote + proposal_prior)`. The explicit deterministic 4x4 prior has 16 distinct offsets, maximum Euclidean norm .06; it is not evidence inferred from an image. Learned residuals remain bounded coordinatewise by .5.
- Common maximum scaling occurs before raw moments and before converting a finite extreme FP64 input to default FP32. The backbone receives image divided by its global RMS, preserving color ratios and exposure invariance. Local zero statistics use `COLOR_FLOOR=1e-12` after common scaling. Network arithmetic is FP32 by default and FP64 with `net.double()`; autocast is disabled within network operations.
- Nonfinite pixels, negative pixels, black rows, and any globally absent channel cause `valid=false`. Sanitized neutral fallback keeps all diagnostics and positive unit output finite. Invalid risks are diagnostic constants of 90 degrees and 2/3 sin², not physical GT measurements.
- `query(cache, actions)` returns angular risk, sin² risk, and 16 normalized weights. Shape validation and a device-side finite/absolute-coordinate<=4 assertion enforce the query domain without Python scalar synchronization per GPU query. Query does not detach parameters or local hypotheses.
- Transport routes simplex mean/RMS at each queried action and the relative action. Action mode fixes simplex color at the point but retains relative action. Posterior fixes both color at the point and relative action at zero. Direct uses posterior-style diagnostic weights but selects only the point.
- `select(cache, steps=1|2|4)` runs the specified grid radii .24/.06/.03/.015, clips actions to [-2,2], retains the preceding decision as the grid center, and appends the original point after the first stage. Total candidate counts are 25/51/103 for transport/action/posterior, and 1 for direct. Both action and predicted-risk trajectories are returned. Grid querying reuses the encoded cache. Lower predicted risk does not guarantee lower true error.
- `forward(image)` returns selection plus context. No label or GT argument exists in encode/query/select, and no image-dependent Python loop is used.

## Analytical geometry and smoothing disclosure

`analytic_costs(hypotheses, actions)` returns NxKxH costs. From stable max-subtracted exponential residual RGB `r`, let `D=(rR-rG)^2+(rG-rB)^2+(rB-rR)^2`. The smooth sin² formula is exactly `D/(3*sum(r^2))`, avoiding cancellation at null and yielding exact zero for equal actions/hypotheses. It agrees with the separately implemented NumPy geometry oracle.

Differentiable angular cost is `degrees(atan2(sqrt(D+1e-16),sum(r)))`, with `ANGLE_NORM_EPS=1e-8`. This norm smoothing keeps action second derivatives finite at exact equality. Its absolute angular bias is bounded by approximately 5.73e-7 degrees in the declared scale convention; the exact-null value is approximately 1.91e-7 degrees. Smoothing affects estimated/training angle only. Measured GT reproduction angles must continue to use the independent exact atan2 oracle. No smoothing is applied to sin².

## Parameter accounting

All four modes have the same declared module shapes and **3,097,189 parameters**:

| Component | Parameters |
|---|---:|
| Backbone | 2,971,952 |
| Context head | 61,504 |
| Point head | 130 |
| Local projection | 46,128 |
| Local vote head | 7,746 |
| Router | 9,729 |

The 118→64 router's final two input columns are inactive in posterior/direct because their input is fixed zero: **128 scalar weights**, not 236. They are active inputs in action/transport. These are disclosed inactive parameters in the shared graph, not dummy padding. Equal stored parameter counts do not assert equal effective capacity. Direct selection also does not use the evidence decision rule, although shared diagnostics remain available.

## Observed verification

1. Before production code existed, `pytest tests/test_cc_v4_model.py -q` failed all **29 tests** on the explicit missing-core assertion (1.44s): observed RED.
2. The first linear-log draft passed 29 tests (3.79s). This numerical pass did not prevent the subsequent conceptual reparameterization finding; that limitation is preserved above.
3. A standalone action-gradient-penalty probe confirmed finite, nonzero backpropagation into vote head, router, and backbone without angular-loss rescue.
4. Tests for nonlinear physical router inputs, the missing action mode/helper, graph parity, and explicit singleton training guidance were run against the previous draft: **5 targeted failures** (2.26s), as expected. The nonlinear implementation then passed 33 tests (4.09s).
5. Refinement tests were extended across transport/action/posterior. Final fresh command `.venv/Scripts/python -m pytest tests/test_cc_v4_model.py -q`: **39 passed in 4.32s**.
6. Final fresh command `.venv/Scripts/python -m ruff check scripts/cc_v4_model.py tests/test_cc_v4_model.py`: **All checks passed**.

The final CPU suite checks independent NumPy cost agreement, independent analytic action gradients, finite-difference first/second derivatives, joint log-action translations, exact-point null behavior, physical mean/RMS statistics, nonlinear simplex correction and derivatives, weight normalization and action dependence/null behavior, standalone Sobolev gradients to all evidence components, invalid inputs and finite fallbacks, exposure scales 1e-200/1e200 in double, input nonmutation, repeated cached queries, deterministic 1/2/4-stage refinement for all three learned modes, bounded action validation, FP32 under autocast/half input, double computation, direct-point behavior, graph parity, and the 1–5M parameter constraint.

## Remaining limits and integration notes

- Public-benchmark training, measured accuracy, runtime/VRAM accounting, checkpoint selection, licensing ledger, confidence calibration, and stronger empirical novelty controls belong to the parent experiment work; none is claimed here.
- Use eval mode for singleton32x32 deployment. Standard BatchNorm remains unchanged; no fallback silently changes training statistics.
- Report predicted risk and measured GT reproduction separately. Repeated refinement may worsen true error despite nonincreasing predicted risk.
- Action-dependent routing is a learned evidence mechanism; its same-sized action null is necessary to test whether physical color transport adds empirical value beyond generic action conditioning.
- The proposal prior, color floor, angular norm smoothing, inactive posterior/direct columns, and all later-stage candidates must remain disclosed in experiment accounting.
