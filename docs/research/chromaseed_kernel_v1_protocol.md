# ChromaSeed-K v1: compact kernel models and bounded approximation

2026-09-13, prospective, before this series is fitted. The previous goal turn is classified **progress**: ChromaSeed-R delivered450 fits,90 storage evaluations and evidence of transfer/early-exit failures. The broader compact/fast/high-quality model goal remains active. This series tests stronger compact predictors rather than treating that earlier series as goal completion.

## Data, endpoints and interpretation

Only original TRAIN cache SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`:966 rows,24 people. Load color36, instrument native Lab, patient/site/device metadata, no images or legacy validation/calibration/test. Reuse frozen historical mixed18/6, SLR→iPod8/16 and reverse16/8 person roles and their inner three-person folds. All are exposed, overlapping exploratory protocols. Camera-group transfers also change people/color distributions.

Primary: person-balanced native DeltaE00. Secondary: image/site-person means,p90,seed variability, numerical payload bytes, full bank/search and standalone fit times, batch-one CPU inference including normalization but excluding image/face feature extraction. A few held people and many prior trials cannot establish fresh smartphone-face accuracy or a novel invention.

## Registered models and search

All models use color36 and normalization fitted only on current fit rows. Store normalization/centers/coefficients FP32; compute deployed kernel evaluation in FP64, stated separately from file size. Compute fit coordinates using the stored FP32 normalization, then cast them to FP64; round kernel width to FP32 before the kernel algebra. This gives one consistent geometry for fitting, deployment and the approximation diagnostic. Target normalization uses stored fit-only parameters. Balance people→sites→images as in the frozen helper.

Gaussian kernel `exp(-mean((x-c)^2)/(2 width^2))`. Base width is the lower median of positive pairwise fit distances (matching torch median convention), scaled by0.5/1/2. Ridge alpha0.1/1/10, no additional bias after target centering. Candidate ranks16/32/64/128. Spectral floor is1e-8 times the maximum eigenvalue when inverting a landmark Gram matrix; omitted directions and effective rank are recorded. No free eigenfloor tuning.

- `exact`: full weighted kernel ridge, all9 width/alpha combinations; deterministic seed17. Width-factor1 reproduces the earlier KRR reference to numerical tolerance; the wider search is a distinct control for any hyperparameter benefit.
- `nys_random`: Nyström ridge using weighted random landmarks without replacement, seeds17/29/43.
- `nys_pivot`: Nyström ridge using greedy pivoted Cholesky of sqrt(W)Ksqrt(W), deterministic first argmax, seed17 only.
- `nys_rpchol`: Nyström ridge using residual-diagonal randomized Cholesky pivots of the same weighted matrix, seeds17/29/43.
- `project_rpchol`: project the full kernel teacher into the nested landmark spans from randomized Cholesky, same three seeds. This is a reduced-set teacher projection, not a backprop-trained neural student. Teachers use the current fit labels and never held labels.

Nyström uses the RKHS regularizer: whiten the landmark kernel, solve ridge in whitened features and collapse coefficients for deployment. It differs from the older RBF readout's Euclidean coefficient penalty and linear skip. Landmark prefixes are shared across all three alphas and four sizes for a given mode/seed/width. Kernel, spectral decomposition and design algebra are reused/batched where possible; receipt counts distinguish fitted readouts from independent dataset preparations.

For each role, compact family and nominal rank, choose width/alpha by concatenated inner OOF person-mean error averaged over available seeds. Ties prefer smaller alpha, then width factor. Exact chooses over9 combinations. All selections are locked before final evaluation. Computing the entire final algebra bank on fit rows is allowed for reuse and the adaptive-prefix variant, but only the registered selected variants are evaluated on outer rows. Expected inner readouts:3321; final bank readouts:1107; primary selected outer models:123. Store realized counts rather than labeling algebra outputs as independent subjects or neural training runs.

## Repeated refinement with an approximation bound

For `project_rpchol`, additionally construct an adaptive prefix predictor with ranks16/32/64/128 using the width/alpha selected for rank128. The same centers are nested, so inference evaluates only the additional kernels at each pass. Each level has its own projected coefficient vector. Keep original selected settings; no choice from outer curves.

After FP32 rounding of teacher and projected coefficients, compute the exact-arithmetic RKHS residual `r=f_teacher-f_projection`: its squared norm q for each normalized Lab channel and residual values at each landmark are obtained from the fit kernel. Store these diagnostic values in FP64. They are properties of two fitted functions, not predictions of true skin-color error.

For a query x choose the landmark c with largest kernel correlation k. Because k(x,x)=k(c,c)=1, decomposition along the single kernel section at c gives the component bound

`|r(x)| <= |k*r(c)| + sqrt(max(q-r(c)^2,0))*sqrt(max(1-k^2,0))`.

Multiply each channel bound by its stored target scale and combine in Euclidean native Lab norm. This bound remains applicable to a truncated/rounded projection because it does not assume the residual is exactly orthogonal to every landmark. Use a small numerical allowance and explicitly verify it against actual teacher differences; no formal floating-point certification is claimed. Do not describe it as a bound on instrument error.

Choose an exit tolerance from0(force128)/0.1/0.25/0.5/1.0 native Lab, using inner OOF only: smallest mean number of centers with person error ≤ forced128+0.02 and imagep90 ≤ forced128+0.1; ties prefer smaller tolerance. Store this policy scalar in FP64 so the chosen decimal0.1 is not changed by float32 serialization. At rank128 always return the prediction, but report if its approximation bound remains above tolerance. Thus meeting the tolerance is not promised for every input. Record per-level teacher drift, bound violations, used ranks, tolerance-met coverage, actual CPU runtime and additional payload bytes. Early exit here approximates an explicitly fitted stronger reference; that reference can still be wrong.

## Limited correction control

Separately blend `nys_rpchol` rank64 and `project_rpchol` rank128 with the earlier selected guided-RBF models for the exact same fit/OOF rows. The previous source/artifact hashes must verify. Each seed is paired with the matching guided seed; predictions are combined per model, not ensembled across seeds. Choose guided weight rho from0/0.25/0.5/0.75/1 using current inner OOF person error (ties prefer smaller rho). This convex correction includes zero correction and cannot overshoot the two predictions. No residual-error head or additional fit on held labels. Report both constituent storage/latency and the combined cost. This is a bounded constant-blend control, not learned dynamic routing.

## Verification and resources

Before real fits: test Nyström full-landmark equivalence to independent exact weighted KRR, Cholesky residual reconstruction/prefix uniqueness, reduced-set projection geometry, the pointwise residual bound including rounded/truncated coefficients, actual adaptive-rank execution and ordinary blend endpoints. Compare CPU/GPU synthetic algebra throughput before freezing an execution choice; GPU is authorized but a slower GPU path is not used merely for utilization. No external implementation or pretrained weights are imported.

Execution choice from the synthetic734-row coefficient-bank preflight (one warmup, three repetitions, three alphas): full exact solve median46.80ms CPU vs37.78ms CUDA; rank128 Nyström median2.40ms CPU vs11.62ms CUDA, with coefficients agreeing at1e-8 relative/1e-9 absolute tolerance. Use a hybrid: full teacher spectral decomposition on CUDA, compact landmark/design algebra on CPU. Include data preparation, transferred results, reusable algebra and output persistence separately in measured receipts. The preflight covers core algebra, not a claimed end-to-end training speedup. Direct blocked pairwise differences exclude identical samples when calculating the base width.

Freeze the new core/runner/protocol and reused helper sources. Independent audit re-derives normalization, kernel predictions, row roles, hyperparameter/exit/blend choices and all final metrics, including teacher-bound checks. Preserve every negative result and the earlier source locks.

Primary background: [Rudi et al., Less is More: Nyström Computational Regularization](https://arxiv.org/abs/1507.04717), [Chen et al., Randomly pivoted Cholesky](https://arxiv.org/abs/2207.06503). These are established techniques. The bound above is a direct RKHS Cauchy–Schwarz decomposition for this diagnostic, not a novelty claim. Dataset and derived-weight commercial rights remain separate from code rights.
