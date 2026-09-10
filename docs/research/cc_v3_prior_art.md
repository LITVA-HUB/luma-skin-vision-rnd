# CC v3: bounded prior art and the original FFCC reference

Evidence cutoff: **2026-09-10**. **LITERATURE/SOURCE INSPECTION; NO REPRODUCED RESULTS.** This note complements `cc_v2_prior_art.md` and `cc_v3_spec.md`. No images, labels, pretrained weights or third-party ports were acquired for this review. This is a bounded primary-source review, not a patent opinion or proof that an architectural combination is absent from the literature.

## What is already established

The proposed architecture is materially different from the v2 residual CNN as an experiment: full image-selected color coordinates, spatial graph processing and a directional mixture with transported risk. Its ingredients do not support a claim of fundamentally new equivariance mathematics.

| Primary source | Closest overlap and limit of inspection |
|---|---|
| [Lin, Helwig, Gui and Ji, ICML 2024, Minimal Frame Averaging](https://proceedings.mlr.press/v235/lin24i.html); [author implementation](https://github.com/divelab/MFA) | General-purpose frame construction and arbitrary-backbone equivariance. The author repository explicitly lists GL(d,R), SL(d,R) and Aff(d,R), with a full-column-rank restriction. Thus extending an arbitrary graph network using a full linear frame is already within published architectural territory. This review does not establish that their particular frame selector equals the proposed max-determinant patch selector. No MFA code was adopted. |
| [Kaba et al., ICML 2023, Equivariance with Learned Canonicalization Functions](https://proceedings.mlr.press/v202/kaba23a.html) | Canonicalize, use an unconstrained network, transform the output back. Learning or hand-designing the canonicalizer is a design choice in an established pattern. |
| [Dym, Lawrence and Siegel, ICML 2024, Equivariant Frames and the Impossibility of Continuous Canonicalization](https://proceedings.mlr.press/v235/dym24a.html) | Exact symmetry does not guarantee continuity; discrete canonicalization/frame choices can introduce discontinuity, while weighted frames address continuity for studied group actions. Do not extrapolate the paper's impossibility theorems to this precise restricted GL input domain without checking their hypotheses. Its robustness concern applies directly to switching maximizing triples. |
| [Shumaylov et al., ICLR 2025, Lie Algebra Canonicalization](https://proceedings.iclr.cc/paper_files/paper/2025/file/056521a35eacd9d2127b66a7d3c499c5-Paper-Conference.pdf) | Contemporary canonicalization for noncompact groups using infinitesimal generators and unconstrained models; a different computational construction and PDE application. Noncompact-group canonicalization itself is not new. |
| [Brill, 1985, Reflectances giving volumetric color constancy in daylight](https://opg.optica.org/abstract.cfm?uri=oam-1985-TUJ7); [Brill, 1985, Decomposition of Cohen's Matrix R into Simpler Color Invariants](https://www.jstor.org/stable/1422514) | The Optica primary abstract explicitly identifies tristimulus volume ratios as a basis of computation. The JSTOR primary record establishes the related article's identity, but its full text was inaccessible. This is a historical lead, not a claim to have reproduced its complete algorithm. |
| [Healey and Slater, JOSA A 1994, Global color constancy](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-11-11-3003) | Primary abstract describes illumination-invariant color-distribution descriptors under a three-dimensional linear reflectance model. Color-distribution invariants long predate neural frames. Full equations were not inspected. |
| [Barron and Tsai, FFCC, CVPR 2017](https://arxiv.org/pdf/1611.07596) | Compact learned illuminant posterior, directional statistics on a torus, uncertainty and downstream temporal use. FFCC is not an S² mixture in an image-dependent full frame, but posterior color constancy and useful uncertainty outputs are already established. Source details below are derived from the Apache-2.0 implementation. |
| [Buzzelli and Bianco, Pattern Recognition 2025, Uncertainty estimation in color constancy](https://www.sciencedirect.com/science/article/pii/S0031320324009269) | Primary abstract and section snippets describe uncertainty estimated with at most one inference, and uncertainty-guided cascading. They do not establish calibration under this project's held-out-camera contract. Full equations were not available; no estimator is reconstructed from the abstract. |
| [Wei et al., Integral FFCC, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wei_Integral_Fast_Fourier_Color_Constancy_CVPR_2025_paper.html) | Integral histograms and parallel Fourier convolution extend efficient FFCC to spatially varying illumination. Spatial/color processing and compact FFCC remain active prior art. This does not replace the global single-illuminant control. |

The 2025–2026 color-constancy sources already recorded in `cc_v2_prior_art.md` (CCMNet, VLM-CC, CSNet, BRE) remain relevant to cross-camera assumptions, residual correction and normalized scene cues. This bounded follow-up establishes no additional 2026 source as closer than the explicit GL frame work. That is a search limitation, not evidence of novelty.

## Mathematical audit of the proposed combination

These are elementary derivations for this experiment, not claims of new theorems. With ordered patch vectors `b1,b2,b3`, `B=[b1 b2 b3]`, Cramer's rule gives

```text
(B^-1 x)_j = det(B with column j replaced by x) / det(B).
```

The canonical channels are therefore determinant/volume ratios. For any invertible `T`, every candidate determinant scales by the same `det(T)`, so the exact set of absolute-volume maximizers and its lexicographic first member are preserved. Hence `B(TX)=T B(X)` and `B(TX)^-1 TX=B(X)^-1 X`. Common scalar image normalization changes the frame equality by that scalar but leaves canonical inputs and final projective transport unchanged. Fixed spatial averaging commutes with `T`; RGB-dependent masks do not generally do so.

A graph on these invariant features is an ordinary graph network after canonicalization. Its edge gates and diffusion do not constitute an additional GL-equivariance theorem. Removing nine global linear degrees of freedom can discard predictive absolute color priors. Real camera spectral changes need not equal one global 3×3 transform across all materials. Source accuracy and outer-camera performance remain independent empirical questions.

Critical restrictions:

- Max determinant does not minimize condition number. Nearly coplanar patch means can still produce a badly conditioned maximizing frame, and full-rank pixels can have rank-deficient spatial patch means.
- A unique maximizing triple gives local stability only while its margin stays positive. A tie-breaking rule makes exact selection deterministic but does not make it continuous across a switch.
- Condition number and smallest-singular-value acceptance thresholds are not GL-invariant. Report the algebraic identity on **jointly accepted** inputs and separately report changed validity. Rank is invariant in exact arithmetic; numerical rank tests are not.
- GL(3) does not preserve the nonnegative RGB domain. Positivity and invalid-output rules restrict the operational identity to transformations with admissible input/output, not all of GL acting on every RGB image. A pseudoinverse, clipping, diagonal jitter or a GW fallback must not silently inherit the full-frame proof.
- The canonical target `q=normalize(B^-1 g)` can contain negative components. Constraining `q` to the positive octant would exclude valid positive camera-space illuminants.

For `q` on S² and camera direction `e=normalize(Bq)`, the area Jacobian is

```text
J_B(q) = abs(det B) / ||Bq||^3,
p_camera(e) = p_canonical(q) / J_B(q),
NLL_camera = NLL_canonical + log J_B(q).
```

This follows by transporting two tangent vectors and taking their projected area. It is invariant to common positive scaling of `B`. Raw canonical NLLs from direct, diagonal and full-frame modes refer to different image-dependent measures and must not be compared as if they were the same camera-space density. For fixed nonlearned `B`, the Jacobian is constant with respect to the network parameters, so it need not change parameter gradients, but it matters for likelihood reporting and comparisons. These statements concern the full sphere; restricting to positive mapped RGB changes normalization and requires reporting excluded probability mass.

Reproduction error uses `r(g,e)=angle(g/e, [1,1,1])` (componentwise division). A practical posterior proxy transports every quadrature hypothesis to camera RGB before applying `r(hypothesis,point)`. Using only mixture means discards concentrations and is not posterior integration. Finite tangent quadrature is an approximation; it must retain invalid-mass information and does not furnish a coverage guarantee. Joint diagonal gains cancel in `g/e`; general channel mixing does not. Therefore identical canonical posteriors can induce different reproduction risks after transport. Calibrate any error predictor on source-held-out residuals and assess actual risk/coverage on locked evaluation data.

## Pinned original FFCC source and license

Reference: [google/ffcc at commit 2fa9e1316954dbd3913630b7d597927941b4dd32](https://github.com/google/ffcc/tree/2fa9e1316954dbd3913630b7d597927941b4dd32), inspected 2026-09-10. Repository owner is Google; repository is archived. Local unmodified reference is `cc_v3_sources/google_ffcc/`: **85 source/license/reference files, 241,767 upstream bytes**, plus our manifest/provenance. `MANIFEST.json` records upstream Git blob IDs, byte sizes and SHA-256. No `data/`, image, label, model/checkpoint, project fitted hyperparameter file or minFunc dependency was downloaded. The reference is documentation, not executable project integration.

The [pinned LICENSE](https://github.com/google/ffcc/blob/2fa9e1316954dbd3913630b7d597927941b4dd32/LICENSE) is Apache-2.0; retained MATLAB files carry Google's Apache headers. Preserve license/attribution and mark changed/adapted files. Paper availability and code licensing do not clear datasets, trained weights or the separate MATLAB/minFunc runtime. No scientific-paper PDF is redistributed here. **Code license: recorded Apache-2.0. Weights: none adopted. Data: none adopted.** This is provenance recording, not a legal clearance opinion.

## Exact source-grounded compact-control contract

The formulas in this section are a transcription/derivation of permissively licensed source behavior. Pinned files in the local reference are the authority where paper prose and code disagree. Proposed Python interfaces are specified for independent implementation, not claims that an implementation already exists.

### Input and two histograms

`features(rgb, mask, params) -> hist[2,n,n], average_rgb[3], diagnostics`

- Input: black-level-corrected linear RGB. Saturation/chart removal and resizing belong to explicit preprocessing. A cached project tensor/thumbnail is not automatically the original paper's input pipeline.
- `ChannelizeImage.m` uses two RGB feature images: masked RGB and `MaskedLocalAbsoluteDeviation.m`. For each center and channel, the latter averages `abs(neighbor-center)` over the eight offsets in its 3×3 neighborhood, weighted by `mask(center)*mask(neighbor)`. `Pad1.m` replicates image and mask border values. It is neither Sobel magnitude, standard deviation, nor absolute difference from the neighborhood mean. For no valid neighbors the double path yields NaN, later excluded. Integer paths have extra casts/rounding and need separate parity if adopted.
- `FeaturizeImage.m`: `u=log(G)-log(R)`, `v=log(G)-log(B)`. Keep pixels with finite log ratios, true input mask and every feature channel at least the minimum. `PrivateConstants.m` minimum is `1/256` of full scale; integer input scales by its type's maximum. The source's literal `isa(im,'float')` branch is a runtime/type ambiguity to verify in MATLAB; a Python float policy should explicitly use `[0,1]` full scale and record this compatibility decision rather than copying an ambiguous test.
- `Psplat2.m` makes **nearest-bin**, periodic, **unit-count** histograms. No norm/intensity weighting and no bilinear input splatting. In zero-based Python notation:

```text
i = round_matlab((u - lo)/h) mod n
j = round_matlab((v - lo)/h) mod n
hist[c,i,j] += 1
hist[c] /= max(machine_epsilon, sum(hist[c]))
```

`round_matlab` rounds halves away from zero; NumPy/PyTorch round-to-even differs. u is the row axis, v the column axis. Each channel normalizes independently. No valid samples gives the zero histogram. `GehlerShiConstants.m`: `n=64`, `h=1/32`, scalar `lo=-0.4375` for both axes, period `L=n*h=2`. A different source-selected origin must be logged; target labels must not select it. The paper's Eq. (6) interval indicator is not a substitute for the implementation's rounding rule.

### Circular filtering and prior

`score(hist, filters[2,n,n], bias[n,n]) -> logits[n,n], pmf[n,n]`

```text
score = real(ifft2(sum_c fft2(hist[c]) * fft2(filter[c]))) + bias
P = exp(score - max(score)) / sum(exp(score - max(score)))
```

Forward uses **no conjugation**, no fftshift, and no kernel centering. It is circular convolution, `score[i,j]=sum_c,a,b hist[c,a,b]*filter[c,(i-a) mod n,(j-b) mod n]+bias[i,j]`. A delta input/filter fixture must establish orientation; a generic conv2d correlation is different. Use unnormalized forward FFT and inverse divided by `n²`, as MATLAB does.

`EvaluateModel.m` at this commit has `score=FX+B` and **no learned gain map**. The FFCC paper's full prior formula additionally includes `exp(G)*FX+B`; implementers must distinguish the paper configuration from the pinned shallow code. This source's `LEARN_BIAS=true` learns an absolute chroma bias, equivalent to an unnormalized multiplicative prior after exponentiation. A zero-bias variant is an ablation. A spatial-domain filter parameterization yields `2*n²+n²=12,288` real parameters with bias, but is a changed optimization parameterization relative to original packed/preconditioned Fourier training.

### Decode, covariance and de-aliasing

`decode(P, lo, h, eps_bins, average_rgb, unwrap_mode) -> mu_uv[2], covariance_uv[2,2], pred_rgb[3], confidence, diagnostics`

`FitBivariateVonMises.m` uses circular moments, not argmax and not the ordinary planar mean. For zero-based `k=0..n-1`, `theta_k=2*pi*k/n`, form marginals `p_u=sum_v P` and `p_v=sum_u P`. For each marginal:

```text
s = sum_k p[k]*sin(theta[k]); c = sum_k p[k]*cos(theta[k])
mu_index = (atan2(s,c) mod 2*pi) * n/(2*pi)
```

The MATLAB output adds one for indexing; omit that one only when every downstream formula is zero-based. A uniform/balanced multimodal PMF has zero/near-zero resultant and an undefined/unstable circular mean. Do not silently describe an arbitrary angle as confident. This is especially relevant for zero-initialized filters at the start of nonconvex training.

Covariance is a second-moment calculation around an integer-shifted torus chart, **not** `-log(resultant)` and not simply the expected squared difference about the circular mean. Exactly reproduce the source with its one-based mean `m=mu_index+1`:

```text
k = 1..n
t_u[k] = mod(k - round_matlab(m_u) + n/2 - 1, n) + 1
t_v[k] = mod(k - round_matlab(m_v) + n/2 - 1, n) + 1
E_u = sum p_u*t_u; E_v = sum p_v*t_v
V_uu = sum p_u*t_u^2 - E_u^2
V_vv = sum p_v*t_v^2 - E_v^2
V_uv = sum_ij P[i,j]*t_u[i]*t_v[j] - E_u*E_v
```

Default: no forced isotropy, and `V <- V + eps_bins*I` before converting units. `DefaultHyperparams.m` has `eps_bins=1`; it is a starting default, not an optimal benchmark setting. `mu_uv=lo+h*mu_index`, `covariance_uv=h²*V`. The optional clamp path requires forced isotropy and differs from padding.

Default gray-light decoding leaves `mu_uv` inside `[lo,lo+L)`; that absolute range is a prior. Optional source gray-world decoding is

```text
a_uv = (log(avg_G/avg_R), log(avg_G/avg_B))
mu_uv <- mu_uv - L*round_matlab((mu_uv-a_uv)/L)
pred_rgb = normalize([exp(-mu_u), 1, exp(-mu_v)])
```

`PrecomputeTrainingData.m` averages **linear RGB** over all spatial pixels and then normalizes it; this is not the mean of pixel log chromas described in the paper's gray-world equation. Zero-filled invalid pixels participate in that average, although common zero padding cancels in RGB ratios. Changing that policy is a preprocessing deviation. Confidence in `EvaluateModel.m` is `exp(0.5*logdet(eps_bins*h²*I)-0.5*logdet(covariance_uv))`; it measures concentration and does not certify actual reproduction risk.

### Loss, regularization and optimizer

`loss(P, mu_uv, covariance_uv, gt_rgb, params) -> crossentropy, gaussian_nll_shifted, regularizer, total`

Target `y=(log(gt_G/gt_R),log(gt_G/gt_B))`. Cross entropy is `-sum T*log P`. **Source naming trap:** `SMOOTH_CROSS_ENTROPY=true` enters the **one-hot nearest-bin** branch in `EvaluateModel.m`, despite the opposite comment in `PrivateConstants.m`. False calls `UvToP.m` and bilinearly distributes target mass to four wrapped neighbors of `(y-lo)/h`. Record which behavior the Python control actually uses. Bilinear targets do not imply bilinear input histograms.

The source's `VON_MISES_LOSS='likelihood'` is a Gaussian likelihood in decoded UV space, using the toroidal moment fit, not the exact normalized BVM density:

```text
d = y - mu_uv
loss_nll = 0.5*d.T*inverse(Sigma)*d + 0.5*logdet(Sigma) + log(2*pi)
          - log(2*pi*eps_bins*h²)
```

Default `pad` uses the padded Sigma. No shortest-wrapped label residual is introduced here; de-aliasing occurred in the decoder. Stable Cholesky/solve is mathematically equivalent away from singularity; adding an extra numerical floor changes the formulation and must be reported. The source has optional squared-UV and expected-squared-UV losses. Their existence does not make an RGB angular/reproduction training loss a faithful default FFCC loss. Do not blindly port the optional expected-error derivative: source forward uses `32*||d||²+32*trace(Sigma)` but lists `d_mu=2*d`, missing the corresponding factor 32.

Regularization in `TrainModel.m` and `TrainModelLossfun.m` is a **quadratic** smoothness penalty in full FFT coordinates, despite the source using the term total variation. Define

```text
A[u,v] = |FFT_n([-1,1]^T/sqrt(8))|² + |FFT_n([-1,1]/sqrt(8))|²
R_filter[c] = lambda_filter[c]*A + shift_filter[c]
R_bias = lambda_bias*A + shift_bias
Reg = 0.5*data_mass*sum_Q,u,v R_Q[u,v]*|FFT(Q)[u,v]|²
```

Here `Q` covers filters and enabled bias. With the source's sum of sample-weighted data losses, regularization is multiplied by the sum of sample weights `data_mass`. An averaged loss should divide the entire objective by that mass. Do not confuse L1 TV, AdamW weight decay or squared spatial differences without the Parseval factor with this objective. Defaults are filter lambda `n^-4` for each of two channels, filter shifts `2^-8`, bias lambda `1`, bias shift `2^-8`; these are documented initial values, not transplanted tuned hyperparameters.

The original optimizes packed real Hermitian Fourier coefficients through a regularizer-derived reparameterization. `Fft2RegularizerToPreconditioner.m` and `Fft2ToVec*.m` preserve the conjugate-pair multiplicities and special DC/Nyquist bins. In preconditioned coordinates its regularizer is `0.5*data_mass*||z||²`. A straightforward real spatial filter with full FFT regularizer can preserve the objective but changes conditioning and optimization trajectory.

Default shallow training starts at zero and uses two full-batch minFunc L-BFGS passes: 16 iterations of cross entropy, then 64 of likelihood (not both losses for every iteration). General annealing linearly interpolates loss weights. Source options: `Corr=num_iters`, `MaxFunEvals=4+2*num_iters`, `optTol=0`, `progTol=0`. PyTorch L-BFGS has different line search/stopping/evaluation rules; Adam and minibatches are larger deviations. Replacing minFunc avoids acquiring that dependency but does not reproduce its optimizer. No pretrained weights or tuned project hyperparameter files were adopted.

### Minimum independent port evidence

Before calling a control faithful at the formula level, use handcrafted, data-free fixtures for axis/sign conventions, negative and half-bin rounding, periodic wrap, masked replicated edges, unequal channel hist masses, delta convolution, stable softmax, wrapped concentrated posterior, covariance padding/units, both target-label branches, de-aliasing, likelihood and regularization scaling. Gradient checks must cover the trained objective away from chart switches and zero circular resultants. Compare against MATLAB outputs if MATLAB is available; otherwise report that numerical MATLAB parity remains unverified. Source reading alone is not numerical parity.

The compact control should expose histograms, PMF, decoded UV mean/covariance, illuminant and concentration score so it can be audited independently of training. Suggested result metadata: source commit, histogram coordinate/rounding/edge policy, channels, origin/bin settings, label branch, bias/gain flags, de-alias policy, covariance mode/epsilon, FFT normalization, parameterization, regularizer, loss schedule, optimizer, input resize/mask/quantization, fitting splits and all deviations.

An independently written Python control with changed optimizer/input pipeline should be labeled **FFCC-inspired, source-grounded compact control** until suitable parity and benchmark reproduction exist. It must not claim original FFCC benchmark numbers, original mobile runtime, original full-paper gain-prior configuration, original optimizer trajectory, exact continuous diagonal equivariance, calibrated posterior risk, matched patent novelty, or facial Lab/DeltaE accuracy. A weak/undertrained FFCC-inspired control cannot justify a claim to beat established FFCC.
