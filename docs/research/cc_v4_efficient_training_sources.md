# V4: efficient training, compact models and bounded iterative correction

Research-only note, checked 2026-09-10. Scope follows [the V4 research plan](cc_v4_research_plan.md): 1–5M parameters preferred, below 10M desirable, RTX 4060 8 GB training, one local image at inference. This note studies transferable mechanisms, not a selected final architecture. No code, datasets, pretrained weights or teacher outputs were downloaded; no models were trained. Published measurements below are author-reported, not measurements in this repository. Proposed experiments are **PLANNED hypotheses**.

The most defensible routes are (1) a few affordable, complementary training experts distilled into one compact student and (2) a shared correction block with two to four supervised iterations. Their ingredients are established prior art. Whether their combination helps this photometric problem requires matched experiments. Foundation-model accuracy, puzzle reasoning scores and attractive mobile latency do not establish photometric accuracy or universal recovery of original surface color.

## Primary-source evidence and actual scale

| Source and date | Verified mechanism and scale | Relevance and limits |
|---|---|---|
| Apple, [MobileCLIP2](https://arxiv.org/html/2508.20691v1), 2025-08-28, TMLR 2025; [official repository](https://github.com/apple/ml-mobileclip) | Offline reinforcement stores teacher information for reproducible augmented views. Two DFN ViT-L/14 teachers, a CoCa captioner and teacher-specific distillation temperatures improve training. S0 has **11.4M image + 63.4M text parameters**, 256² input; table latency 1.5 + 3.3 ms, with the paper's mobile comparison on iPhone 12 Pro Max. S0 sees 13B samples from DFNDR-2B. | Borrow cached complementary supervision and temperature selection. Even the image encoder exceeds our preferred size; the pretraining recipe is far beyond one 4060. Image-text invariance and synthetic captions are not physical color labels. |
| Google, [MobileNetV4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05647.pdf), preprint 2024-04-16, ECCV 2024 | Universal Inverted Bottleneck and hardware-aware operator choices. Conv-S: **3.8M parameters, 0.2G MACs**, 73.8% ImageNet top-1; 2.4 ms Pixel 6 CPU, 0.7 ms Pixel 8 EdgeTPU, 0.6 ms iPhone 13 CoreML. The **87% headline belongs to Hybrid-L**, a 35.9M/7.2G architecture with distillation and JFT pretraining, not Conv-S. | A genuine compact architectural control. Operator layout, memory movement and target runtime matter alongside MACs. No inference time in this table predicts our Windows/4060 batch-one latency. |
| Apple, [FastVLM / FastViTHD](https://arxiv.org/html/2412.13303v2), first 2024-12-17, CVPR 2025 | Later attention at lower resolution reduces vision tokens and LLM prefill. FastViTHD has **125.1M parameters**; table 3 reports 6.8 ms encoder latency at 224² on an M1 MacBook Pro. Other reported TTFT comparisons include an LLM and higher resolutions. | Useful principle: expensive global mixing belongs after spatial reduction. This backbone is not a 1–5M deployment option, and TTFT is not an illuminant-estimation benchmark. |
| Tsinghua MIG, [RepViT](https://github.com/THU-MIG/RepViT), preprint 2023-07, CVPR 2024 | A CNN informed by efficient transformer design with a separate fused deployment structure. M0.9 has **5.1M parameters, 0.8G MACs**; 78.7/79.1% ImageNet after 300/450 epochs. Reported 0.9 ms uses iPhone 12, iOS 16, Xcode 14. | A near-budget compact control; classification gains alone do not justify its color representation. The release predates 2024 despite its CVPR 2024 venue. |
| Samsung SAIL Montréal, [Tiny Recursive Models](https://arxiv.org/html/2510.04871v1), 2025-10-06 | Shared two-layer updates maintain an answer and latent state, with deep supervision. Attention variant is **7M**, Sudoku MLP variant **5M**. The default recurrence is n=6, T=3 and at most 16 supervision stages; partial trajectories are detached, then a complete small recursion is differentiated. | Relevant to sharing a small correction operator. Evidence concerns discrete puzzles, not continuous photometry. Neither fixed-point convergence nor monotonic accuracy follows from recurrence. Copying its whole optimizer/unroll recipe would be unjustified. |
| NVIDIA/MIT, [SANA-Sprint](https://research.nvidia.com/labs/eai/publication/sana-sprint/), March 2025, ICCV 2025 | Continuous-time consistency distillation plus latent adversarial distillation compresses generation to 1–4 steps. The smallest released diffusion transformer is **0.6B**; authors report 1024² generation in 0.1 s on H100 and 0.31 s on RTX 4090. | Supports studying many-step teacher to few-step student compression. It is not a small photometric model. Perceptual generative realism can change the very colors we need to measure. |
| NVIDIA, [Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf), retrieved report dated **2026-03-05** | Full-precision teacher supervises a quantized student using KL; evaluated models are LLMs/VLMs, including 8B–30B families. This extends a known quantization/distillation idea. | A current training principle, not evidence for photometry or RTX 4060 NVFP4 speed. Consider output-preserving quantization only after a good floating-point student exists. The hardware/precision path requires its own supported-runtime measurement. |

The MobileNetV4 supplement makes two especially useful details concrete. Its cached target is computed **after** augmentation and JPEG encode/decode, so the teacher sees the stored view; student-time augmentation is then restricted. Its teacher is EfficientNet-L2, **480M parameters and 290G MACs**. Mobile benchmarking uses INT8 for CPU/Hexagon/EdgeTPU, FP16 for mobile GPU, and CoreML FP16 for Apple Neural Engine. A reported 2.3-hour Conv-L distillation run uses **128 TPU v5e**, not a desktop GPU. Borrow view matching and operator measurement, not the resource budget or JPEG preprocessing for linear scientific RGB. [Supplement, sections B and H](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05647-supp.pdf).

TRM is also not a cheap-training result merely because it has few parameters. Its official smallest Sudoku recipe takes approximately **18 hours on one 48 GB L40S**, with extensive puzzle augmentation; ARC-AGI uses **four H100s for about three days**. These are different data, training versions and hardware from our problem. The repository explicitly documents those recipes; they are not estimates of what a 4060 can reproduce. [Official training instructions](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

## Code, model and data terms are separate

This is an evidence ledger, not a legal clearance opinion. Unknown asset terms mean **COMMERCIAL USE NOT CLEARED** under this repository's policy. Studying an algorithm does not require adopting its downloadable assets.

| Family | Code terms checked | Weight / teacher / data status | Consequence here |
|---|---|---|---|
| MobileCLIP / MobileCLIP2 | [MIT code license](https://raw.githubusercontent.com/apple/ml-mobileclip/main/LICENSE) explicitly separates models and data. | [Apple model terms](https://raw.githubusercontent.com/apple/ml-mobileclip/main/LICENSE_MODELS) restrict models and derivatives to noncommercial research and exclude product development; [data license](https://raw.githubusercontent.com/apple/ml-mobileclip/main/LICENSE_DATA) is CC-BY-NC-ND 4.0. Upstream teachers and original images have additional provenance. | Study the training recipe. Do not treat weights or reinforced teacher caches as cleared for the product track. MIT code does not remove the model restriction. |
| FastVLM | [Custom Apple software license](https://raw.githubusercontent.com/apple/ml-fastvlm/main/LICENSE), not MIT or Apache-2.0; dependencies separate. | [Model terms](https://raw.githubusercontent.com/apple/ml-fastvlm/main/LICENSE_MODEL) carry the same noncommercial research/product-development exclusion. Upstream LLM and image-text data are additional assets. | No weights or caches proposed for adoption. |
| FastViT, historical ICCV 2023 antecedent | [Custom Apple software license](https://raw.githubusercontent.com/apple/ml-fastvit/main/LICENSE), with separate subcomponents. | No distinct checkpoint/data clearance established in this audit; original ImageNet training is not blanket data clearance. | Structural reparameterization is useful prior art; do not infer asset permission from the paper. [Official architecture reference](https://github.com/apple/ml-fastvit). |
| MobileNetV4 | [TensorFlow Models code license](https://github.com/tensorflow/models/blob/master/LICENSE) is Apache-2.0. | This review does not establish terms for a particular pretrained checkpoint or the EfficientNet teacher. ImageNet and private JFT are separate data dependencies. | An independently trained compact control is feasible; no claim that the full published pretraining corpus is available or cleared. |
| RepViT | [Official repository license](https://raw.githubusercontent.com/THU-MIG/RepViT/main/LICENSE) is Apache-2.0. | The linked checkpoint's separate provenance/teacher rights were not established; code licensing is insufficient to certify every weight/data asset. | Architecture study or an independently trained control, pending normal adoption review if code is copied. |
| TRM | [MIT license](https://raw.githubusercontent.com/SamsungSAILMontreal/TinyRecursiveModels/main/LICENSE), Samsung 2025. | Puzzle datasets and trained checkpoints were not cleared separately. | Recurrent training principles only; no puzzle assets needed. |
| SANA-Sprint | Complete code/dependency audit not performed. | [Official 0.6B model card](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px) identifies Apache-2.0 for the diffusion model and **additional Gemma terms/policy** for Gemma-2-2B-IT. A diffusion checkpoint is not the entire pipeline; training data provenance is separate. | No generator or text encoder proposed for deployment or teacher adoption. |
| NVFP4 QAD / early-exit research | No implementation or checkpoint adopted; full asset-license audit not performed. | Unknown per-checkpoint/data terms remain unknown. | Principles only. |

## Nearest overlap and reliability limits

Cached teacher distributions and view replay are established independently of MobileCLIP2. Microsoft's [TinyViT, ECCV 2022](https://arxiv.org/abs/2207.10666) already sparsifies and stores teacher outputs to lower pretraining overhead; its [official training guide](https://github.com/microsoft/Cream/blob/main/TinyViT/docs/TRAINING.md) explicitly stages caching before student pretraining. This is historical prior art, not a new 2024–2026 result. Ensemble-to-student transfer, deep supervision, recurrent parameter sharing and branch fusion therefore cannot themselves be claimed as our invention.

For adaptive compute, [Beyond Greedy Exits](https://arxiv.org/html/2509.23666v1), first 2025-09-28, studies unreliable confidence and uses an online bandit to adjust exit thresholds. Its learned reliability function remains a stated limitation. We should borrow the warning, not claim its classification/VLM risk result automatically transfers to continuous color error. Its cross-image online adaptation also differs from our frozen single-image inference protocol.

[Conformal Risk Control](https://arxiv.org/html/2208.02814v4), ICLR 2024 with a 2022 first preprint, gives expected-risk control for suitable monotone bounded losses under its exchangeability conditions. It does **not** establish arbitrary out-of-distribution guarantees, per-scene certainty or a guarantee for any confidence-sorted conditional error curve. Calibration, model selection and fitting the confidence head need distinct roles and held-out residuals. We must define whether risk means a bounded error expectation, failure probability or mean error among accepted samples; these are different quantities.

The task-specific nearest overlap is also the existing photometric/cascaded-estimation literature, covered by the parallel photometry review. This note does not establish novelty by absence of search hits. A possible contribution must lie in a precisely specified evidence representation, physical update and training/evaluation result, not renaming recurrent inference as reasoning.

## Feasible 4060 experiments: hypotheses, not measured results

### A. Complementary small experts to one student — first training experiment

**Hypothesis:** complementary errors from spatial evidence and global chromatic evidence can smooth supervision for a compact student without carrying an ensemble into inference. A larger semantic teacher is not required to test that hypothesis.

Use the strongest existing direct compact model as the student baseline. Candidate teachers are (i) that spatial model, (ii) the independently implemented FFCC-inspired control and (iii) a separately trained small model with a genuinely different input statistic or crop aggregation. Screen each on source validation; exclude an expert that only introduces bias. A weak FFCC control is not automatically useful merely because its architecture is different.

Train experts sequentially, never concurrently in GPU memory. For learned reliability or gating, generate source training predictions out of fold with scene/subject grouping; fit gates only from those held-out residuals. Freeze teacher and gate selection before the external test. Compare student trained with GT alone, best single teacher, uniform expert mixture, and a residual-trained mixture. Equalize student steps, source images, augmentation views and seeds; report the extra teacher training cost separately.

For compatible distributions on the **same physical output chart**, a concrete candidate is

$$q_T(e\mid I)=\sum_{j=1}^{J}w_j(I)q_j(e\mid I),\qquad w_j\ge0,\quad\sum_jw_j=1,$$

$$L=L_{GT}(e_S,e^*)+\lambda\,\mathrm{KL}(q_T^{(\tau_T)}\Vert q_S^{(\tau_S)}).$$

These equations are a proposed task adaptation of known distillation, not the MobileCLIP2 loss. Keep GT supervision; include lambda=0. Choose temperatures and lambda on source validation, never copy contrastive teacher temperatures into a color posterior. For a point-only teacher, distill its point with a bounded robust distance or explicitly chosen smoothing kernel; do not invent calibrated uncertainty. A toroidal FFCC posterior must be unwrapped and represented on the same physical support before mixture/KL; canonical-direction and camera-direction densities cannot be mixed without a valid map and density correction.

Every cached prediction must correspond to the student's actual view. Store transformation parameters and cache keys, not approximate “equivalent” crops. A lossless cache or deterministic replay preserves linear RGB; MobileNetV4's JPEG cache is an engineering example, not a photometric preprocessing recommendation. For N images, V views, J teachers and D posterior values, float16 storage is 2NVJD bytes: N=1,000, V=8, J=3, D=256 is about **12.3 MB**, excluding metadata/images. This is arithmetic, not a benchmark. Start with a point-target cache if posterior support matching is not yet reliable.

**4060 starting budget:** 128² thumbnails, 1–3M student, batch 16 initially, AMP for ordinary layers and FP32 for sensitive losses; increase only after measured peak VRAM. Train at most three small experts sequentially. A 100-step timing/memory pilot estimates a fixed training budget before scaling. No stated latency or VRAM is guaranteed by parameter count alone.

### B. Shared iterative photometric correction — architecture experiment

**Hypothesis:** after a preliminary correction, previously ambiguous spatial/global evidence becomes easier to interpret; a shared operator can reconsider that evidence with fewer parameters than independent cascaded models. This is useful only if later iterations demonstrably reduce error.

A minimal candidate keeps raw-image features F(I), a current centered log-gain a_t and a small state z_t. With C(v)=v−mean(v), define

$$\tilde I_t=I\odot\exp(-a_t),\quad (z_{t+1},\Delta_t)=f_\theta(F(I),s(\tilde I_t),a_t,z_t),$$

$$a_{t+1}=C(a_t+\eta\tanh\Delta_t),\qquad \hat e_t=\operatorname{normalize}(\exp a_t).$$

Here s is a cheap masked spatial/statistical summary recomputed under the current correction. F is computed once; the conditioning must actually expose changed evidence rather than repeat the same head on an unchanged input. Preserve original chromatic cues alongside normalized features. Start with K=1,2,4 and shared parameters, train all steps with GT, and backpropagate the complete short unroll. Detaching trajectories is a separate ablation, not a theorem-based necessity. Bounded updates restrict jumps but do not prove convergence or correctness.

Match against a single-pass model with equal parameters and another with approximately equal measured compute; compare an unshared cascade as a separate capacity control. Report error at every step, the fraction of images worsened by each update, severe-error tails, and sequence sensitivity. Reject recurrence if extra compute merely repairs deliberately weakened first-stage predictions. Count all head evaluations and statistic recomputation, not just the shared encoder.

The displayed state covers **global diagonal illuminant/gain correction**. It does not identify arbitrary spatial lighting, spectral reflectance or unknown nonlinear processing. Expanding the state to a tone curve, local illumination field or color matrix requires separately defined supervision and identifiability assumptions. Do not relabel this baseline as universal surface-color restoration.

### C. Distill a useful trajectory into one step — only after B works

Freeze the best source-selected K-step model and train the same compact single-pass architecture against GT plus its final prediction/distribution, replaying identical views. Compare against A and a GT-only student with equal training steps. This tests whether improvements came from learnable supervision or intrinsically necessary inference computation.

A consistency-style auxiliary loss can align outputs along an explicitly defined correction trajectory, but must have an endpoint anchored to GT or a validated teacher. The generic equation “outputs should agree” admits a constant predictor. There is no present reason to import a diffusion generator, adversarial discriminator, text encoder or flow-matching objective for a low-dimensional calibrated illuminant target. Those methods become relevant only if a justified conditional distribution over a harder physical state is necessary and evaluable.

### D. Adaptive compute and uncertainty — after a fixed-step accuracy gain

On held-out source residuals, predict both current error and the **benefit of one more step**, b_t=r_t−r_{t+1}. Small update norm or confident softmax is not evidence of small color error. Compare a fixed K with a frozen rule that continues when estimated benefit exceeds a chosen compute cost, subject to a hard maximum K=4. An optional reject decision is separate from an early correct answer.

Calibrate after fixing the accuracy model and exit policy. Plot coverage versus reproduction/angular risk, empirical failure probability at prespecified error thresholds, and conditional risk by known source strata. Report average and p95 latency, maximum evaluations and reject rate. Keep weights and thresholds frozen across images; no target-stream adaptation. Source calibration may fail under new processing/light/surface distributions, so test that failure rather than promise a distribution-free universal confidence score.

### E. Deployment simplification — after A/B/C establish accuracy

Compare an independently trained 1–5M conventional CNN against a fused multi-branch training structure inspired by FastViT/RepViT or a MobileNetV4-style bottleneck. Verify numerical equivalence before/after branch fusion. Parameter/operation count is a starting constraint; measure the actual local runtime at batch one including masks, reductions, histogram/statistic construction and preprocessing.

Then test supported FP16 or INT8 execution, optionally with a frozen floating-point teacher to preserve outputs. Keep sensitive input/color statistics and output decoding at sufficient precision; assess chromatic error on near-neutral, dark and saturated cases. The 2026 NVFP4 work motivates studying accuracy recovery, not assuming 4060 support or importing an LLM post-training stack.

## Transformation and evaluation contract

For a known positive diagonal gain D in unclipped **linear RGB**, I'=DI and e'=normalize(De) under the global diagonal model. This supplies transformed labels, not a new independent real benchmark. Clipping, unknown tone curves, local processing and general spectral changes can invalidate this simple law. Arbitrary hue jitter, grayscale conversion, color-invariant semantic augmentations, or mixing images with different illuminants must not silently retain the same illuminant label. Crops from mixed-illumination scenes also need more than a global-image label assumption.

No distillation recipe adds missing physical ground truth. Evaluate the measurable task actually supervised; angular illuminant accuracy does not establish surface DeltaE, skin measurements or recovery of an unknown original. Wider scene/surface/light/processing robustness requires genuinely held-out data in those dimensions and explicit failure reporting.

| Experiment priority | Expected value, still hypothetical | Main risk | Decisive stop condition |
|---|---|---|---|
| A: small-expert distillation | Highest near-term chance to improve an already competitive compact model at unchanged inference cost | Shared teacher bias; cache/view mismatch; leakage | No consistent gain over GT-only and best-teacher controls under matched student budget |
| B: 2–4 shared corrections | Higher effective computation with few extra weights; task-specific reconsideration of evidence | Drift, overfitting, hidden loss of first-step quality | Later steps fail to beat compute-matched single pass or worsen severe tails |
| C: trajectory to one pass | Recover part of B's gain without iterative deployment | Merely copying mistakes | No gain over A or GT-only student |
| D: calibrated extra compute | Spend time where measured benefit exists; improve selective behavior | Confident failures and shifted calibration | Inferior risk/latency curve to fixed K; no reliable improvement prediction |
| E: fusion/quantization | Smaller/faster implementation of an established accurate model | Hardware mismatch and color-sensitive numerical loss | Accuracy loss or no measured end-to-end latency benefit |

The final mechanism should be selected after comparison with the parallel visual-supervision and photometric-prior-art notes. This document supports a research decision; it reports no new model accuracy and grants no novelty or universal robustness claim.
