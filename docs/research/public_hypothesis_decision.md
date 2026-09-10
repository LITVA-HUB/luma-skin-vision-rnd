# Public hypothesis decision: provisional Direction C experiment

## Final bounded-milestone decision

Eight real training/evaluation runs are complete. The mixture mechanism is deprioritized: three-seed risk80 is1.586° versus strong control1.556°, and Sony transfer5.520° loses to Gray World4.073°. The modest full-coverage and same-sensor holdout improvements do not establish robust superiority or novelty. Continue direction C using the ordinary compact estimator and improved risk calibration as the reference, with new diverse-camera data and locked tests. Physical downstream surface-color supervision remains contingent on actual reference data. [Full measured report](../benchmarks/public_benchmark_report.md), [independent review](public_review_record.md). The ranking/proposals below describe the pre-experiment rationale, not additional completed experiments.

Date: 2026-09-10. The ranking below was provisional before experiments. Stage1–7 update: four classical baselines and the compact MobileNet baseline are now reproduced on real SimpleCube++; the standard model achieves 2.219° mean reproduction error versus Shades of Gray3.573° on462 official test images, but fails on the external Sony30 pilot (9.739° versus Gray World4.073°). Proceed with the already frozen mixture/disagreement feasibility variant in [protocol v1](public_protocol_v1.md), without changing its hyperparameters or fitting to Sony. This makes direction C's transfer problem concrete; B is the bounded mechanism, and A remains an angular-risk proxy only. Final outcomes supersede planned statements below; see the public benchmark report. Evidence and source/license details: [public_color_prior_art.md](public_color_prior_art.md). Historical synthetic evidence remains separate.

## Decision and ranking

Provisionally prioritize **Direction C**, a narrow test of compact, single-image illuminant estimation with selective reproduction-error risk. **C+ means the strong matched baseline throughout this project; it never denotes the proposed method.** Direction C is the candidate backbone; A supplies the deployment decision; B is a bounded ablation. Final direction selection waits for the requested evidence stages 1–7 and local baseline measurements. This is a testable engineering research hypothesis, **not a novelty finding**. A possible contribution would require a reproducible accuracy/coverage/latency improvement over C+ under the same information budget.

Scores below are provisional qualitative judgments after the linked primary-source review; 5 is favorable. They do not represent measurements, probabilities or a final choice.

| Direction | Distinctiveness after review | Available GT | RTX 4060 / speed feasibility | Commercial fit | Duplication risk | Decision |
|---|---:|---:|---:|---:|---|---|
| A: downstream-color-risk selective normalization | 2 | 2 physical Lab; 5 angular proxy | 5 | 4 if from scratch on cleared data | High: uncertainty/cascade and reproduction-error literature | Rank 3 standalone; use angular-risk selection only. Physical color claim deferred. |
| B: cheap hypotheses + disagreement/context residual prediction | 2 | 5 angular | 5 | 4 if independently implemented | Very high: Reweight-CC, GC3, C4 and uncertainty2025 | Rank 2; practical cheap comparator/ablation, not lead novelty claim. |
| C: compact single-image unseen-camera generalization | 2 | 3 until truly distinct cleared cameras acquired | 5 | 4 conditional on data rights | High: C5, CCMNet, recent generative methods | Rank 1 feasibility; strict camera-blind/single-image constraint defines matched scope. |

None scores highly on demonstrated distinctiveness. Combining known parts does not itself resolve duplication. If comparisons show only routine gains or no benefit, report the result and stop rather than rename it as a new method.

## Proposed Direction C hypothesis and experimental contract

**Hypothesis:** a from-scratch 1–5 million parameter image encoder with an independently fitted risk head using out-of-fold residuals improves selective reproduction angular risk relative to the strong matched C+ baseline, while keeping useful coverage and local inference costs. The immediate experiment uses a common-sensor source and an external author-provided Sony 30-image pilot; this small external check cannot establish broad unseen-sensor generalization.

The positive result must hold at matched input size, training data, masking, augmentation budget, model selection procedure, seeds, accepted coverage and runtime reporting. It must survive controls that isolate the risk target, source-camera training and candidate disagreement. There is no basis yet for a claim of improved physical skin color or measured Delta E 2000.

Inputs are one downscaled linear raw-RGB image with black level handled and calibration target masked. Fixed documented preprocessing may read the information needed to decode raw; **no camera identity, target CCM, target ground truth, other target images or target-set histogram statistics enter inference**. Ground-truth metadata is evaluator-only. No target-camera fine-tuning, threshold fitting, batch adaptation, batch-norm updates or test-time training. An evaluation server can know camera IDs for scoring without exposing them to the predictor.

SimpleCube++ is the first real-data pipeline/reproduction benchmark. Its two cameras share sensor type, so a random split is explicitly **within-source**. Narrow the immediate practical plan to this source plus the author-provided Sony 30-image external pilot, conditional on the dataset audit recording its origin, GT and terms. Keep this pilot untouched during training, tuning and calibration; label every result as a small external pilot, not a definitive unseen-camera benchmark. Avoid Cube+/Cube++ sensor overlap and cross-dataset duplicates. NUS-8/Intel-TAU are future leads, not required acquisitions for this bounded experiment. Broad cross-camera generalization remains **UNTESTED** even if the Sony pilot improves.

## Matched baseline ladder

1. Identity/no correction for diagnostic context; Gray World, Shades of Gray and a documented Gray Edge implementation. Fix hyperparameters on source validation only. Measure complete preprocessing cost.
2. **C+ strong matched baseline, required:** a standard compact architecture trained from scratch with conventional illuminant regression and a strong source-only confidence/risk selector. Match Proposed on inputs, parameter budget, training split, augmentations, optimizer budget and calibration data. Include no-rejection, simple image-quality and cheap disagreement selectors so Proposed is not compared only with a weak accept-all model.
3. **Proposed:** the same compact backbone plus the predeclared residual/context or training modification being tested. An out-of-fold risk head alone is not sufficient differentiation if C+ already uses one. Isolate each actual modification in ablations; do not call ordinary architecture implementation a paper reproduction.
4. **Optional publication reproductions:** FC4, Reweight-CC, query-only C5, GC3 and uncertainty2025 are valuable additional comparisons only when equations, source behavior and terms can be verified within budget. None has been run in this review. Label adaptations as reimplementations; preserve the different camera-normalized Reweight contract. Never relabel published multi-image C5 scores as query-only results.
5. CCMNet/VLM-CC/GCC published numbers belong in a separate contextual table with their additional inputs and hardware; they are not matched baselines or locally reproduced results.

Architecture starting point: 256-pixel longest side, compact convolutional encoder, positive illuminant output, small scalar risk MLP; total target 1–5M parameters. Optional B candidate bank is fixed Gray World/Shades of Gray/Gray Edge plus the one network estimate. It uses no second neural pass. Add thumbnail context and candidate angular distances to the risk MLP only in the designated ablation. Exact widths, resolution and optimization are chosen on source validation and frozen before test. Do not assume declared parameter count proves speed or 8 GB fit.

## Leakage-resistant fitting

For the immediate study, reserve source capture-scene groups for within-source testing and the Sony 30 images for an external pilot. Inside the common-sensor source, separate train, model-selection validation and threshold-calibration groups. If the same scene appears across camera captures, keep all versions in one group. A common-sensor source cannot support meaningful camera-held-out inner validation; label this limitation. Never use external-pilot imagery to construct synthetic camera augmentations. A future multi-sensor study would require outer sensor-held-out and scene-group-disjoint splits.

Generate each risk-training label from a base model that did not train on that image's group. Fit risk heads on these out-of-fold labels. Freeze the base model, risk head and all thresholds before the outer test is read. With a refit base model, do not present fold residuals as exact residuals of the refitted model: calibrate on a fresh disjoint source group or predeclare a fixed fold-ensemble deployment contract and its cost. Threshold calibration on sources cannot guarantee calibration under an unseen-camera shift.

## Metrics and bounded ablations

Primary loss is established **reproduction angular error**; recovery angular error is secondary for compatibility. Report mean, median, trimean, worst-quartile mean and 95th percentile, by camera and pooled. Report invalid estimates and clipping separately. Use scene-group bootstrap confidence intervals and equal camera weighting in cross-camera aggregate summaries so one large dataset cannot dominate.

For a scalar predicted risk `s`, accept iff `s <= t`. Coverage is accepted fraction. Selective risk is mean reproduction error among accepted images; no accepted images means undefined risk, not zero. Risk-coverage curves/AURC and retained risk at 50%, 80%, 90% coverage compare rankings. Target-test quantile curves are diagnostic only; deployed decisions use source-calibrated frozen thresholds and report their realized test coverage. Include random selection, accept-all and oracle ranking as labeled controls; oracle is not deployable. A predeclared bad-correction event (e.g. reproduction error >5 degrees, an operational threshold rather than a universal perceptual bound) allows Brier score and reliability diagrams.

Bound the first study to the baseline ladder and these controls, sharing three fixed seeds:

| Control | What it resolves |
|---|---|
| Same backbone, no rejection vs pooled confidence vs learned risk | Selection benefit beyond prediction changes |
| Predict recovery residual vs reproduction residual | Whether the target choice matters; does not claim a new metric |
| Disagreement-only vs context-only vs combined risk inputs | Whether B adds information beyond cheap image cues |
| Ordinary source training vs predeclared source-only augmentation | Whether the proposed training change adds value with identical data budget; no claim of multi-camera training on this source |
| Brightness augmentation off/on for both C+ and Proposed | Guards against attributing known robustness training to Proposed |

First train and measure the standard compact C+ baseline, then run one-seed source-validation feasibility across these controls; expand only promising predeclared contrasts to three seeds. This is a local baseline run, not reproduction of a named paper unless independently verified as such. Published comparison tables remain separate from actual runs. Record commit, manifest/hash, split hash, source-license snapshot, seed, hyperparameters and checkpoints. Do not tune on source-test or external-pilot curves.

## Go / no-go criteria and hardware accounting

Pre-register before final evaluation: pursue Proposed only if it improves mean reproduction risk at 80% diagnostic coverage by at least 10% relative to C+ with its strongest matched feasible selector, has a scene-bootstrap confidence interval excluding no improvement on the adequately sized source test, and does not worsen full-coverage mean by more than 5%. Frozen source-threshold test coverage must remain useful (target at least 70%). Report the 30-image external pilot separately with uncertainty and individual failures; it is too small to carry a broad sensor-generalization conclusion. These are provisional project decision margins, not literature facts or guarantees. Final direction choice follows the requested stages and local measurements, not this literature ranking.

Use an RTX 4060 8 GB, batch one, fixed precision and resolution, warm-up then synchronized timing. Record median/p95 latency for model-only and decode/preprocess/model/postprocess, peak allocated/reserved GPU memory, parameters, checkpoint bytes, CPU timings and training wall time. A provisional engineering gate is <=20 ms model p95 and <=6 GB measured peak GPU memory, to preserve working headroom; revise only on source feasibility evidence before final test. No paper's GPU figure is a measured 4060 result. CPU smoke tests verify execution, not real-time operation.

Current conclusion: proceed with public-data baseline infrastructure, measure C+ on the common-sensor source, then evaluate frozen models on the small Sony external pilot. Keep the ranking provisional and leave broad cross-camera and physical downstream color accuracy unclaimed. No model has been shown better, faster or commercially deployable by this review.
