# Decision after90real-photo fits and complementary combinations

This goal turn is PROGRESS:54spatial fits,36training-only-branch fits,72inference
step diagnostics,27constant-offset evaluations and45fixed combinations completed.
No independent MSKCC TEST/CAL or UMINHO held-out endpoint was loaded. The goal
remains active and unmet. These are repeatedly examined source roles, not fresh
independent confirmation or ordinary smartphone facial validation.

## What survived the opponent checks

1. The original spatial-diffusion-at-inference hypothesis fails: mixed graph3
   mean3.6314 versus plain3.4855 and ordinary conv3.4901. Real adjacency loses to
   scrambled adjacency in all three mixed seeds. More passes do not guarantee
   improvement. Do not market this as a better neural architecture.
2. Post-hoc deletion of the graph at inference initially yielded4.5562/4.9341
   on SLR->iPod/iPod->SLR source transfer. Once zero-step checkpoint selection
   was fixed before a new experiment, results became5.8339/4.9736. Much of the
   first directional result therefore does not survive the selection protocol.
3. The second direction survives: fitting-only graph, iPod->SLR,4.9736 versus
   strong historical capture-plainMSE5.5089; all three seed gains. Its mixed mean
   3.6394 loses to3.4771; it is not a universal replacement. It has924,932active
   inference parameters and990,727active fitting parameters. Pure conv branches
   removed at inference fail badly; matching a weak removal control is insufficient.
4. Random graph-branch dropping recovers mixed accuracy3.4493 but is slightly
   worse than capture-mixture3.4406. Its reverse transfer5.7700 does not retain
   the always-graph training effect. Thus standard DropPath alone is not the answer.
5. A constant offset from TRAIN predictions only explains some graph-removal
   benefit: original checkpoint source-transfer5.3743/5.1920 versus unmodified
   6.3611/5.8068 and graph0 4.5562/4.9341. It does not fully explain the effect.
   This control uses no evaluation inputs/targets to calculate its offset.

## Combine successes, retain strong ordinary combinations

Fixed equal-weight capture-mixture + fitting-only graph gives mixed3.4043,
SLR->iPod5.0697 and iPod->SLR5.2806. The matched ordinary mixture+plain pair
gives3.3680/5.1702/5.6523. Proposed pair wins both transfer means but only1/3seeds
in the first direction and loses mixed mean. The ordinary pair is the strongest
observed mixed combination in this new set. No across-protocol victory.

Candidate pair has1,854,229active parameters; measured prepared-token batch1
RTX4060nativeLab output1.4243msmedian, peak40.68MiB. Ordinary pair1,858,594active
parameters. Timing is sequential CUDA-event microbenchmark, not total photo time
or evidence that similar architectures have reliably different latency. Models
were neither optimized nor exported. File sizes retain unused training modules.
All281repository tests pass; combination curves have7,920integer coverage points
and270independently checked fixed-coverage aggregations. The repeated plain
baseline exactly reproduces27prediction arrays from the earlier source experiment.

At80%disagreement-ranked coverage candidate pair worsens to5.3043/5.4846 in the
two transfer directions. Inter-model disagreement is not a reliable error head
here. Do not fit a production accept threshold from these explored endpoints.

## Next bounded experiments, and what would falsify them

Keep a separate candidate for **training-time nuisance perturbation with an
unchanged compact inference core**. Before more architecture scaling, isolate
whether benefit comes from the image-dependent perturbation, the learned branch,
or altered output calibration. Compare fixed versus learned diffusion and
nonspatial matched perturbations, preserving every method's absolute-color path.
Use both fitting directions and strong capture-plain/mixture controls. If a
simple constant or feature-independent perturbation explains the gain, discard
the special graph mechanism claim. Avoid claiming universal camera handling
from the one favorable direction.

Allocate a separate small screen to marginal/dependence representations: retain
raw absolute-color statistics while adding rank/coplanar channel-dependence
context. Copula representations have prior art. Strictly monotonic per-channel
changes preserve ranks, but channel mixing/clipping/local ISP can break this.
Cheapest test: verify the limited invariance, then compare matched raw-histogram
versus rank-dependence context on real source nativeLab. This remains a candidate,
not a selected winning architecture, and must not erase absolute lightness.

Next meaningful milestone still requires a better accuracy/coverage tradeoff
against strong matched controls and appropriate independent reference evidence.
Current source gains cannot replace the preserved400-image independent skin test.

[Spatial report](../benchmarks/skin_spatial_v1/report.md),
[training-only follow-up](../benchmarks/skin_train_branch_v1/report.md),
[fixed combinations](../benchmarks/skin_branch_combination_v1/report.md).
