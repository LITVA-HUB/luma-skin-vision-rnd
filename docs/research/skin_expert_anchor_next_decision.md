# Decision after removal and conditional expert supervision

Actual instrument skin Lab / DeltaE00 remains the endpoint. The experiment
compares architecture removal with explicit expert supervision, not illuminant
accuracy or a different skin-tone label. The source cohorts are heavily reused.
No new independent or ordinary-phone validation is claimed.

## Mechanism evidence

On mixed source validation, baseline mean is 3.4406. Plain removal scores
3.4770, uniform anchoring 3.4755, conditional anchoring 3.5004. Thus neither
removal nor explicit capture specialization improves this primary population.

With the real mode supplied only for a diagnostic, baseline head mean is
4.3679, uniform-anchor head 3.5182 and conditional head 3.4865. Conditional mode
recognition reaches 0.7184 versus baseline 0.6048. Those auxiliary improvements
are real but do not make deployed skin color more accurate. A correct internal
label and individually better heads are not sufficient for the product endpoint.

SLR-to-iPod conditional raw mean is 6.2317 versus baseline 5.0193.
The final full table, both transfer directions, all seeds and same-acceptance
coverage curves are retained in [the report](../benchmarks/skin_expert_anchor_v1/report.md).
A label-assisted diagnostic must never be used as the achieved inference score.
No calibrated C+ comparison or uncertainty bound follows from common novelty
ranking. New source findings do not revise the old independent MSKCC test.

The useful partial finding is the plain model plus observed-patch sampling.
It improves plain SLR-to-iPod from 5.8301 to 5.4150 and iPod-to-SLR from 5.5087
to 5.3047, in all six seed comparisons. Descriptive patient intervals for these
paired-minus-raw differences are [-0.5887, -0.1252] and [-0.2507, -0.1353].
However, mixed mean worsens from 3.4770 to 3.5436. The strongest prior forward
control is 4.8328 and the historical reverse graph control is 4.9736. This is
transfer regularization evidence on reused source data, not a universal winner.
Common novelty ranking also fails to reduce reverse error: plain paired is
5.5999 at 80% versus 5.3047 at full coverage. Preserve that selective failure.

## Cancel another assumption

The present family assumes that discriminatively mapping appearance to native
Lab is the most useful direction of learning. The next competing mechanism
will test a conditional appearance model: what observed color statistics are
plausible given a candidate skin Lab, while marginalizing capture nuisance at
inference. Inverting this model would compare candidate colors by how well they
explain the observation. Camera ID remains unavailable at inference.

Mechanism: preserve multiple explanations rather than forcing one camera mode
or one regression mean early. Key assumption: appearance conditioned on measured
skin color is learnable with the existing real source diversity. Advantage if
true: an explicit score for unsupported or ambiguous observations, with cheap
candidate refinement. Likely failure: flexible nuisance explains any skin color,
so likelihood is uninformative or favors high-variance/bias solutions. A good
forward reconstruction alone would not validate inverse skin accuracy.

Cheapest falsifier: a regularized low-dimensional conditional appearance model
on TRAIN regional statistics, with fixed source-only inverse scoring and direct
regression controls. A single joint Gaussian reduces to ordinary conditional
Gaussian regression and cannot be treated as a fundamentally novel solution.
Only a justified nonlinear/multimodal nuisance model would test the distinct
hypothesis. Check candidate identifiability and actual inverse DeltaE00 before
adding a neural renderer or iterative inference. Do not invent reference colors
or assume a known camera transform to score real data.

Retain a separate learning-curve check: vary the number of TRAIN people under
fixed optimization budgets to distinguish data support from model complexity.
Reserve one bounded combination of the earlier training-only graph mechanism
and observed-patch sampling, with raw and matched plain controls, after checking
that its spatial/input assumptions survive the derived bags. This combines two
partial transfer leads; it must not consume the whole search or preempt the
alternative forward-appearance hypothesis.
Neither direction is currently a measured improvement or an invention claim.
Freeze their protocols before fitting; no new fits are reported in this decision.

## Product boundary

Independent MSKCC primary mean remains 4.4570, selected 80% mean 4.1591;
ordinary fusion remains stronger at 4.3005 and 4.1447. These are known-camera,
new-person clinical/dermoscopic tests. Exposed TEST/CAL remain evaluation archives.
No ordinary facial selfie accuracy, cosmetics matching or universal device claim.
No new external datasets/weights, no export, no external publication. Preserve
negative synthetic, prior color-constancy and current skin results. Goal unmet.

All 72 fits are complete: 216 exact color-array replays, 18 exact original
baseline best/final state refits, 12,672 scalar color cases, 432 coverage rows,
12,672 curve points, 3,456 anchor arithmetic checks and 9,504 label-assisted
scalar scores. Full test suite: 337 passed, 14 historical warnings in 40.02 s.
Plain model has 924,932 parameters; others 929,297. Peak fit allocation 108.51
MiB; no new isolated inference latency or export claim. All jobs terminal.
