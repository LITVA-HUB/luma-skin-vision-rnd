# Decision: population support matters; this pixel branch is not a breakthrough

27 equal-update real skin fits and 36 frozen-model interventions are complete.
The endpoint is actual instrument-native Lab / DeltaE00, not illuminant angular
error or a tone class. Only the existing original TRAIN cache was read. The six
internal holdout people were previously in TRAIN in older experiments, so this
is strictly an exploratory mechanism screen, not an independent generalization
claim. [Full report](../benchmarks/skin_support_curve_v1/report.md).

## Measured outcome

The same baseline on nested 6/12/18-person sets has internal holdout mean
7.2439 / 6.2843 / 6.1082. Training error is 3.0095 / 3.4065 / 3.5823. All use
930 optimizer updates, identical holdout and matched sample streams per subset.
Thus the improvement is not explained by simply giving larger sets more updates.
But image count, site diversity, skin-color support and person diversity co-vary;
this does not identify one sole cause or an irreducible error floor.

The 18-person learned-pixel model scores 6.1258 versus baseline6.1082 and matched
statistics-adapter6.1560. Its descriptive patient interval versus the statistics
adapter crosses zero at every size. At80% these three models score respectively
5.4081 / 5.3918 / 5.4256 under identical input-novelty accept sets. No calibrated
selective result and no strong new accuracy win.

The added pixel branch is active: removing it from the fitted 18-person model
worsens mean error to6.6175. However shuffling RGB triples inside patches changes
predicted Lab by only0.0175 RMS and gives6.1266. Replacing each patch by its mean
gives6.1199 and0.0160 RMS change. Original statistical tokens are preserved in
these interventions. A necessary branch in a learned parameterization is not
proof that it contributes new information; useful local structure is unproven.

Do not compare these internal numbers with historical source3.4406 or independent
test4.3005 as if the same model became worse: population and budget differ.
No validation, calibration or independent test was reopened. Independent MSKCC
primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447 remains unchanged.

## Inverted assumption and next falsifier

The default assumption was that a better neural representation must be the next
bottleneck. This screen instead suggests testing how optimization allocates the
existing observations across actual skin colors. Increasing the branch alone is
not supported. A branch that barely needs texture does not justify a spatial
architecture novelty claim.

Next bounded study: keep the compact original core, same TRAIN-person roles and
equal update budget, compare image-uniform, person/site-uniform and TRAIN-color
support-balanced sampling. Only selected TRAIN native Lab may determine sampling
weights; holdout references cannot guide weights, bins or thresholds. Fix capped
weights in advance and preserve standard image-uniform evaluation. Camera labels
must not be inputs or balancing targets. Report mean, p95 and tails separately.
The mechanism is allocation of training effort to underrepresented measured color
regions. The failure mode is over-weighting noisy/extreme sites or harming common
colors. The cheapest falsifier is the fixed-role18-person screen before any wider
model search. Target balancing itself is established methodology, not an invention.

Retain an alternative representation path: color-sensitive local evidence with
explicit tests of what information it retains. Any new spatial/context mechanism
must first survive removal or permutation tests against a retrained matched
control. Do not revive a large teacher or renderer solely because this branch lost.

Recent [colorimeter-supervised dermatoscopy work](https://arxiv.org/html/2602.10265v1)
already directly learns Lab; our endpoint alone is not novel. Author protocols,
external pretraining and primary ITA reporting are not matched local baselines.

## Product and evidence limits

This experiment provides an auditable negative result and a useful next decision,
not a universal skin-color technology. Ordinary facial selfies, unseen modern
phone accuracy, calibrated refusal, cosmetics matching and innovation advantage
remain unproven. Original MSKCC CC-BY; no new outside data or model weights.
Goal stays active and unmet. No export/deployment promotion or external publishing.
