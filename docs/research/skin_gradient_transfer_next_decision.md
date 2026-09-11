# Decision: gradient conflict alone is not the missing skin mechanism

This turn is PROGRESS: a frozen six-model TRAIN audit,288 finite interventions,
random-group controls and actual scalar-verified skin errors change the next
action. The preceding user-facing clarification was a status response, not a
new experiment. The innovation goal remains active and unachieved.

## Findings that constrain the search

Mixed-camera models have lower true-person gradient agreement than shuffled
groups. That distinction is absent consistently in single-camera models. For
SLR/image, true mean cosine -0.1155 versus shuffled -0.1290 to -0.1159; for
iPod/image, true -0.0242 versus shuffled -0.0546 to -0.0167. Random grouping alone
can therefore produce the supposed symptom of person-specific bad learning.

Every one of288 fixed steps reduces its own group's objective, but116 increase
the pooled TRAIN mean skin DeltaE00. These are small, correlated local probes,
not288 independent experiments or a validation gain. All six pooled color/mode
cosines are positive; weighted auxiliary-to-color norm0.0423 to0.3310. This does
not identify either component as the cause of deployment error.

Cancel the assumption that negative group cosine is sufficient reason to
enforce gradient invariance. Do not launch a large Fish/Fishr sweep on this
evidence. They remain established comparator methods, not potential IP of this
project. Do not repeat the already tested no-mode/direct-squared-DeltaE arms
as if newly invented. Keep the stronger ordinary models and all negative runs.

## Next alternative space: relative color evidence

The absolute direct mapping from image statistics to native Lab has dominated
the search. Investigate whether comparisons with fixed TRAIN reference support
can offer useful information that survives capture differences. This is a
candidate relational model, not a discovered innovation or a new measurement.
The inference input would remain one new image; any support bank is fixed at
training time and counted in model bytes/compute. No test-camera references.

First attack a tempting but degenerate design analytically: if the predicted
relative color is f(x)-f(anchor), then averaging reference-anchored estimates
gives f(x)+mean(y_anchor-f(anchor)). It is only an absolute predictor plus a
constant offset, already an inadequate universal remedy in the previous phase.
Exact cycle consistency on a complete comparison graph likewise implies a
potential-difference representation. More comparisons alone do not create
physical information. Pair count does not increase independent people.

The narrow open hypothesis is a learned, context-dependent comparison quality
that weights compatible TRAIN observations without camera IDs. Non-additive
pair context may help matching, but permits inconsistent comparison cycles and
may memorize camera/person/color correlations. Static prototypes or compressed
support would have to meet the same compact single-image compute requirements.
This overlaps existing metric learning, relation networks and kernel regression;
perform a primary-source prior-art pass before designing fits.

Cheapest next falsifier: use original TRAIN only to test whether observational
same-site capture pairs admit a camera-label-free compatibility representation
that also preserves genuine between-site native-Lab differences. Compare true
same-site pairs with wrong-site pairs matched within person and capture mode;
preserve color-distance distributions where feasible and report matching
failures instead of inventing pairs. Compare ordinary descriptors, existing
learned context, and a simple color-distance predictor. Enforce excluded-person
fitting for any learned probe. Record both nuisance invariance and preservation
of actual color distinctions: collapsing all features is not success.

An encoder already trained on every original TRAIN person would leak that
person's labels into an apparent excluded-person probe. Either refit the encoder
inside each fold or label its frozen-feature comparison as descriptive only.
Excluding a person only from the final probe head is not sufficient. Historical
reuse of original TRAIN also makes this a source exploration, not confirmation.

Before neural pair fitting, verify the additive/cycle degeneracy and audit
support, pair registration limitations and person/camera confounding. If no
compatibility signal beyond simple color/identity shortcuts survives controls,
reject the relational mechanism. If it survives, freeze a small matched
absolute-versus-additive-versus-contextual-relative trial and both source camera
directions, with ordinary C+ and full/80% DeltaE00 comparisons. Never consume
exposed independent TEST/CAL for selection. No large architecture sweep first.

## Product and evidence boundaries

[Diagnostic report](../benchmarks/skin_gradient_transfer_v1/report.md).
Original MSKCC CC-BY, no newly downloaded data/weights, no proprietary collection,
cloud spending, publication or external messaging.288 exact state restorations,
12 independent finite replays,12 recomputed gradient Gram matrices,189336 scalar
color cases. Six original checkpoint files unchanged. No new fitted model or
independent skin accuracy, risk-coverage, unseen-phone or deployment claim.

Independent primary4.4570full/4.1591at80% versus ordinary fusion4.3005/4.1447
remains the authoritative public skin test. Product aspiration median<=2,p95<=5
at>=80% on suitable new facial-phone data remains unmet. Goal active and unmet.
