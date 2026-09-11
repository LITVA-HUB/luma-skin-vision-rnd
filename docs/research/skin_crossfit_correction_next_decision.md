# Decision: preserve stable-color correction, reject OOF-superiority explanation

Previous goal turn was PROGRESS:27 larger neural-reference fits and complete
audits were committed. This turn is PROGRESS:9 excluded-person encoders and9
matched heads were fitted, original cores replayed, and results audited.

## Measured partial improvement

Actual native skin DeltaE00, same original-TRAIN18-person support /6-person
internal evaluation,3 individual seeds:

| Method | Mean | p95 | Mean at80% |
|---|---:|---:|---:|
| Unchanged core | 6.1082 | 14.9960 | 5.6609 |
| in_full correction | 5.8579 | 13.6359 | 5.4313 |
| in_matched correction | 5.6560 | 12.9644 | 5.2608 |
| out_person correction | 5.8945 | 13.4503 | 5.5260 |

All9 corrections beat their core on mean error. Leading in_matched improves all
three seeds and5/6 people after seed averaging, mean7.40% lower. It uses a
188,035-parameter stable39-input head, total1,117,332; no reference memory or
camera metadata at inference. Historical person/site/color allocation means
6.0084/6.0012 on this cohort are also higher, but their compute/capacity differ.
No independent or universal win: both cameras are known and these6 people have
been repeatedly evaluated. Existing independent4.4570 versus fusion4.3005 stays.

## Scientific opponent: the proposed explanation lost

out_person trains on genuine encoder-held-out errors, but is not the best head.
in_matched uses the SAME12-person inner encoders/budget, routed to queries they
did see; it wins. Full-table base errors3.5823,matched3.3675,OOF4.3241 show real
supervision differences, not merely labels attached to identical predictions.
The12-to18 deployment-support shift affects matched/OOF similarly but remains
a limitation. It does not permit asserting why the matched arm wins.

Stable color features replace551-dimensional neural context and reference
regression in this screen. Because head shape, representation and supervision
all differ from the previous phase, the gain does not isolate context removal
as its cause. Known stacking/correction is not novelty. Do not turn a failed
OOF-superiority hypothesis into a positive story by renaming the winning arm.

## Next decisive experiment: unchanged heads on camera-held-out source banks

Keep all three correction arms and fixed39->384->384->64->3 architecture,
300 head steps,930 inner-core steps, existing losses/optimizer/seeds. Preserve
the no-head baseline and all current positives/negatives. No configurations
should be tuned to source unseen-camera outcomes.

Use source TRAIN24 and VALIDATION6, with explicit mixed,SLR-only andiPod-only
protocols. Prefer existing skin_sampling_transfer_v1 image final cores for
seeds17/29/43: same930-step fixed-final recipe, no source checkpoint selection.
Verify exact core training membership/scales/hashes before reuse.

Four balanced inner folds permit full8SLR/16iPod support: each fold holds2SLR/
4iPod in mixed;2SLR or4iPod in single-camera protocols. Inner cores use18 mixed,
6SLR or12iPod people, with each query person's full data excluded for out_person.
Matched inclusion routes to the next cyclic fold, one encoder per query.
36 inner fits +27 head fits +9 unchanged-core controls across3 protocols/seeds.
This is a separately frozen follow-up, NOT ALREADY RUN. Adapt only role counts,
not head settings or candidate selection. Every source validation person must
remain absent from every fit and scale. No independent CAL/TEST access.

If stable corrections retain benefit in both unseen directions, then compare
them with stronger80-epoch historical recipes and a truly matched enlarged
representation control before claiming an innovation. Investigate multi-predictor
training as a regularizer only with its own ablation; do not infer that multiple
inference passes would help from these single-step results.

If transfer fails, retain the internal improvement but stop calling these
corrections camera-general. Revisit the image representation/acquisition shift
and whether a correction head is needed at all. The earlier context/pixel
negative experiments must inform that branch instead of being repeated blindly.

## Integrity and resources

All fits terminal.9 inner/3 outer replays;9 scale/exclusion checks;4404 routing
rows;1 full core/3 full head refits;9 NumPy heads;12 exact arrays;9390 scalar
color values;72 coverage rows. Complete model4,477,653bytes and78 scale scalars;
no bank. No new latency/export claim. Original MSKCC CC-BY; no new data/weights.
Ordinary phone skin accuracy, calibrated rejection and novel superiority remain
unvalidated. Goal active and unmet.

[Audited report](../benchmarks/skin_crossfit_correction_v1/report.md).
