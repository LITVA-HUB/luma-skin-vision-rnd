# Fixed combinations after complementary source effects

Post-discovery SOURCE experiment, not independent confirmation. This protocol
fixes equal nativeLab averaging before combination metrics are computed. No
fitted blend weight, opposite-camera checkpoint selection, new data or test use.

Same-seed pairs for each of mixed/from_SLR/from_ipod, seeds17/29/43:

1. capture-mixtureMSE + graph_always (branch disabled, new zero-step selection).
2. capture-mixtureMSE + graph_drop.
3. capture-mixtureMSE + conv_drop (ordinary spatial-training comparator).
4. capture-mixtureMSE + capture-plainMSE (strong ordinary two-model comparator).
5. capture-plainMSE + graph_always.

All pair members were trained on the same source data with the same80epoch
budget; their own same-camera validation selected their checkpoints. The combined
system has two networks and approximately twice the single-network fitting and
inference cost. Stored and inference-active parameters must be distinguished.
No ensemble or stochastic-depth novelty claim. Correlated models and shared
source selection weaken any uncertainty interpretation.

Average two nativeLab predictions with fixed.5/.5weights. As a diagnostic only,
use their mutualCIEDE2000 disagreement to rank acceptance. Report full mean/
median/p95, all100/95/90/80/70/60%coverage endpoints, and tails. Disagreement is
not calibrated expected error and may rank confident errors incorrectly. Compare
with constituent scores and the ordinary matched two-model system. Retain all
45pair/seed/protocol results; no best-seed or camera-dependent reporting.
