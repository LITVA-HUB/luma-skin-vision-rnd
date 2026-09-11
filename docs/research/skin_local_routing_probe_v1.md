# Post-hoc TRAIN routing diagnostic, not image-model validation

Use six frozen mixed-source models (baseline/paired_union, seeds17/29/43).
Use all1421unordered TRAIN same-site pairs, one mixed bag per pair, fixed
NumPy seed4017, paired_union sampling with augmentation forced on. Inputs and
original native Lab targets are identical across compared routing rules.
No new model fitting or held-out endpoint access.

Inspect the existing per-patch/four-mode color hypotheses and confidence weights.
Compare learned global gate; known source-mode proportions as a global gate;
confidence-weighted known proportions as a global gate; known per-patch source
mode routing; and fixed within-bag permutation of those per-patch mode labels.
All known-mode variants are teacher-assisted TRAIN diagnostics, not deployable
single-image results. True Lab is used for scoring only, never route selection.

The comparison asks whether local correspondence between hypotheses and actual
capture source can help these frozen experts. It cannot prove a learnable local
router will work, identify why earlier fits failed, or rule out differently
trained experts. Pooling and gating need not commute when the gate varies by
patch; numerical unit tests are not scientific accuracy evidence.

The same permutation stream(seed9123) is used for every model. Record all five
rules and all six models, exact input/weight bindings, independent scalar color
checks, and aggregation identity for constant gates. No positive-only selection.
