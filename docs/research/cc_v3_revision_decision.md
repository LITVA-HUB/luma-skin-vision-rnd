# Decision after the first color-frame graph screen

**Do not promote the initial full-frame architecture.** Three capacity/budget-matched source runs show4.281° mean reproduction for full-frame Proposed,2.942° diagonal and2.547° direct RGB. Full-frame risk80 is4.155° versus2.304° direct. Results are development validation, one seed, and are preserved in [the report](../benchmarks/cc_v3_report.md).

The experiment is materially different from the earlier CNN, but originality of appearance is not a scientific success criterion. The prior-art review found explicit GL frame/canonicalization precedents. A useful contribution still requires better measured accuracy/coverage/compute under the intended camera contract.

## Evidence-driven hypothesis revision

The source conditioning audit found median frame condition222 and median canonical GT/GW separation65.36°, despite only4.52° mean ordinary GW reproduction error. Full-frame training also barely improves over GW on the training population. The positive full-rank test did not ensure a useful representation. Diagonal posterior spread ranks this validation population poorly despite decent point estimates.

The next architecture candidate is a **graph with an explicit global color state**. Keep canonical relational scene features, but separately retain image-derived global chromatic statistics and frame geometry. Let a learned gate combine these branches. Predict a strictly positive camera-space correction and a camera-space posterior, avoiding an ill-conditioned canonical angular target. This deliberately relaxes full-GL invariance; do not carry the old exact transport theorem onto it.

This is a planned, unverified hypothesis, not implemented or selected as superior. Before fitting, specify exact global features, fusion/decoder equations, parameter budget and loss; compare direct/diagonal/full relational inputs with the same global branch and same budget. Ablate the global branch and transported geometric risk. Keep camera identity, CCM, test-time population statistics and target-camera fitting absent where the protocol requires them. Restore no negative result under a renamed model.

Also complete source training of the source-grounded FFCC-inspired control. Its compact posterior and original formula provenance make it an important comparator. Without MATLAB/protocol/optimizer parity, label it as an inspired control and never claim to reproduce author benchmark scores. The current FFCC numerical implementation is not yet a trained real-data baseline.

## Camera experiment after the source decision

The384 new images are an acquisition population, not a frozen evaluation protocol. Audit near duplicates and decide before new GT/error inspection whether to exclude the67 rows sharing old V2 reference bytes from all new primary roles, leaving317. This choice must be based on identity/leakage evidence, never measured prediction errors. Freeze complete fold-local train/validation/risk/calibration/test roles, grouping reference hashes and keeping each outer camera entirely absent from every fitted component. No hyperparameter search on pooled outer-fold results.

The original BY-SA data terms and separate model lineage apply to any later multicamera training. SimpleCube-only models remain separately identified. No publication, paid resources or proprietary facial collection is required.

The broader innovation goal remains active. Next required work is an actually tested revision and stronger compact control, followed by locked camera generalization and held-out error calibration. The initial source negative is a completed experiment, not project completion.
