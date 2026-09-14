# Local-search research asset lineage

2026-09-13. User-authorized local research resumed in an isolated worktree.

- Data: the already acquired original MSKCC CC-BY photographs / instrument-native Lab, via the hash-locked original TRAIN feature cache. Attribution and existing acquisition receipts remain applicable: [MSKCC provenance](skin_mskcc_provenance.md). No new dataset or pretrained weights adopted.
- Representation: existing local color36 feature extractor. These experiments consume cached statistics, not a full deployed face detector or identity model.
- Code: locally implemented PyTorch/NumPy ridge, Gaussian features, Schur search, Adam MLP and fixed post-training ablations. Existing dependencies and their separate license notices apply.
- Weights: fitted locally; RBF centers and KRR reference vectors derive from training features and are included in model size. Raw photos, row identifiers and prediction arrays remain in ignored local directories. These are research artifacts, not a release package.
- Prior art: OMP, stochastic configuration networks, ELM and kernel ridge. Algorithm implementation and training-error reduction alone do not establish patent novelty.

No external publication, upload, integration with a retailer, or claim of ordinary-phone facial-color validation is part of this run.
