# V3 architecture and code provenance

The color-frame graph/posterior code was implemented locally in `scripts/cc_v3_model.py` without pretrained weights or imported frame-network code. The mathematical canonicalize/process/restore pattern and GL frame construction have published antecedents documented in [the V3 prior-art note](../research/cc_v3_prior_art.md). This implementation record does not establish inventorship, patent novelty, patentability or freedom to operate.

`scripts/cc_v3_ffcc.py` is an explicitly marked adapted PyTorch implementation of formulas from original `google/ffcc` at commit2fa9e1316954dbd3913630b7d597927941b4dd32. It retains Google2017 attribution and Apache-2.0 notice, names source files, and links to the pinned unmodified source/license under `docs/research/cc_v3_sources/google_ffcc/`. That directory has per-file SHA-256 and upstream Git blob records. No original trained weights, datasets, MATLAB runtime or minFunc dependency were adopted. No MFA or other frame/canonicalization code was adopted.

Current graph training lineage is SimpleCube++ only, original CC BY4.0. New INTEL-TAU bytes follow original CC BY-SA4.0 and are governed by the separate [V3 usage decision](../data/cc_v3_usage_decision.md); no graph weights in this source screen use them. Do not infer model/data rights from a mirror's MIT label or from the Apache code license. No external publication has occurred.

The real full-frame graph result is negative and retained. The FFCC core currently has only constructed numerical/source-correspondence evidence, with no MATLAB runtime parity or real benchmark training result. Do not market either as a proven novel improvement. A revised graph/global-color-state design remains planned and requires independent empirical evidence.
