# Exact C fold recovery forensics

Status: C_FOLD_ASSIGNMENTS_NOT_RECOVERED_FROM_AVAILABLE_REPO_HISTORY. This is a derived experiment-state loss; original TRAIN photographs and references remain available. No training or fold generation was performed.

## What is recoverable

The original MSKCC TRAIN membership is recoverable from original metadata and `scripts/skin_mskcc_data.py::patient_roles`. The immutable split receipt records 966 images / 24 people / 248 exact sites (643 iPod, 323 SLR). It supplies the original role-manifest SHA256 `ad3669c87d92410b080b6d06a37ae392fe0f85fd96aa79a1ba20ed334b2bae08`, but contains no intra-TRAIN fold assignments. Recovering TRAIN membership does not recover its cross-validation partition.

## Searches performed

- Current full repository source/docs including September 14 archive: exact terms private_folds, run1, oof_anonymized, LUMA_TRAIN_OOF_C, luma_face_v3; no historical C implementation or assignment file.
- Fully fetched git history: `git rev-list --all --objects` checked for named artifact paths. Counts are below.
- `git log --all --oneline -S'private_folds' -- .` returned only the September 15 audit report documenting its absence.
- `git log --all --oneline -S'LUMA_TRAIN_OOF_C' -- .` returned no commit.
- Old source-package inventory and package_manifest.json describe the September 10 Luma-2.0 package, not the later face_v3/C OOF package.

## An unrelated reproducible fold recipe exists

`skin_mskcc_selective_core.patient_folds` sorts 24 TRAIN patient IDs by SHA256("LumaMSKCCRISKv1|"+patient ID), assigning 4 people per fold (six folds). Its paired protocol explicitly names plain patch-vote color models, three seeds 17/29/43, and source-validation-selected risk heads. That is a different older experiment. It is not evidence for the later color36 36–64–3 MLP C folds. Other archived families use three or four folds. Substituting any of these for C would manufacture lineage.

## Why aggregate matching is insufficient

The five historical aggregate metrics cannot identify the missing row-to-fold mapping. Trying folds/seeds until aggregates agree would select on the historical evaluation summary and would not restore the original experiment. An authenticated original mapping, code with exact version/order, or its original manifest/commit lineage is required before a run can be called reproduction of C.

No direct participant or image IDs are included in this note.

## Evidence hashes

- `scripts/skin_mskcc_data.py`: `12f7ce2df93ca3586fa068d92751670c33372f7f69bad4d2ca550463e4de0fc5`
- `docs/data/provenance/mskcc_skin_v1/split_receipt.json`: `4bd7ac3ce0276a6cfb4be7e0cf38d61bd8873706ce5d7e10a615fbf38fc30ed4`
- `scripts/skin_mskcc_selective_core.py`: `a5151967444791123ee5a3a70707419313cc0e187f06a091b2fd8538ecc052b4`
- `docs/research/skin_mskcc_selective_protocol_v1.md`: `a48c53f872332bc541e4ca66149e6cf0b62adddeda35e0a910452e27ebbf60f9`
- `docs/context/provided_start/package_manifest.json`: `29085002019f5869c5f79ebd01a91b0ed2f9ac9d8f2289fd54e341a012c78243`
- `docs/context/provided_start/archive_manifest.txt`: `1a7d11f53c89e2accef0c60f6f6b64436438bc9d4c1d986e21bd5d8f5a3c1189`

## Full-history object-path match counts

{
  "private_folds": 0,
  "run1/": 0,
  "LUMA_TRAIN_OOF": 0,
  "oof_anonymized": 0,
  "luma_face_v3": 0,
  "abc_experiment": 0
}
