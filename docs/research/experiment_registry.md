# Experiment registry

Each `experiments/runs/<id>/run.json` is a transparent registry entry. Failed runs retain FAILED with reason; successful runs have unique directories. `python scripts/list_experiments.py` lists all historical runs. This table identifies the canonical final smoke; earlier smoke records are retained locally as superseded engineering runs.

| Method | Run | Git commit | Source SHA256 | Status |
|---|---|---|---|---|
| baseline_a0 | [20260910T155416_baseline_a0_4124afad](../../experiments\runs\20260910T155416_baseline_a0_4124afad/run.json) | c5f0707cbe752c94f3a485e1779c16648d4cadb8 | b12faad7b06b77262cc7c69fb8a3ad469d8c3e8035a64f4b2406060e92fe3335 | COMPLETED / SYNTHETIC |
| baseline_a1 | [20260910T155423_baseline_a1_8d8b4a07](../../experiments\runs\20260910T155423_baseline_a1_8d8b4a07/run.json) | c5f0707cbe752c94f3a485e1779c16648d4cadb8 | b12faad7b06b77262cc7c69fb8a3ad469d8c3e8035a64f4b2406060e92fe3335 | COMPLETED / SYNTHETIC |
| baseline_a2 | [20260910T155430_baseline_a2_262f80c7](../../experiments\runs\20260910T155430_baseline_a2_262f80c7/run.json) | c5f0707cbe752c94f3a485e1779c16648d4cadb8 | b12faad7b06b77262cc7c69fb8a3ad469d8c3e8035a64f4b2406060e92fe3335 | COMPLETED / SYNTHETIC |
| baseline_c | [20260910T155437_baseline_c_78c51802](../../experiments\runs\20260910T155437_baseline_c_78c51802/run.json) | c5f0707cbe752c94f3a485e1779c16648d4cadb8 | b12faad7b06b77262cc7c69fb8a3ad469d8c3e8035a64f4b2406060e92fe3335 | COMPLETED / SYNTHETIC |
| baseline_c_plus | [20260910T155456_baseline_c_plus_dd6ab32d](../../experiments\runs\20260910T155456_baseline_c_plus_dd6ab32d/run.json) | c5f0707cbe752c94f3a485e1779c16648d4cadb8 | b12faad7b06b77262cc7c69fb8a3ad469d8c3e8035a64f4b2406060e92fe3335 | COMPLETED / SYNTHETIC |
| proposed_v1 | [20260910T155516_proposed_v1_3c4c3c76](../../experiments\runs\20260910T155516_proposed_v1_3c4c3c76/run.json) | c5f0707cbe752c94f3a485e1779c16648d4cadb8 | b12faad7b06b77262cc7c69fb8a3ad469d8c3e8035a64f4b2406060e92fe3335 | COMPLETED / SYNTHETIC |

Command transcript: `C:\Users\dimal\Documents\просто\luma-skin-vision-rnd\artifacts\smoke_20260910T155410\commands.json`. Configs, optimizer settings, precision, seed, resolution, augmentation, dataset/split hashes, pretrained provenance, environment, VRAM, duration, checkpoint and evaluations are stored in the linked run directories. Those directories are excluded from Git because models/private-data derivatives belong under controlled artifact storage.
