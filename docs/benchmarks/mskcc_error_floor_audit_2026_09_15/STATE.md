# STATE — 2026-09-15

Status: REFERENCE_CAPTURE_AUDIT_COMPLETE; C_ERROR_DECOMPOSITION_BLOCKED_MISSING_OOF.

Completed with original TRAIN966/24people/248sites, no training:
- original six metadata SHA checks;
- all966 original JPEGs,1062327919bytes, identity/license/content integrity checks;
- instrument3-assessment repeatability, exact-site shared-label audit, diagnostic oracles;
- full143.011s CPU capture decode and1421 same-site pair comparisons;
- coordinate histograms and frozen reference/capture/clipping diagnostic masks;
- anonymized features, numerical report, figures, reproducible analyzer code.

C and inference were neither refitted nor replaced. The current workspace began
empty; the former C checkpoints/OOF/cache/ZIP are not present. Original remote
main remains9cad271aad159d73083259967b4107b2ce828eb1. No push or merge performed.
No source-validation/calibration/test numerical endpoint was analyzed.
No new fold assignment was generated.

Needed next: user reattaches LUMA_TRAIN_OOF_C.zip with
experiment/private_review/oof_anonymized.npz. Keep original archive provenance.
Do not regenerate C predictions by training.

Continue without images:
1. Restore original metadata with scripts/error_floor/restore_metadata.py.
2. Run instrument_audit.py to reconstruct private original row mapping.
3. Reuse attached data/capture_features_anonymized.npz + frozen masks.
4. Inspect OOF NPZ keys, then run c_oof_audit.py with explicit keys and
   --capture-anonymized. It validates indices, targets, groups and masks.
5. Finish C grouped residuals/top50/counterfactual tables and qualified decision.

Even after OOF recovery, independent cross-session site reference repeatability
cannot be estimated: one shared S4 triplet/site; zero same-mode photo repeats.
Do not turn measured capture differences into an irreducible model error floor.
No job remains scheduled to continue in the background.
