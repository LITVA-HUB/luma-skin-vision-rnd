# Native skin-color candidate-loss field: frozen source experiment

Question: does directly learning downstream color-loss geometry improve an
image model over ordinary Lab regression and a soft categorical-loss control?
Mixture/density likelihood is not required. The simplex parametrization still
admits a discrete-distribution interpretation; this is not a new Bayes principle.
Affine weights deliberately remove nonnegativity as an extreme ablation.

## Data and dictionary

Only existing TRAIN/sourceVALIDATION MSKCC caches. No TEST/CAL, ISSA or UMINHO
endpoints. For mixed and each training-camera subset separately, dictionary
atoms are exact distinct TRAIN native Lab reference values (np.unique order).
Candidates are a25x25x25 Cartesian Lab grid, each axis from TRAIN minimum-5 to
TRAIN maximum+5. It is a numerical decision grid, not a measured skin gamut.
Cost C[j,k] is actual CIEDE2000(atom_j,candidate_k). No new color reference is
manufactured. Count oracle nearest-grid TRAIN errors before fitting; later
report source-validation nearest-grid error separately from model accuracy.

Factor the full centered risk Gram matrix: G=(C-mean_rows(C))(C-mean_rows(C))^T/K,
normalized by mean squared centered cost. Keep all eigenvectors, clipping only
roundoff-negative eigenvalues; no low-rank truncation. Features Phi satisfy
Phi Phi^T=normalized G. Squared error of w Phi versus the true atom's Phi is
exactly the uniform candidate-risk MSE when sum(w)=1. Verify this identity.

These supervision targets are fixed-candidate losses against instrument Lab,
not residuals of an already fitted predictor. They require no in-sample residual
error head. Any later C+ residual calibration still requires person-held-out fits.

## Matched models and objectives

Four arms x3seeds17/29/43 x3protocols =36fits. Shared original CaptureColor backbone,
same local/context/vote features and an extra256-to-number-of-atoms linear head.
All share exact parameter shapes/initial states within a protocol/seed. Camera
and capture mode are never inputs. All arms retain0.1capture-mode CE auxiliary.

- direct: original standardized native-Lab MSE; atom head inactive.
- soft_ce: soft-target CE over atoms, target weights proportional to
  exp(-DeltaE00(true_atom,atom)^2/(2*2^2)). Width2 is fixed label smoothing,
  not a claimed measurement-noise model. w=softmax(logits).
- risk_simplex: MSE of w Phi versus targetPhi, with w=softmax(logits).
- risk_affine: same field loss, w=(1+logits-mean(logits))/number_of_atoms.
  Negative weights and negative predicted risks are possible and must be reported.

All80epochs, AdamW0.001,wd0.01,cosine to0.00001,16sitepairs/32images per batch.
For all atom arms the primary output is candidate[argmin_k(w C)_k]. Checkpoints
minimize same-camera source-validation patient-mean actual DeltaE00 of that
primary output. Direct selects its ordinary point. Also report atom-weighted Lab
as a secondary endpoint using the same checkpoint; no extra selection on it.
Evaluate other-camera only after same-camera selection. All36fits run unchanged.

## Metrics and rejection

Primary real reference error: mean,median,p95,patient-balanced mean,above5/10;
fixed100/95/90/80/70/60%coverage and full curves. Atom primary score is minimum
field value; direct uses existing hypothesis dispersion. Neither is calibrated.
Affine negative scores are explicitly unsupported expected errors, not clipped.
Report negative-weight/negative-risk fractions. Do not call scores guarantees.

This is repeatedly inspected source development, not independent confirmation.
Camera shifts also change people/capture composition. Compare strongest earlier
ordinary ensembles/graph and matched controls; no universal or novel win from
one direction. Record checkpoint size plus required fixed dictionary/cost storage,
active parameters and peak allocation. No export/latency work before evidence.

Tests/audits: Gram identity; affine unit mass and negative weights; finite model
gradients; fit-only dictionary/grid; scalar CIEDE2000 costs/metrics; exact model
replay; coverage and checkpoint selection. No participant identifiers or private
dictionary targets committed; publish only aggregates locally in Git.
