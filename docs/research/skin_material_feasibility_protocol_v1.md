# Post-hoc TRAIN-only material representability diagnostic

This diagnostic is specified after mixed image fits and during the frozen
camera-transfer run. It cannot become a new model-selection criterion.
No weights are changed and no VALIDATION/TEST target is fitted here.

Question: can a single material code reproduce the actual MSKCC TRAIN site
Lab references at all, or might decoder geometry itself obstruct learning?
Deduplicate the248TRAIN sites, asserting all copies have identical targets.
For each site solve8latent coefficients against its known nativeLab using
the frozen nonlinear material decoder. This is a target-informed oracle,
not prediction from images. Optimize rawLab squared residual with analytic
Jacobian, scipy least_squares, max_nfev400, ftol/xtol/gtol1e-10.

Report separate coefficient boxes +/-1,+/-3,+/-6 (in ISSA score-standard-
deviation units). Two starts: zero and the clipped minimum-norm tangent
solution. Choose smaller rawLab squared residual. Save convergence status,
evaluations, norm and achieved DeltaE00. Do not call a nonzero local residual
a certified infeasibility bound. A sufficiently small achieved residual proves
representability for that site under the assumed decoder, not recovery from
an image or physically correct latent identity. The actual image decoder also
mixes four hypotheses, so this single-code diagnostic is not its full gamut.

All initial values, boxes and solver tolerances are fixed before this diagnostic.
No new training/search follows automatically from whichever box looks best.
The purpose is to distinguish representational failure from inverse-learning
failure, not to turn a label-fitted result into an accuracy claim.
