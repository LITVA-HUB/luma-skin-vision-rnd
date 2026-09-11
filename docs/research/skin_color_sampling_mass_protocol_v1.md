# Post-screen person-mass confounding falsifier

The original15 sampling fits are terminal and scalar/probability audits pass.
Color allocation mean6.0012 is almost tied with person/site balancing6.0084.
Before interpreting a color-specific gain, freeze two additional controls on
the exact same18 TRAIN/six-person internal holdout and930-step recipe. Seeds
17/29/43; six new fits. Retain original color models as unchanged comparators.

Both controls start from the original selected-TRAIN color site proposal. They
preserve its exact total probability for every TRAIN person:
- person_mass: redistribute that person's probability uniformly across their
  sites, then uniformly across each site's photos.
- within_person_shuffle: permute original site probabilities among sites of
  the same person using RNG(seed+314159), sorted person/site traversal. Preserve
  original site-mass multiset and each person's total mass, then uniform photos.

All training losses remain unweighted native standardized Lab MSE+0.1modeCE.
No camera labels in sampling, model input or altered loss. Same constants, scales,
fixed final checkpoint, no model selection. No new data/weights or endpoint roles.
This is a post-screen exploratory falsifier, not prospective independent evidence.
The null is that the modest gain reflects person reweighting rather than a useful
association between within-person site-color density and training emphasis.
Failure to outperform these controls weakens a color-specific explanation; it
does not prove the effects identical. Score ordinary and patient-balanced errors,
tails and same input-novelty coverages. No new calibrated-risk or phone claim.
