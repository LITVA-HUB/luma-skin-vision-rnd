# Frozen real skin allocation / source camera-transfer screen

Original MSKCC TRAIN and reused source VALIDATION only; actual native instrument
Lab / DeltaE00. No CALIBRATION/TEST. Prior internal sampling screen motivated a
combination; it does not provide independent evidence. This wider source screen
is also exploratory because VALIDATION was extensively reused in earlier work.

Six samplers: image, person_site, site, color, color_ipw and person_color. First
five exactly preserve the frozen skin_color_sampling.py definitions. person_color
starts from the color photo proposal, normalizes within each person's photos and
assigns every person mass1/P. Thus it preserves within-person site-color ratios
but equalizes people. TRAIN native Lab/person/site labels only; no camera labels
in weights or inference input. Density bandwidth5, inverse-density caps1/3..3,
uniform mixture0.5 remain frozen. No hyperparameter tuning on transfer results.

Three TRAIN protocols, seeds17/29/43:54 fits.
- mixed: all24 TRAIN people/966images; known evaluation all6 VALIDATION/264images.
- from_SLR:8 TRAIN people/323images; known3SLR validation people/132images and
  separately unseen3iPod validation people/132images.
- from_ipod:16 TRAIN people/643images; known3iPod validation people/132images and
  separately unseen3SLR validation people/132images.

No camera from the unseen set enters training, scaling, sample-density estimates
or model selection. People are disjoint across TRAIN/VALIDATION. Camera and person
composition remain confounded, so these are source acquisition-transfer protocols,
not isolated causal sensor effects or ordinary smartphone proof.

Same original SkinRepresentation('baseline') / CaptureColor mixture929,297params.
Exactly30 rounds x31 batches x32 draws =930 optimizer updates for every fit,
irrespective of TRAIN population size. AdamW lr.001 wd.01; cosine after each
round to.00001. Selected TRAIN photo target mean/std, standardized Lab MSE plus
modeCE0.1. Mode is training auxiliary only. color_ipw uses the bounded per-image
importance correction for the entire loss; others are unweighted. FP32
deterministic RTX4060. No augmentation, external weights or image synthesis.

Primary fixed final checkpoint; no early stopping or best-epoch selection.
Validation is evaluated after all930updates, never used for fitting/sampling.
Record TRAIN loss history, final state/draw/proposal hashes, counts and resources.
Evaluate90 known/unseen color arrays with native CIEDE2000 mean/median/p90/p95,
image/patient/site means, fractions>5/>10, all fixed100/95/90/80/70/60% coverages
and full curves. Model-independent ranking uses selected TRAIN mean patch stats
with TRAIN std floor1e-6 and nearest Euclidean distance. Same accept sets within
each training protocol/evaluation domain; not calibrated C+ or guaranteed refusal.

Hypothesis: combining uniform person mass with within-person color support gives
a more stable skin-error tradeoff under unseen acquisition. Failure: it amplifies
reference noise or population shortcuts and fails on transfer. Do not select
camera-specific winning arms post hoc as a universal model. Compare every seed
and strongest locally reproduced control under the exact same930-step budget.
Older80-epoch numbers remain separately labeled, never presented as matched.
Balancing/importance sampling are established; no innovation claim from the
combination alone. Original data CC-BY; no new data/weights/publishing.
