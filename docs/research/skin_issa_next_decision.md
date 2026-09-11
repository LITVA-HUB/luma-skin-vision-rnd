# Decision after measured skin material controls

## What changed the next action

Real skin material compresses well. On the fixed 1,156-record native-color validation subset, the
three 2D representations have mean DeltaE00 2.3533-2.5323 and p95 5.5246-5.7065.
Those particular projections lose color even with full-spectrum input. This
does not refute nonlinear two-variable optical models or every possible 2D
representation. Such a constraint still needs its own color-preservation test;
the PCA experiments do not justify a general dimensionality lower bound.
The eight-dimensional optical-density oracle reaches mean 0.02365 / p95 0.06113.
The three-dimensional ordinary linear basis reaches mean 0.44949 / p95 1.05644.
Neither result measures inference of these coefficients from a photo.

The tempting inference "0.024 means we solved skin color" is false. The input
already contains substantially more physical information than the desired RGB
input. Keep the independent MSKCC photo result (4.457 / 4.159 at80%) as primary.
Ordinary fusion remains stronger (4.3005 / 4.1447). No new photo model promoted.

## Data limitations that survive numerical verification

- Original ISSA v4 is CC BY4.0. Data source rights are separate from claims.
- 15,256 records locally verified, 2,107 subject labels versus 2,113 reported.
- Seventeen VALIDATION spectra exactly match TRAIN despite different labels.
  Two linked validation labels contain all those records. They are outside the
  1,156-record color subset; removing them leaves every reported color score
  unchanged. Near duplicates and underlying participant identities remain
  unresolved. This is an exploratory source validation, not clean independent
  person confirmation. Never silently repair the source or overwrite the split.
- Native XYZ uses per-record spectral support; native Lab shares a 400-700nm
  white. We reproduced it, but this is not the MSKCC D65/10-degree convention.
- Three source cohorts and known-source TEST remain reserved and unread.

## Next bounded research route

1. Resolve the observer/spectral-support interface before inserting any ISSA
   coordinates into the MSKCC predictor. A CIE1931/2-degree color must not be
   treated as an instrument D65/10-degree label. Common-band approximations
   require explicit error analysis; missing reflectance is never measured zero.
2. Compare a flexible measured-material decoder against ordinary matched
   bottleneck heads on the existing real-image source development cohorts.
   Those cohorts are already heavily explored; any gain there is provisional.
   Keep an unconstrained color residual or refusal path until real targets
   demonstrate that hard restriction does not remove valid skin variation.
3. Allocate a competing branch to predicting a feasible color set and its
   downstream error, rather than selecting one plausible material. Mechanism:
   multiple material/photometric explanations that fit the same image should
   cause refusal when they imply materially different reference skin colors.
   Failure: an over-narrow assumed camera/lighting family gives false certainty.
   Cheapest test: fixed controls with known ambiguous forward projections,
   then actual image-to-instrument residual prediction versus matched C+.
   Derived projections can falsify a mechanism, not establish phone accuracy.

The proposed competing branch challenges the assumption that the answer must
always be a point estimate; it has not yet been validated experimentally.
It does not claim that inverse rendering, spectral PCA, physical skin models,
Bayesian ambiguity or selective prediction are novel. The contribution, if
any, must be an actual accuracy/coverage/compute advantage on real color labels.

No foundation model has earned deployment cost in our source screens. Preserve
the 108 global-teacher and36 local-teacher negative fits. No further generic
teacher sweep or export work is justified by these material results alone.
