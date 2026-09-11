# Source-only skin reference and capture correspondence diagnostic

Frozen before diagnostic execution. No new fitting, selection, image decoding,
or CAL/TEST access. Original source caches only. Native Lab references and
CIEDE2000; no RGB-to-Lab conversion or manufactured illumination-derived labels.

Deduplicate site references across images and require exactly identical three
readings, plus a unique patient per site. Fail on conflicts. Summarize all
three reading pairs, all readings against their mean, and each reading against
the other two readings' mean. Reading spread is not a proved irreducible floor.

Inspect existing frozen capture plain-MSE, capture mixture-MSE and training-only
graph-always predictions, all three seeds, mixed fitting and both source camera
transfer protocols. Bind all result/prediction files before metric computation.
Verify target order and recorded errors; independently evaluate every computed
CIEDE2000 value using the scalar implementation.

Compute image-, site-, and patient-balanced single-image error, errors against
each individual instrument reading, all-within-site capture disagreements, and
diagnostic error of mean native-Lab predictions across site captures. This last
diagnostic needs multiple photographs and is not a valid single-image model
result. No label-derived fusion coefficients are fitted.

Decompose Euclidean native-Lab squared error into shared site mean bias and
within-site prediction variance, both image- and site-weighted. Verify the
algebraic identity. It is not a CIEDE2000 decomposition or proof of causation.

Stratify by existing device/capture mode/image type and source TRAIN site-level
reference-L quartiles. These strata describe errors; they are not production
inputs. Camera cohorts have different people: transfer differences can reflect
people, capture and camera effects and must not be attributed only to sensors.

No patient, image or site identifiers are written into public aggregate output.
Preserve all 27 prediction endpoints and unfavorable findings. Use results to
choose a subsequent bounded experiment; do not tune a new independent claim
on previously exposed source validation or test results.
