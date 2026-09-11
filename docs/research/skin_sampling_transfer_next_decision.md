# Decision: allocation combination fails the universal-transfer hypothesis

54 source fits and90 known/unseen evaluations are complete on original MSKCC
photographs and actual instrument-native Lab. All fits use930 updates and the
same929,297-parameter core. Original TRAIN and reused source VALIDATION only;
no TEST/CAL access, checkpoint selection or transfer-informed hyperparameters.
[Full report](../benchmarks/skin_sampling_transfer_v1/report.md).

## Measured outcomes

| Evaluation | Combined person/color | Strongest matched control by mean |
|---|---:|---:|
| Mixed known cameras | 3.7435 | color3.7864 |
| SLR-trained, unseen iPod | 6.1545 | image5.4877 |
| iPod-trained, unseen SLR | 6.4074 | person/site6.0825 |

All values are mean skin DeltaE00 averaged across three separate runs. No
different per-camera winner is selected as a universal deployment model.
The mixed difference is small: -0.0428 versus color, descriptive patient interval
[-0.1667,0.0802]. In reverse transfer the combination loses to person/site in
all three seeds, with descriptive patient interval[0.1971,0.4194]. These reused
source cohorts are exploratory, not independently confirmatory.

At80% coverage the same three combined scores are3.7264 /5.6200 /6.7609.
Reverse selection worsens error relative to accepting every image. The common
input-novelty ranking is therefore not a reliable color-risk estimator. It must
not be presented as calibrated refusal or matched C+.

Known-camera results for the single-camera combined models are3.4478(SLR) and
4.0492(iPod), with sharply higher errors on the other acquisition. Camera and
people/capture composition remain confounded: the experiment does not isolate
a causal camera effect. Ordinary facial smartphone accuracy is still unmeasured.

## What is rejected and retained

Reject the claim that combining these two balancing mechanisms yields robust
camera-independent skin-color accuracy. Preserve the small mixed-source gain as
an optimization observation. Do not expand the same sampler sweep or increase
architecture size merely to regain that narrow result. Prior80-epoch source
references3.4406 /4.8328 /4.9736 also remain stronger under different schedules;
they are historical, not matched930-step baselines.

The assumption that covering the marginal distribution of measured skin colors
is sufficient for transfer has failed this screen. Training image appearance
conditional on true color can change with acquisition; balancing targets does
not by itself identify that conditional mapping or make uncertainty reliable.
This is an interpretation of the observed negative result, not a proof that no
camera-blind model can succeed.

## Next cheap diagnostic before choosing a new mechanism

Determine whether the transfer residual resembles a shared native-Lab offset or
requires image-dependent correction. Freeze a diagnostic on the current source
predictions: for each evaluation person, estimate a three-channel offset using
only the OTHER evaluation people's measured references. Average residuals by
site, then by person. Evaluate two fixed offset strengths0.5 and1.0, retain both,
no chosen winner. Compare known and unseen domains. Never include a person's
reference in the offset applied to that person's images.

This is a PRIVILEGED REFERENCE-CALIBRATION COMPARATOR, not our single-image,
no-calibration method. Its labels come from existing public source data; no new
proprietary acquisition is requested. It is a diagnostic for deciding which
error mechanism to investigate, not a deployable gain, fresh test result or
claim of camera independence. Keep its metrics separate from original model
scores and preserve all uncorrected predictions.

If offsets explain little, deprioritize global color-bias fixes and investigate
image-dependent reliability/representation. If they explain much, investigate
what legitimate single-image signal could infer such correction without those
references, with a falsifier before implementing another large model. Either
outcome is informative; neither establishes physical identifiability or accuracy
on arbitrary phone images. Offset calibration itself is established methodology.

## Integrity and product scope

90 exact prediction arrays; six complete fixed-budget refits;14256 scalar color
cases;476490 scalar TRAIN density pairs;540 coverage rows;144 hybrid mass/ratio
identities.359 tests pass,14 historical warnings in33.75s. Training peak106.23MiB
measures the feature-based fit, not image preparation or end-to-end inference.
No new isolated latency/export claim.

Independent MSKCC remains primary4.4570/80%4.1591 vs ordinary fusion4.3005/4.1447.
No new strongest independent-baseline win, universal camera model, calibrated
refusal, cosmetics-matching validation or facial-phone proof. Original MSKCC
CC-BY; no new outside data/weights or publishing. Goal active and unmet.
