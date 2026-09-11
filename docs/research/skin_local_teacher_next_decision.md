# Decision: local teacher correspondence is not the missing mechanism here

Completed 36 source fits, using unchanged strong RGB mixture, aligned teacher,
global teacher and within-image shuffled teacher descriptors. Ground truth is
actual MSKCC native instrument Lab. All models output one-image predictions.
The source validation cohorts have been repeatedly explored; no new independent
TEST/CAL or UMINHO held-out endpoints were accessed.

| Mean of three separate seed scores | Mixed DeltaE00 | SLR to iPod | iPod to SLR |
|---|---:|---:|---:|
| Exact historical mixture control | 3.4406 | 5.0193 | 5.9635 |
| Aligned local teacher | 4.1720 | 6.0116 | 7.8197 |
| Global teacher | 4.0992 | 5.9560 | 7.7831 |
| Shuffled local teacher | 4.1826 | 5.9191 | 7.9399 |
| Strong historical plain control | 3.4771 | 5.8301 | 5.5089 |

Aligned descriptors do not beat global descriptors in any protocol mean.
Small aligned-versus-shuffled differences are not a robust mechanism win;
all teacher arms lose badly to strong RGB-only controls. At80% coverage,
aligned error is4.1590 /6.0805 /7.9010: transfer rejection worsens the means.
Do not promote, distill or optimize this teacher-assisted architecture.

The trained teacher-assisted head has1,011,601parameters, but inference also
requires22,056,576teacher parameters: total23,068,177. Plain uses929,297active
parameters, although its checkpoint retains the unused adapter. Cached teacher
features do not turn the experiment into a1M-parameter deployed model. Maximum
head-fit allocated CUDA memory232.62MiB excludes the offline teacher extraction
stage and CPU caches. No new latency/export claim is made.

## Opponent check: was capture-mode supervision the culprit?

Teacher variants improve capture-mode recognition despite harming color.
A post-hoc TRAIN-only diagnostic checked48batches from12selected mixed models,
288group/batch gradient records, without changing any weights. All-parameter
cosine between color and weighted mode gradients averages -0.0630 plain,
-0.0026 aligned, +0.0050 global and -0.0067 shuffled. Weighted auxiliary/color
norm ratios average0.1104 /0.2285 /0.3135 /0.2400 respectively.

Opposing directions occur in roughly half the batches, including the plain
control. This does NOT establish that mode supervision caused the teacher's
generalization failure, or that removing it would help. Reject the tempting
causal story until an actual intervention supports it. No coefficient search
or another large teacher sweep is justified by this diagnostic alone.

## Verification and the preserved precision amendment

All1,230local8x8x384feature grids exactly replay, and2,048pooled cells were
independently recomputed inFP64. An initial1e-6absolute verifier tolerance failed
at a16.05685feature value (gap1.66893e-6). Before ANY head fitting, only that
verifier was changed to the standardfour-termFP32summation bound. Original
script/receipt/failure diagnosis are archived; all cache bytes are unchanged.
The auditor compares ASTs with only the manual-check branch removed and verifies
receipt changes are confined to that script binding.

All108color arrays replay exactly, with36gate/hypothesis sets,6,336independent
scalar DeltaE00 cases and216fixed-coverage cases. Nine RGB-control final base
weight sets exactly reproduce the historical mixture. Fit-only teacher/target
normalization and same-camera epoch selection pass.307tests pass, with14
historical warnings in32.99s. All jobs are terminal.

## Change the source of prior knowledge: measured skin material

Generic visual teacher descriptors, pooled and local, have now failed the
strongest-control comparison. Next prioritize a physical material prior from
real measured skin spectra, rather than another generic semantic adapter.
Original ISSA v4 rights are now verified asCC BY4.0 and the9.32MBworkbook was
downloaded with matching originalMD5 and localSHA256. Author-described15,256
records/2,113subjects greatly expand the possible measured-material source.
Numeric endpoints have not yet been decoded or used.

First perform a coding/identity/instrument/support audit, then freeze source
roles before learning a spectral prior. The key mechanism hypothesis is that
measured admissible skin reflectances constrain color reconstruction more
usefully than generic image semantics. Its failure mode is non-identifiability
under unknown illumination/sensor response, missing spectral support, or
incompatible measurement geometry. The cheapest falsifier is a TRAIN-only
colorimetry/support and ambiguity audit, not immediate RGB accuracy claims.

Do not equate spectra with paired camera images or manufacture DeltaE00 from
missing wavelengths. ISSA D65/CIE1931 color fields differ from MSKCC's native
D65/10-degree convention. Rendering measured spectra would produce derived
inputs; real-photo accuracy must still be evaluated against genuine photo
references. Earlier UMINHO spectral oracles remain useful but are not RGB wins.

The primary independent MSKCC result remains4.4570full /4.1591at80%; ordinary
fusion4.3005 /4.1447remains stronger. No novel-model superiority or ordinary
unseen-phone facial accuracy is established. Goal active and unmet.

[Full local-teacher results](../benchmarks/skin_local_teacher_v1/report.md),
[gradient diagnostic](../benchmarks/skin_local_teacher_v1/gradient_diagnostic/results.json),
[ISSA rights and scope](../data/issa_v4_verified_inventory.md).
