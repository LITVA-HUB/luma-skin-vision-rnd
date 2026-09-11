# ISSA measured skin material controls

**Input is the full measured spectrum. These are oracle compression controls,
not photograph/RGB skin-color accuracy. No proposed innovation is established.**

Original ISSA v4 (CC BY4.0), 15,256 records. Local file contains 2,107 subject
labels, six fewer than the article-described population. Three labels have
conflicting gender codes and one conflicting age code. Missing age/gender
cells are preserved. Disjoint codes are not independent identity verification.

Fitting: 8,680 records / 1,232 labels. Validation: 1,822 / 262. Roles were
frozen before participant spectra/color values were inspected. Source origins
9-11 and 1,917 known-source test records remain reserved and numerically unread.
See [metadata](metadata.json), [split lock](split_lock.json), and
[pre-validation fit lock](material_lock.json).

All 31 common measured bands (400-700nm) are used for spectral error. Native
DeltaE00 is evaluated only on 1,156 validation records whose original support
is exactly 400-700nm, using the supplied Lab and the workbook-specific white.
Other records use 360-740nm in their original XYZ calculation. Dropping their
extra measured bands would itself change color. These are neither full-visible
integrations nor MSKCC D65/10-degree targets. No missing wavelengths are filled.

Each subject has equal total weight when fitting a basis. All configurations
were fixed before validation and are reported; no selected winner is promoted.

| Representation | Latent width | Mean DeltaE00 | Median | p95 | Worst source mean | Mean spectral relative L2 |
|---|---:|---:|---:|---:|---:|---:|
| mean | 0 | 7.0840 | 5.7777 | 17.5594 | 15.4591 | 25.237% |
| reflectance | 2 | 2.3959 | 2.1301 | 5.5471 | 3.5630 | 3.182% |
| reflectance | 3 | 0.4495 | 0.3602 | 1.0564 | 0.6944 | 1.765% |
| reflectance | 4 | 0.4207 | 0.3308 | 1.0115 | 0.7692 | 1.154% |
| reflectance | 6 | 0.2039 | 0.1374 | 0.6062 | 0.3478 | 0.539% |
| reflectance | 8 | 0.0567 | 0.0490 | 0.1332 | 0.0675 | 0.351% |
| density | 2 | 2.3533 | 1.9902 | 5.5246 | 4.5389 | 4.433% |
| density | 3 | 0.6834 | 0.5786 | 1.5333 | 0.8750 | 2.018% |
| density | 4 | 0.3810 | 0.3024 | 1.0357 | 0.5315 | 1.161% |
| density | 6 | 0.2063 | 0.1500 | 0.5790 | 0.2775 | 0.556% |
| density | 8 | 0.0236 | 0.0170 | 0.0611 | 0.0706 | 0.409% |
| logit | 2 | 2.5323 | 2.2303 | 5.7065 | 4.3523 | 3.631% |
| logit | 3 | 0.6224 | 0.5431 | 1.3795 | 0.8551 | 1.921% |
| logit | 4 | 0.3893 | 0.3125 | 0.9624 | 0.5778 | 1.112% |
| logit | 6 | 0.2045 | 0.1478 | 0.5725 | 0.2885 | 0.536% |
| logit | 8 | 0.0259 | 0.0192 | 0.0675 | 0.0593 | 0.395% |

## Interpretation

The tested two-dimensional bases lose substantial color variation (mean about
2.35-2.53, p95 about 5.52-5.71 DeltaE00), even with the full spectrum supplied.
This does not rule out nonlinear two-variable optical models. Three linear components
reach mean 0.4495 / p95 1.0564. Eight optical-density components reach mean
0.0236 / p95 0.0611, but this extra fidelity does not prove those coefficients
can be inferred from an ordinary photo. Reflectance is better than log/density
at width three; the opposite ordering at width eight is not a universal
advantage of a physically named transform.

The 8-component decoder has 279 stored coefficients; it is not a complete
camera model. All fitted basis states occupy 24,634 bytes together. CPU
fitting of the three bases took 0.0156 seconds on this run. GPU training,
inference VRAM and camera-model latency were not measured by this screen.

## Verification

18,496 independent scalar color-error cases; maximum metric gap 6.11e-15.
Independent weighted SVD versus covariance eigensolver reconstruction gap 1.32e-14.
Validation spectra exactly matching TRAIN at all 31 bands: 17.
TRAIN spectrum fingerprints shared by different subject labels: 24.
These exact-duplicate checks do not prove absence of near duplicates.

## Exact-spectrum overlap sensitivity (post hoc)

Seventeen validation spectra match TRAIN exactly despite different subject
labels. Removing every validation label connected to TRAIN through exact
spectra (including transitive links) excludes two labels / 17 records, all
in the 360-740nm group. Consequently none of the 1,156 primary color
records is removed and every DeltaE00 value above is unchanged. This is
a descriptive sensitivity check, not proof that all remaining identities
are independent. The frozen split and original results remain preserved.

| Representation | Width | Mean spectral relative L2 without linked labels |
|---|---:|---:|
| mean | 0 | 25.257% |
| reflectance | 2 | 3.184% |
| reflectance | 3 | 1.761% |
| reflectance | 4 | 1.150% |
| reflectance | 6 | 0.535% |
| reflectance | 8 | 0.350% |
| density | 2 | 4.429% |
| density | 3 | 2.016% |
| density | 4 | 1.157% |
| density | 6 | 0.553% |
| density | 8 | 0.406% |
| logit | 2 | 3.629% |
| logit | 3 | 1.919% |
| logit | 4 | 1.108% |
| logit | 6 | 0.532% |
| logit | 8 | 0.392% |

All TRAIN/VALIDATION native XYZ/Lab values replay numerically. The source
contains an unrelated hue-formula reference outside the data table; hue is
not used. The original workbook was not modified.

## Next decision

A material basis preserves color in this exploratory check, with the
identity limitations above. Next test
whether it reduces actual image-to-instrument Lab error under a correctly
matched observer/support convention, against ordinary matched bottlenecks.
Alternatively use its feasible color set to reject ambiguous inputs rather
than forcing one plausible skin color. Physical skin manifolds and spectral
PCA are existing ideas; the optical skin-color locus is also addressed by
[2026 prior art](https://pmc.ncbi.nlm.nih.gov/articles/PMC13307969/).

Preserved independent MSKCC result remains primary mean 4.4570 / selective
80% mean 4.1591; stronger ordinary fusion 4.3005 / 4.1447. These ISSA oracle
numbers must never replace that result in a camera-accuracy claim.
