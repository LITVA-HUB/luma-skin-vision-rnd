# Measured-skin spectral representation: source mechanism screen

**No new real-photo accuracy result or novel-method victory.** This probe tests
a representation needed by a possible physical inverse model. It supplements
the frozen MSKCC result; it does not replace that benchmark with derived images.

Original licensed UMINHO-HSFD TRAIN data: three faces, nine manually inspected
50x50skin regions,22,500measured33-band spectra. Total acquired cubes236,848,136B;
only159,933,841B added beyond the prior pilot. Original size/MD5 and localSHA256
checked. No held-out UMINHO cubes or exposed MSKCC endpoints were opened.
All selected samples finite and positive, with no values above1 and no clipping.
The mask is a source-only manual selection, not externally validated segmentation.

## A compact material representation is plausible, but three dimensions are weak

Each column holds out one TRAIN face and fits on the other two. Entries are
mean relative spectralL2 error, percent. **All methods see the full33-band query
spectrum to obtain their coefficients or neighbor: oracle representation, NOT
single-RGB inference.** Pixels are correlated and there are only three faces.

| Representation | Source fold0 | Source fold1 | Source fold2 |
|---|---:|---:|---:|
| Mean training spectrum |32.56|131.42|23.08|
| Nearest training spectrum |4.68|69.07|3.87|
| Linear PCA,3dimensions |4.06|9.00|3.64|
| Log PCA,3dimensions |3.93|9.92|4.08|
| Linear PCA,8dimensions |2.99|3.41|2.16|
| Log PCA,8dimensions |3.65|6.14|3.33|

The8-dimensional linear representation improves all three folds over its
3-dimensional version. A mean plus8basis vectors needs297scalar values before
any inference network. This is only representational storage, not model size.
Nearest measured material can fail badly outside source tone coverage. These
results justify a broader material family, not projecting everyone onto a
typical skin tone. Spectral compression error is not instrument DeltaE00.

## Illumination ambiguity: suggestive, not a demonstrated exact collision

Enumerated all36regional pairs and four polynomial log-illuminant degrees:
144fits. Among27cross-face pairs, **zero** have less than1%radiance relativeL2
mismatch at any of degrees0/1/2/3. Thus the strict near-identity hypothesis is
NOT supported in this declared low-degree spectral family.

An exploratory quadratic example yields2.185%radiance mismatch despite a
mean-reflectance ratio0.3602; required illuminant gains0.2195–0.5345. This is a
mathematical positive smooth-SPD stress, not a measured lamp or phone experiment.
No full-visible color difference, ordinary RGB collision, or universal
impossibility theorem follows. Log fitting also does not minimize the reported
linear radianceL2 score, so increasing degree need not improve that score.

## Attack the regional conclusion using shared spatial evidence

A declared POST-SCREEN follow-up fits all three region means jointly for each
face pair: shared illuminant, shared spectral shape plus local exposure, or
independent per-region illuminants.36fits retained, including degrees0–3.
Quadratic pooled radiance mismatch, percent:

| Source pair | Shared illuminant | Shared shape + local exposure | Independent regions |
|---|---:|---:|---:|
|0/1|12.95|9.17|7.34|
|0/2|10.20|5.49|4.33|
|1/2|17.00|5.92|4.76|

Joint constraints leave more evidence distinguishing faces in these examples.
Allowing local exposure explains much of that difference. This motivates
testing shared capture variables with local shading, but does not show that
RGB preserves the distinguishing spectral evidence. These manually selected
forehead/cheek regions are not registered corresponding surface points.

## Verification and decision

40saved arrays exactly replayed;270,000independent scalar spectrum cases,
30independent covariance-eigensystem projections,18brute-force neighbor checks,
144normal-equation illumination checks. Maximum metric disagreement1.12e-16;
projection1.37e-13;illumination4.89e-15.273tests passed,14historical ONNX warnings.
All36spatial follow-up fits also checked against normal equations and replayed.

Next: attack the current real-photo model's exchangeable-patch assumption.
Test spatial relations and a shared capture latent while retaining absolute
color, against parameter-matched ordinary spatial and permutation-invariant
controls. Keep the proposed solver optional and test zero/one/multiple steps.
Spectral coefficients do not become MSKCC nativeLab targets: observer, wavelength
support, sensor response and measurement geometry differ. Do not inject an
unverified spectral-to-nativeLab mapping into training.

[Frozen protocol](../../research/skin_spectral_probe_protocol_v1.md),
[all results](summary.json),[independent audit](audit.json),
[spatial follow-up](spatial_followup.json),
[original data and limitations](../../data/uminho_hsfd_verified_inventory.md).
