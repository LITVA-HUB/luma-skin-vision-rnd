# ISSA v4: original rights and acquisition verified 2026-09-11

Original release: [Figshare dataset](https://doi.org/10.6084/m9.figshare.28228571.v4).
Authors: Yan Lu, Kaida Xiao, Michael Pointer and collaborators listed in the
archived original metadata. Dataset title: The International Skin Spectra
Archive (ISSA): a multicultural human skin phenotype and colour spectra collection.

**CLEARED FOR R&D AND COMMERCIAL MODEL TRAINING** under the original dataset's
[CC BY 4.0 grant](https://creativecommons.org/licenses/by/4.0/): retain attribution,
license link and indication of modifications. This updates the earlier
LICENSE UNCLEAR inventory entry after checking the original release and API;
it does not rely on a re-upload's license or on the article's separate license.

| Item | Verified status |
|---|---|
| Original release version | 4; DOI 10.6084/m9.figshare.28228571.v4 |
| Original metadata API | https://api.figshare.com/v2/articles/28228571/versions/4 |
| Author-described population | 15,256 records, 2,113 subjects, 11 source datasets |
| Locally verified population | 15,256 records, 2,107 subject codes; six-code discrepancy unresolved |
| Ground truth | Measured skin reflectance and supplied colorimetric fields |
| Declared color convention | D65 / CIE1931 observer; differs from MSKCC native D65/10-degree |
| Camera photographs | No paired everyday-camera image release established here |
| File | ISSA_17_Jan_2025_Yan_Lu.xlsx |
| Download | 9,320,520 bytes, acquired from original Figshare file52744241 |
| Original MD5 | 67e15398f96dd9af08b5d209d2522e19; supplied/computed/local all agree |
| Local SHA256 | 7396aa7608f1a6f91ec70bf8f8421252a9d81f71a2bca066eb7e98155b9d1161 |
| Numeric endpoint inspection | TRAIN8,680 / VALIDATION1,822 audited; TEST and reserved origins unread |
| RTX4060 suitability | Small tabular source; proposed physical-prior work feasible in storage; training not measured |

The [original paper](https://www.nature.com/articles/s41597-025-04857-5)
describes varying spectral support across constituent datasets. Do not treat
all records as full360-780nm measurements merely because those spreadsheet
columns exist. Missing wavelengths must not silently become measured zeros.
Instrument, specular mode, body location, actual support, subject identity and
calculated versus directly supplied color fields require a schema audit first.

Next: inspect coding scheme and identity metadata, freeze person/source roles
before fitting or evaluating spectral models, then inspect appropriate TRAIN
endpoints and reconstruct the supplied colorimetry convention. Do not mix
MSKCC and ISSA Lab targets without resolving observer/geometry differences.
Any camera rendering will be derived input, not a photograph or independently
measured camera benchmark. This acquisition proves no skin-photo accuracy.

[Original metadata](provenance/issa_v4/original_article_metadata.json),
[acquisition and checksums](provenance/issa_v4/acquisition.json).

## Completed material audit

The workbook stores percent reflectance. Native XYZ uses each record's
360-740nm or400-700nm support; all native Lab uses the400-700nm white.
Both replay numerically. No missing tails are filled. Three gender-code and
one age-code conflicts,2,716 missing age cells and nine missing gender cells
are retained.17validation spectra match TRAIN under different subject codes;
a two-code exclusion sensitivity is documented. Counts describe codes, not
independently verified people. [Results and limits](../benchmarks/skin_issa_v1/report.md).

The acquisition-stage next-step paragraph above is retained as history.
Current next action is [observer-safe image integration](../research/skin_issa_next_decision.md).
