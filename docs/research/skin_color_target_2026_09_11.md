# Product target: instrument-referenced facial skin color

Latest user steering explicitly prioritizes actual skin-color accuracy. Angular
color-constancy results are component diagnostics and cannot satisfy the Luma
product acceptance criterion. Preserve their evidence, including V7's loss, but
do not spend the next major training cycle optimizing only that proxy.

The product target is regional skin color under a specified reference viewing
condition, with prediction compared against corresponding instrument readings.
Measure CIEDE2000 median, p90/p95, catastrophic errors and error at fixed accepted
coverage, plus repeatability across captures, lighting and camera models. Keep
the same people out of all fitting roles and the final test. Repeatability alone
cannot prove accuracy: consistently wrong predictions are still wrong.

Working engineering aspiration, not an established cosmetics standard or a
measured result: median DeltaE00 <= 2 and p95 <= 5 among at least 80% accepted
valid captures, on independently held-out people and supported phone models.
All-image errors, exclusions and achieved coverage must also be shown. Set the
final product tolerance using intended shade decisions and measurement noise;
do not promise a clinically or perceptually universal threshold.

Current standalone R&D skin-image accuracy is **MEASURED ON MSKCC HELD-OUT
PEOPLE**: primary mean4.457 DeltaE00,80%-accepted median3.883/p958.097. The
above working target is not met. [Independent report](../benchmarks/skin_mskcc_selective_v1/report.md).
Accuracy of the actual Luma product on ordinary facial-phone images remains
**NOT MEASURED**. Both MSKCC camera families were seen during development.
No angular-error-to-DeltaE conversion is valid for the present illuminant-only
benchmarks. Existing sRGB-to-Lab conversion code is not a measurement reference.

## Historical public-route exploration (before the MSKCC image milestone)

He et al.'s original [Zenodo deposit](https://zenodo.org/records/5532176) is
CC BY 4.0. Its 395,693-byte workbook is acquired and publisher-checksum verified.
It supplies measured XYZ and paired RAW/JPG regional RGB for 200 facial sites
from 40 training people and 100 sites from 20 different testing people. There
are also two measured training charts. Full face images are not in this deposit.
Use this for a bounded RGB-to-measured-XYZ calibration pilot, not to train or
validate an image normalization network or assert unseen-camera accuracy.

The [paper](https://doi.org/10.1002/col.22737) specifies Canon 6D Mark II,
controlled polarized illumination and CM-2600d SCI/SAV reference measurement.
Its XYZ references use the measured LED spectrum. A numeric reference white
or that exact spectrum has not been verified in the deposit. CCT alone is
insufficient to silently substitute standard D65.

A related [spectral deposit](https://zenodo.org/records/5730472), also CC BY 4.0,
was acquired to check whether its calibration data resolved this gap. The
chart RGB values differ, and its spectrum/reflectance reconstructions fail
to reproduce the first workbook's XYZ. The correspondence check is negative;
do not use that spectrum as the skin experiment's reference white.

Immediate authorized work: establish a locked, author-split XYZ baseline and
retain predictions for future valid perceptual scoring. Genuine DeltaE00 remains
unavailable until the exact white-reference convention is established. Search
other original instrument-paired releases in parallel with this bounded pilot.

ENCoDE has iPhone/Pixel images paired with skin instruments, including forehead
data, but [PhysioNet](https://physionet.org/content/encode-skin-color/1.0.0/)
requires credentialing, training and a DUA. No access, agreement, identity claim
or author contact has been made. DAST and CHROMA-FIT are instrument-paired face
candidates, but original access/usage terms require further verification.

No proprietary dataset collection, paid resources or external publication is
introduced. This steering changes the next research priority, not the recorded
meaning of prior results.
