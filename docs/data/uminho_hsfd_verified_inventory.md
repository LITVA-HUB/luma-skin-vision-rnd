# UMINHO-HSFD original terms and initial source-cube audit

LATEST EXTENSION: three smallest preassigned TRAIN cubes now verified/acquired,
236,848,136bytes total. Nine manually inspected50x50skin regions provide22,500
measured spectra, with no clipping or excluded numerical outliers. Annotations,
face metadata and author rendering source were acquired under their original
CC BY4.0 item terms; source MATLAB was inspected but not executed/adopted.
The original paper describes29participants/29faces; original face metadata has
one row per face. This supports the intended face/person unit, but no independent
identity audit is claimed. VALIDATION/TEST remain unacquired and unexamined.
No independently validated automated mask or new phone data was obtained.
[Extension receipt](provenance/uminho_hsfd_v1/train_expand_v1.json),
[annotation receipt](provenance/uminho_hsfd_v1/support_v2.json),
[mechanism results](../benchmarks/skin_spectral_probe_v1/report.md).
The following initial-acquisition details are retained as historical evidence.

Status: **CLEARED FOR R&D AND COMMERCIAL MODEL TRAINING** under original
CC BY4.0, subject to attribution and the license terms. Verified on2026-09-11
in every original Figshare item AND the author README; no re-upload license.
[Original collection](https://doi.org/10.6084/m9.figshare.c.7163569),
[reflectance item/API](https://api.figshare.com/v2/articles/25598670),
[README item](https://api.figshare.com/v2/articles/25638546),
[license](https://creativecommons.org/licenses/by/4.0/).
Attribution: Gomes, Andreia E.; Linhares, Joao M. M.; Nascimento, Sergio M. C.
(2024), University of Minho Hyperspectral Faces Database: UMINHO-HSFD.

## What is actually measured

29face hyperspectral cubes,33bands spanning400–720nm at10nm intervals. Original
MAT files contain per-pixel reflectance. Hamamatsu C4742-95-12ER monochrome CCD,
12bit,1344x1024sensor, tunable liquid-crystal spectral filter; controlled metal
halide illumination and fixed2.27m capture distance. Head supported, eyes/lips
closed, clean skin, limited facial hair. This is one specialist acquisition
system, not a multi-phone RGB benchmark. Spectral filter bandwidth varies with
wavelength (author setup7nm/10nm/16nm at400/550/720nm); not ideal delta sampling.

Authors' setup sheet reports average2%spectral difference and1.3CIEDE2000
colorimetric error; these are **PUBLISHED BY AUTHORS**, not independently
reproduced here and not a proven error floor for our future estimator. Its
illumination line uses cd/m2 while labelled illuminance; preserve this source
ambiguity rather than silently converting it to lux.

The supplied29RGB JPEGs are RENDERED from the same spectral measurements, with
D65 and the CIE2006 10degree cone-fundamental-based observer, and are redacted
for identity protection. They are not independent camera captures or a second
measurement reference. No RGB files were downloaded or used. Do not pool these
derived inputs with real MSKCC photographs, claim ordinary phone accuracy, or
treat their observer convention as identical to another instrument without
verification. The finite400–720nm band range also limits full-visible integration.

## Size, access and initial acquisition

All29reflectance files total3,035,028,803bytes (about3.04GB). Author RGB files total
3,100,416bytes and are unnecessary for the current loader probe. Acquired only
the smallest file in a preassigned TRAIN subset:76,914,295bytes, plus179,361bytes
of README/setup/file-format documents. Original size and Figshare MD5 verified;
local SHA256 recorded. No payment, contact, publishing or hundreds-of-GB download.
One-at-a-time cube loading is practical for local RTX4060/RAM experiments.

Before cube download/decoding,29face-file IDs were ordered by
SHA256("LumaUMINHOv1|original filename"):first5TEST,next5VALIDATION,remaining19TRAIN.
Manifest SHA256 `3d3b7dc7256eb04250dee2f5ef76880f4101d019313d97ba051480f8d320b5c7`.
This is a face-file allocation; verify distinct participant identity in the
original documentation before asserting subject-disjoint benchmark claims.
Only one TRAIN cube has been decoded. Validation/test cubes remain unacquired.

## Observed source-cube quality

Original variable `datao`, float64,696x587x33, matching author file metadata.
All values finite;325,925nonzero foreground pixels and82,627all-zero background
pixels. Authors explicitly identify zero background as artificial. Foreground
is not yet a verified skin-only mask: lips/eyes/features can remain.

Nonzero reflectance range0.013516to1.656773, no negatives,1,055channel values
above1. Preserve originals; do not silently clip. Directional reflection,
measurement normalization, highlights and artifacts need investigation before
assuming a bounded diffuse-material prior. No color-model accuracy, skin mask,
RGB rendering or spectral reconstruction result is claimed by this loader audit.

[Acquisition receipt](provenance/uminho_hsfd_v1/acquisition.json),
[numeric TRAIN-cube audit](provenance/uminho_hsfd_v1/train_cube_audit.json).
Original documents and participant arrays remain local and unmodified.

Next permitted work: verify masks/reference conventions and a measured-spectral
material representation on TRAIN faces. Rendered-camera experiments, if used,
must be labelled derived-input experiments and validated separately on real
camera photos. This route supplements the existing real skin benchmark; it
does not replace it with another synthetic-only milestone.
