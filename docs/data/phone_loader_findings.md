# Phone loader development findings, 2026-09-11

Measured only on the three preselected official TRAIN scenes: outdoor/443,
outdoor/390 and outdoor/337, both Samsung and Oppo. Six NT input files and six
WT reference files inspected; no reserved test pixels or GT decoded.
[Machine-readable audit](provenance/mobile_screen_2026_09_11/loader_audit_v1.json)
binds source file hashes, executable hash, versions and sampling details.
No scene images are committed. A local annotated preview was inspected for all
six WT captures: polygons19-24 correctly overlay the neutral row even when the
physical chart is rotated. Stored coordinates are (x,y) in the supplied array.

## Released representation

The author calls these camera-RAW images after demosaicing. The actual HDF5
arrays are float32 HWC RGB, Samsung3024x4032x3 and Oppo3072x4096x3. They are
**not Bayer mosaics** despite retaining DNG CFA tags. The
[author example](https://github.com/shirawerman/Beyond-RGB/blob/0a3ed93e37f9f1b67fb23781c5e6ab34631020af/example%20code/H5_image_example.ipynb)
reads them directly as demosaiced camera-space RGB before white balance.
Notebook SHA256cfd55c4ccd060782d9bbbce388c2bbd5b0cb533f160094cd6b71f258efceca58.
Author code was consulted as format documentation, not imported or executed.

All values in the stride8 samples of all12 files lie on the1/255 grid within
2e-5 in scaled units. The files therefore do not provide unquantized10-bit sensor
precision merely because the metadata WhiteLevel is1023. The sample is not an
exhaustive value audit. Do not demosaic again, divide by1023, infer a gamma
from float dtype, or apply camera AsShotNeutral before illumination estimation.
The exact original quantization/processing history is not independently recovered.
Use the released camera-RGB convention and disclose that limitation.

ColorMatrix1/2 and CameraCalibration1/2 are present in all inspected metadata.
Samsung also has forward matrices. Their existence does not make an inference
path camera-independent if it uses them. Our initial raw-RGB path needs no CCM
or camera white-balance estimate. Decoding the known file format is separate
from requiring device calibration at inference. Oppo Orientation9 is not used
to invent a rotation: observed polygons already match the released array.

## Reference quality matters

Central50% polygon samples, with no correction, show:

| Scene | Phone | Max angle between neutral patches20-23 | White patch19 saturation |
|---|---|---:|---:|
|443|Samsung|1.981°|0%|
|443|Oppo|0.959°|100%|
|390|Samsung|2.573°|0%|
|390|Oppo|3.156°|0%|
|337|Samsung|2.087°|0%|
|337|Oppo|1.179°|0%|

These are **reference disagreement diagnostics, not model errors**. Dark-patch
quantization, patch reflectance, local illumination and noise may contribute;
the audit cannot separate them. Using patch19 blindly corrupts the Oppo443
reference. Using the darkest patches indiscriminately raises relative
quantization error. A fixed, model-independent reference-validity rule must be
frozen before reading reserved test images, with exclusions and denominator
reported. Do not choose a reference patch based on which makes a model win.

The WT chart supplies a reference for its location and capture time. It does not
prove that every surface in the NT frame shares exactly that illumination.
No skin DeltaE, absolute surface reconstruction claim or new phone accuracy
number is established by this loader audit.

## Reproduction

Install the research extra (h5py3.16.0 added to uv.lock). The development CLI
only permits scenes marked loader in the frozen selection. Five CPU tests cover
role isolation, native RGB preservation, x/y polygon interiors, saturation,
invalid polygons/bounds and nonfinite values. Tests passed locally.

```powershell
.venv/Scripts/python scripts/cc_phone_loader_audit.py --lock-sha256 86503dc8d2fdb247a57376c5a94ce5e0515003294b61286653b1e68099c0d38d --out experiments/runs/phone_loader_audit_new
```

Existing audit directories are never overwritten. The output contains an
uncommitted preview with display-only gamma; it is not an input to training.
