# Beyond RGB phone reference and preprocessing protocol v1

Frozen before decoding any of the44 reserved test scenes. Three official TRAIN
scenes were inspected for loader development; see phone_loader_findings.md.
This is a custom phone transfer screen using author data, not a reproduction of
the authors' multispectral ISE protocol. Dataset: original CC BY4.0 release.

Input is a single NT HWC float32 demosaiced camera-RGB image. No WT image, chart
location, spectrum, AsShotNeutral, device identity or CCM enters the estimator.
No gamma decoding, second demosaicing or10-bit scale division. Leave array
orientation as released. This does not demonstrate ordinary JPEG/HEIC use.

Require finite values in[0,1]. Zero out pixels whose maximum channel is at least
254/255 or zero, before feature extraction. This explicit saturation margin
reflects the observed quantized release and applies to every method. Preserve
author-blurred regions as supplied; record any provided privacy mask paths.
Calculate the four existing classical experts from the full masked input. For
CNNs use existing cc.data.sample: area-resize128x128, image-only95th-percentile
common-exposure scale, clamp[0,4]. No per-camera gain, fitting or adaptation.

Reference comes exclusively from WT polygons in original(x,y) coordinates.
Take per-channel medians inside the central50% convex polygon, matching the
audited loader. Inspect all24 patches for diagnostics, but use a fixed neutral
reference rule: candidates20,21,22, in that brightness order. A candidate needs
at least64 sampled pixels, no more than0.5% pixels with any channel>=254/255,
and minimum median channel>=8/255. Require at least two usable candidates and
their maximum pairwise recovery angle<=3°. Choose the first usable candidate;
normalise its median RGB to a unit vector. Report chosen patch and the entire
quality record. Patches19/23/24 do not change selection: white saturation and
dark-patch quantization were observed in loader data. This quality policy is
an engineering choice based on TRAIN inspection, not an author standard.

An invalid reference is reported as unscorable, never as a zero model error or
silently removed. Report all88 planned inputs, per-camera scorable counts,
scene pairing and each exclusion reason. Quality filtering depends on reference
integrity only, never model predictions. Fixed-coverage model curves use the
scorable population, with total-population coverage also disclosed. Do not
conflate reference exclusions with model rejection. Report reference mismatch
diagnostics; the policy does not certify globally uniform illumination or an
instrument uncertainty bound. Quantized chart-derived illumination is the
target, not measured skin color or a manufactured DeltaE target.

The first implementation only prepares the three TRAIN loader scenes. Before
permitting reserved-test preparation/scoring, require the complete download
verification and a separate immutable evaluation lock binding method/weight/
calibration hashes and this protocol. No test selection of checkpoints, heads,
reference rules or thresholds. The phone models are unseen by source training;
both phones in each scene stay in the same role. Bootstrap scenes, not individual
paired-phone captures. A source-trained model tested on phones is transfer
evaluation; it is not phone-trained leave-one-camera-out evidence.
