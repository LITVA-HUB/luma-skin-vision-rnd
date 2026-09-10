# Research problem

Status: **PLANNED; efficacy NOT MEASURED.**

The project asks whether an ordinary smartphone image can estimate the D65/2° CIELAB color of two defined cheek sites within a documented operating domain, and can identify images for which that estimate is unreliable. The target is the mean of repeated, registered instrument readings at each site. The primary error is ΔE00; the system may return `ACCEPT`, `RETAKE`, or `UNSUPPORTED`.

The problem is deliberately bounded. JPEG pixels combine illumination spectrum, sensor response, exposure, white balance, tone mapping, local ISP processing, geometry, reflection, cosmetics, and compression. Arbitrary images therefore do not uniquely determine physical reflectance or reference color. The research claim can only concern the collected domain and explicitly locked unseen-device/unseen-light tests.

The proposed mechanism combines measurement-suitability scoring for cheek pixels, a bounded set of non-generative photometric hypotheses, features that quantify disagreement and local instability, compact color regression, and a separate error model trained from subject-held-out residuals. Its useful claim is comparative: lower ΔE00 than strong alternatives at the same accepted-image coverage, with calibrated abstention. The implementation alone does not establish novelty or performance.

Inputs exclude identity features. Outputs do not infer identity, ethnicity, medical condition, or undertone. Skin-tone bands and display swatches may be derived product outputs, but they do not replace continuous color evaluation.

Success requires real, consented, instrument-paired data; repeatable targets; subject-disjoint train/validation/calibration/test partitions; strong C and C+ comparators; and paired uncertainty estimates. Synthetic data tests schemas, training, evaluation, and export plumbing only. It cannot demonstrate successful ROI learning, color accuracy, calibration, or real-world robustness.

