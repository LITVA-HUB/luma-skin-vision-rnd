# Gradient-transfer diagnostic provenance

Project-authored diagnostic code under repository terms; existing project
checkpoints trained on original MSKCC CC-BY real skin images/native Lab. No new
third-party implementation, pretrained weights, restricted data or cloud service.
Fixed models and TRAIN cache are bound by exact SHA256 in the source lock.

Known research reviewed from primary sources on2026-09-11:
[Fish: Gradient Matching for Domain Generalization](https://arxiv.org/abs/2104.09937)
uses an inter-domain gradient matching objective and first-order approximation;
[Fishr, ICML2022](https://proceedings.mlr.press/v162/rame22a.html) matches domain-level
gradient variances. These are prior art, not our innovation. No author accuracy
numbers are represented as local reproductions. This diagnostic implements
neither complete published training method.

The local contribution is controlled evidence about the present skin estimator,
not a patentability assertion. Negative gradients and small TRAIN-only color
changes do not establish a technical advantage. Future relational comparison
and compatibility ideas require a separate prior-art and mechanism pass.

No independently improved accuracy, new calibrated rejection guarantee or
ordinary-phone skin-color proof follows from these artifacts.
