# Optional teacher acquisition, 2026-09-11

Standard DINOv2 ViT-S/14 was downloaded from Meta's original asset URL after
checking its original [model card](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md)
and [Apache2.0 license](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE).
This covers the standard LVD-142M model; it does not adopt XRay/Cell-DINO assets.

- Source commit7764ea0f912e53c92e82eb78a2a1631e92725fc8;19 exact source/document
  files,107,009 bytes, separately archived with SHA256 receipts.
- [Original weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth):88,283,115 bytes;
  SHA256 b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9.
- Local path artifacts/teachers/dinov2-7764ea0f912e, outside committed weights.
- Strict CPU state loading with torch.load(weights_only=True) passed, no missing
  or unexpected tensors. Actual parameter count22,056,576. Synthetic224px
  forward produced finite384-dimensional class and256 patch tokens. No dataset
  image features have yet been extracted; no accuracy was measured.
- The original URL-loading helper compares a torch version string with a tuple
  in this revision. It was not called. Construct pretrained=False, then use the
  already acquired local weights with strict loading. No source changes made.

Candidate purpose: training-only scene-context teacher for a compact student.
It exceeds the intended deployment size and is not being installed into the
product inference path. V1–V5 experiments use no pretrained teacher and retain
their original lineage. A future teacher trial must compare GT-only/own-model
EMA and ordinary distillation controls; semantic invariance cannot substitute
for measured photometric labels. Teacher pretraining image overlap with our
benchmarks cannot be independently audited from the available LVD-142M manifest.
No universal or skin accuracy conclusion follows from this acquisition.

Apache attribution/license obligations are recorded, and original notices are
preserved in [the source archive](../research/cc_v6_sources/dinov2/source/LICENSE).
No upstream data collection or third-party patent clearance is implied.

V7 update: local CPU extraction completed in 238.88 seconds for 1126 TRAIN
images, raw/GT-corrected views and two orientations each. Frozen feature arrays
are outside Git; config/manifest/role audit are archived with the V7 evidence.
No VAL/RISK/CAL/TEST or phone features were extracted. Standard student training
is now underway under the matched five-arm protocol. The no-dataset-use
statement above describes the earlier acquisition stage only. No V7 camera
accuracy claim is established yet; teacher pretraining overlap remains unknown.
