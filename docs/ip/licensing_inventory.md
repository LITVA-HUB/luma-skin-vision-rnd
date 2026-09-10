## Current CC v2 artifact update

Phone-loader dependency h5py3.16.0 is pinned in uv.lock. Installed wheel metadata
and bundled notices (h5py BSD-style license, HDF5, LZF and other notices) are
preserved in [the exact notice receipt](../data/provenance/mobile_screen_2026_09_11/h5py316_notices/manifest.json).
This addition supports local data preparation; it did not upgrade torch/numpy
or change V5 training code. The iPhone SE2/XS Max auxiliary archive has original
CC0 data terms, with scientific-target limitations documented separately.

2026-09-11: an optional standard Apache2.0 DINOv2-S teacher was acquired and CPU-smoke-checked; it has not entered V1–V5 training or deployment. [Exact model/source rights and hashes](dinov2_teacher_adoption.md). Beyond RGB phone data has a verified original-author CC BY4.0 release; its separate code license is not assumed. [Phone data rights](../data/smartphone_benchmark_plan.md).

V2 estimator, risk and threshold parameters are fitted solely on CC BY4.0 SimpleCube++; INTEL-TAU CC BY-SA4.0 remains evaluation-only. No pretrained or teacher weights are imported. The known Cotogni/Cusano normalization/restoration mechanism is explicitly prior art. Positive selective transfer versus matched C+ does not establish patentability; cheap-control competition, source/Canon failures and original negative milestones remain documented. [V2 report](../benchmarks/cc_v2_report.md).

The FP32 ONNX combines independently written wrapper code, torchvision MobileNetV3-large, and locally fitted StandardScaler/Ridge parameters. New research libraries are scikit-learn1.9.0, joblib1.6.0 and threadpoolctl3.6.0 (BSD-3-Clause); SciPy1.18.1 includes its BSD notice plus bundled-library notices. Installed license metadata/texts are preserved in [research dependency receipts](../benchmarks/cc_v2/reproducibility/installed_research_licenses.json). Retain applicable third-party notices and dataset attribution. Dataset-dependent evaluation reports retain INTEL-TAU provenance/conditions; no blanket ruling on hypothetical BY-SA-trained weight licensing is made. No external publication or legal clearance assertion occurred.

Historical inventory and V1 conclusions follow; they are not retroactively rewritten.

---

# Licensing and provenance inventory — 2026-09-10

This is an engineering intake record, not legal clearance. A code license is not automatically a weight/data license. A published paper is not permission to redistribute images, train commercial models, or use a teacher's outputs. No pretrained checkpoint or research participant dataset has been adopted in this iteration. Library versions are fixed by the repository lockfile; before distribution, generate the installed dependency/SBOM license report, retain notices and review transitive binary components.

## Models and research artifacts

| Artifact | Code terms | Weight terms | Data terms | Decision / commercial status |
| --- | --- | --- | --- | --- |
| torchvision MobileNetV3-small | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | NONE imported: `weights=None`; optional ImageNet checkpoint separate review | Synthetic generated examples only in smoke; real data not collected | Code usable subject to notices; no blanket product clearance. Optional pretrained path NOT CLEARED |
| MobileNetV4 official TensorFlow implementation | [Apache-2.0](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py) | Exact checkpoint NOT VERIFIED | ImageNet/distillation sources need artifact-specific review | Candidate only; full chain NOT CLEARED |
| MobileOne | [Apple custom terms](https://github.com/apple/ml-mobileone/blob/main/LICENSE), component acknowledgements | Linked checkpoints: scope NOT VERIFIED | ImageNet rights separate | Candidate only; NOT CLEARED as full pretrained chain |
| FastViT | [Apple custom terms](https://github.com/apple/ml-fastvit/blob/main/LICENSE), component acknowledgements | Checkpoint grant NOT VERIFIED | ImageNet and any distillation lineage separate | Candidate only; NOT CLEARED as full pretrained chain |
| Original DINOv2 | [Apache-2.0 code and original weights](https://github.com/facebookresearch/dinov2) | Original family Apache-2.0 per README; pin artifact | LVD training corpus not conveyed by software license | Candidate only; training-data provenance review required |
| X-Ray DINO in same repository | [Distinct research terms](https://github.com/facebookresearch/dinov2/blob/main/LICENSE_XRAY_DINO_MODEL) | Noncommercial research restrictions | Separate provenance | Excluded; do not infer license from repository name |
| YuNet OpenCV Zoo | Folder [MIT](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE); OpenCV library separate Apache-2.0 | [Model directory](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) has local MIT terms; exact ONNX hash/license must be recorded on supply | WIDER FACE lineage; upstream data rights separate, not independently cleared | Optional adapter only, no model bundled. Keep existing YuNet choice; NOT CLEARED for a new supplied artifact until provenance recorded |
| MediaPipe FaceMesh V2 / Face Landmarker | Framework/sample code Apache-2.0 | [FaceMesh V2 model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf) says Apache-2.0; full task bundle includes detector/blendshape components, check each | Training corpus permission/consent not independently verified | Candidate geometry only; exact bundle NOT CLEARED |
| dlib 68-point landmark predictor | Library Boost license, distinct from model | [Official example](https://dlib.net/face_landmark_detection.py.html) warns of iBUG 300-W noncommercial dataset terms | iBUG 300-W restrictions | Do not adopt pretrained predictor for commercial use without clearance |
| CelebAMask-HQ / its supplied software | [Noncommercial research/education](https://github.com/switchablenorms/CelebAMask-HQ) | A parser trained on it needs separate rights review, irrespective of parser-code license | Dataset agreement explicitly noncommercial, restricts derived data and redistribution | Excluded from commercial training by default; 19 semantic classes are not measurement-reliability labels |
| Other parsing checkpoints (BiSeNet, SegFormer, community face parsing) | Implementation-specific; NOT VERIFIED | Checkpoint-specific; NOT VERIFIED | Trace CelebA/CelebAMask-HQ/LaPa/etc.; NOT VERIFIED | No checkpoint adopted. Architecture license cannot clear its training lineage |
| DeepWB | [Research-only / noncommercial](https://github.com/mahmoudnafifi/Deep_White_Balance) | No separate commercial grant verified | Rendered WB has separate source terms | Excluded from product/teacher pipeline; written clearance needed |
| C5 | [Apache-2.0](https://github.com/mahmoudnafifi/C5/blob/main/LICENSE) | Exact checkpoint scope NOT VERIFIED | NUS/Gehler-Shi/other datasets separate | Code candidate; pretrained chain NOT CLEARED |
| TRUST / BalanceAlb / FAIR materials | [Noncommercial scientific license](https://github.com/HavenFeng/TRUST/blob/main/LICENSE) | Same restriction; commercial-model training expressly prohibited | Associated data covered/restricted; other model dependencies separate | Excluded from commercial use, distillation and pseudo-label production |
| Ren 2023 / HUST 2025 / YLGTD 2026 / VLM-CC 2026 | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | Literature review only; NOT CLEARED |
| SelectiveNet / Guo / LTT reference implementations | NOT VERIFIED in this review | NOT VERIFIED or not intrinsic | Example datasets separate | Independently implement mathematical baseline; do not copy uncleared source |

## Runtime dependency intake

Adopted/locked direct packages reported by the integration run: torch 2.8.0+cu128 (BSD-style), torchvision 0.23.0+cu128 (BSD-3-Clause), numpy 2.5.3 (BSD-3-Clause), Pillow 12.3.0 (HPND-style), pydantic 2.13.5 (MIT), PyYAML 6.0.3 (MIT), pytest 9.1.1 (MIT), Ruff 0.16.6 (MIT); optional opencv-python-headless 4.14.0.94 (OpenCV Apache-2.0 plus wheel components), onnx 1.22.0 (Apache-2.0), onnxruntime 1.29.0 (MIT). Torch/torchvision are a deliberately pinned stable pair, not a latest-version claim. See uv.lock for actual resolved/transitive packages. These are provisional package-family labels, not a completed audit of the exact wheel/transitive CUDA/codec distributions. Authoritative license files: [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE), [Pydantic](https://github.com/pydantic/pydantic/blob/main/LICENSE), [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [OpenCV](https://github.com/opencv/opencv/blob/4.x/LICENSE), [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE). No assertion of cleared third-party model rights follows from installing these libraries.

## Intake record required before adopting any new artifact

Exact installed distribution metadata and license-file SHA256 values are recorded in [installed_license_metadata.json](installed_license_metadata.json), including optional matplotlib 3.11.1 and its plotting dependencies. These are local wheel metadata observations, not blanket legal clearance. Matplotlib uses its project license (PSF/BSD-derived terms); review the recorded exact license files before redistribution. No plotting model weights or dataset were introduced. Numerical Sharma test fixtures retain author attribution in tests/fixtures/README.md; supplementary-data redistribution rights remain to be reviewed before public release.

Record owner, source URL and revision, download timestamp, SHA256, exact code/weight/data license texts, grant scope for commercial R&D/inference/fine-tuning/distillation/redistribution, training lineage, attribution and NOTICE obligations, participant permissions where applicable, reviewer/date and decision. Store license evidence alongside the manifest, not private images in Git. Unknown terms block adoption, not independent literature analysis. Company R&D is not automatically noncommercial research.
