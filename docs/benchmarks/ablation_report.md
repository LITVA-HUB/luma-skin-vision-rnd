# Ablation status

Controlled real-data ablation results: **NOT MEASURED**. The C/C+/proposed smoke exercises different interfaces but is not a tuned causal ablation. In particular C+ changes attention and quality inputs together; it cannot isolate ROI value.

After real gates: hold data/seed/backbone/ROI/preprocessing/tuning budget fixed; remove measurement-aware ROI, hypotheses, ambiguity features, error head and replace with standard residual confidence separately. Include no correction, classical correction only, no teacher, then FP32/FP16 and optional INT8 after a useful model exists. Report identical-image coverage, ΔE00/tail error/subject-paired CI, parameters and measured resource costs. Reject any novelty claim if advantage over strong C+ disappears.
