# Paired-capture research evidence: negative mechanisms

This extends the direct instrument skin-color R&D evidence with27locally
trained source models and two source-camera-held-out fitting directions.
It does not replace the earlier independent400-image test.

Implemented paired-image supervision during training, a training-only context
regularizer, and an empirical nuisance-subspace projection. Inference uses
one image and924,932backbone parameters. Original MSKCC CC-BY data only;
no new third-party pretrained model, paid resource or external publication.

Measured source results reject the tested mechanisms as a large accuracy
improvement: baseline3.4855 versus consistency3.5123/VICReg3.5450/projection4.1101
meanDeltaE00 over3seeds. Consistency also loses all6camera-direction/seed
comparisons. Better repeated-capture agreement does not prove color trueness.

Verification:81exact prediction-array replays,5,544independent scalarDeltaE00
comparisons;265repository tests passed. [Full results](../benchmarks/skin_pair_v1/report.md).

This records a reproducible hypothesis rejection and the decision to test
capture-aware latent conditioning. It is not proof of an innovative mechanism,
ordinary facial-phone accuracy, patentability or an approved Skolkovo category.
