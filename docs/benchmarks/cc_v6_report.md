# V6 completed: canonical-frame combination did not improve the source result

All 12 primary arms and 24 best/final checkpoints completed. Independent CPU
weight replay and FP64 error rescoring passed for every checkpoint. Same 1126
real SimpleCube++ training / 119 reused validation images, CC BY 4.0; 120 epochs,
three seeds and matched common 20-epoch full states. All three CUDA warmup
replays were bitwise equal. No new test/camera/skin accuracy claim.

Metrics below are three-seed arithmetic means, not an inference ensemble.
Best checkpoints were selected on these same validation images; risk is raw,
without a separate calibration. 80% coverage means 95/119 accepted per model.

|Arm|Best full mean °|Final mean °|Raw risk80 °|
|---|---:|---:|---:|
|point|3.2723|3.4136|Not trained|
|posterior_random|3.2980|3.4170|2.6671|
|action_random|3.2214|3.3189|2.5211|
|transport_random|3.2105|3.3101|2.5440|

|Coverage|Posterior|Generic action|Physical transport|
|---|---:|---:|---:|
|100%|3.2980|3.2214|3.2105|
|95%|2.8967|2.8251|2.7988|
|90%|2.7461|2.7741|2.6572|
|80%|2.6671|2.5211|2.5440|
|70%|2.5072|2.3965|2.3348|
|60%|2.4500|2.1764|2.2696|

The best V6 combination remains worse on this reused source validation than
V5's generic action mean 2.4136° / raw risk80 2.1333°, and physical transport
mean 2.4548° / raw risk80 2.0192°. Those V5 figures are locally reproduced
historical controls, not published author numbers. Canonicalization does not
automatically combine the best properties of an anchor and a learned critic.

All architectures contain 3,097,189 parameters; saved model state is 12,599,089
bytes. Peak allocated training memory is 859.98–864.07 MiB including the source
cache and common state. No V6 inference timing or export was measured.

An inherited unused oracle helper clips absolute rather than residual actions.
Separate corrected receipts for all three seeds preserve original artifacts;
all 24 corrected oracle2 minimum arrays have zero change. Training, selection
and achieved error/risk never use the oracle. Corrected receipts, not the old
oracle field, govern any future diagnostic claim.

Decision: retire V6 as the main accuracy candidate; preserve it as a negative
ablation. Continue the distinct semantic/sensor training screen and the compact
Fourier representation branch. Source-only gains must survive a newly locked
external-camera evaluation before changing product claims.

[Per-seed metrics, curves and hashes](cc_v6/three_seed_summary.json).
Original per-image predictions and fixed-coverage/tail tables are archived
under cc_v6/seed17, seed29 and seed43; dataset images and weights are excluded.
