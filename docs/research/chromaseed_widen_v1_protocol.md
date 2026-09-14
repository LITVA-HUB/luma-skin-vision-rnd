# Luma ChromaSeed: capacity expansion WIDE v1

2026-09-13. Latest user steering: «ладно попробуй делать не настолько маленькие модели». Quality now takes priority over the previous kilobyte target. This protocol is written before new implementation, source-data fitting or selection. P8 primary session53555 finished exit0; its independent audit/runtime remain unfinished, so it is preliminary context, not an accepted result. Accepted warm lineage remains NP/ND, with the sealed WA source chain preserved.

## Question and bounded design

Does increasing the patch encoder and color head improve instrument-referenced skin-color prediction? Before implementation, user further prioritizes maximum quality over size. Compare five architectures in the same new experiment; 110k is no longer an upper target:

| Variant | Local width E | Head width H | Learned parameters |
| --- | ---: | ---: | ---: |
| tiny | 8 | 16 | 1179 |
| m31 | 64 | 128 | 30915 |
| m61 | 64 | 256 | 60611 |
| m111 | 128 | 256 | 110979 |
| m832 | 256 | 1024 | 832259 |

Each of64 normalized18-value patch descriptors passes through a shared linear E/ReLU encoder. Concatenate mean, stabilized population spread sqrt(variance+1e-6), and max:3E values. Hidden=ReLU(normalized color36 @ W+b+pooled @ G). Output=hidden @ V+c, then original target normalization is reversed. No iteration, ensemble prediction or external weights. More capacity, not a novelty claim: earlier R and pixel/copula models already used local pooling and larger networks.

Warm start from the same matching NP643 head used by P8, independently per inner fold/final fit and seed. Preserve all x/y normalizers. First16 columns of W/b and first16 rows of V copy NP; additional color W columns use NumPy RNG(seed+880003), uniform +/-1/sqrt36; added b and V are zero. Local U uses NumPy RNG(seed+770003), uniform +/-1/sqrt18; e and G are zero. This preserves the initial prediction within2e-8 native Lab in the NumPy consumer; widened matrix shapes need not be bitwise identical. Added units start learning via initially zero readout weights. No initialization of inner fits from full-fit models.

## Data and selection boundary

Only existing original TRAIN cache d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0:966 rows/24 people. Load color36,tokens64x18,target Lab3,patient/site/device only. No raw pixels, new data/images/packages/weights, legacy validation/calibration/test, external uploads or delegation. Reuse existing mixed,slr_to_ipod,ipod_to_slr roles and three person-disjoint, camera-stratified inner folds. These roles are historically reused exploratory evidence, not a new independent product benchmark. Native D65/10-degree DeltaE00, equal-person mean; three seed errors averaged, never their predictions. Camera/person composition is confounded.

Token normalizers are unweighted original FIT tokens mean/std in FP64 over rows and patches, cast FP32, floor1e-6. Person/site/image training weights and replacement sampling unchanged; camera metadata is not a model input.

## Training

Each capacity has its own fixed six-slot bank, seeds17/29/43 in order, rates0.0001/0.001 in order. Do not alter bank shape during replay or divide full bank time by six. Batch64, original observations only, shared source-index stream per seed RNG(seed+9001). AdamW beta.9/.999,epsilon1e-8,decay.01,per-model gradient clip5; normalized-Lab MSE mean(batch,channel), sum(model). Original fixed horizon8192, cosine factor .1+.45*(1+cos(pi*(t-1)/(8192-1))). Checkpoints0/512/2048/8192;524288 presentations per full trajectory. Short reconstruction uses the original horizon. FP32 CUDA deterministic/TF32 disabled; three CUDA Graph warmups, reset parameters/moments/counter/rates. Immutable base rates cloned. Existing BankAdamW is reused.

45 inner banks and15 final banks, six trajectories each:360 trajectories,1440 checkpoint exports including360 zero-step aliases. Inner1080 exports/204000 OOF prediction vectors. For each role/capacity select among six positive candidates (two rates, three steps) using(mean error,p90,numeric bytes,step,rate). This yields15 capacity-specific choices. Also include one common NP baseline per role and choose an overall policy across all31 candidates:93 total scores,18 choices. NP zero-step aliases must agree across capacities but count as one candidate per role. Freeze all choices before final fitting or held-role evaluation. Record all final curves, including losses from extra training; do not revise choices after seeing them.

## Validation and reporting

Final360 checkpoint records,143760 clean prediction vectors. For each capacity's selected three seeds plus three NP controls per role, evaluate all33 existing color/token affine stress settings (identity plus eight corners at1/255,4/255,16/255,64/255).54 selected/control models,711612 stress vectors including identity. Synthetic fixed-target stress is an assumption, not new measured skin-color ground truth. These diagnostics do not select models.

Before fitting: meaningful synthetic tests for initial function preservation, increased trainable capacity, fit-only normalizers, finite constant-patch backward, permutation invariance, actual learning of added units, fixed-horizon LR prefix, CPU/GPU forward parity and bitwise fixed-bank short replay. Native-Lab NumPy/direct tolerance2e-8, CUDA/NumPy tolerance.002. Freeze primary sources and consumed lineage before inner fitting. Independent audit verifies hashes/splits/normalizers/sampling/initialization, all inner and final clean predictions with explicit neuron reductions, recomputes metrics and every selection, and runs actual single-example consumers on every selected/control stress row. Validate paired per-person differences without claiming independence between roles.

Measure model-only one-example CPU response including token normalization/encoding/pooling,20 warmups and three passes; image decoding/feature extraction excluded. Measure complete selected-capacity construction with all original ND/NP warm training and a fixed six-slot continuation to the selected checkpoint, original8192 horizon; three repetitions, first warmup. Compare selected exports bitwise, charge setup and the whole bank. Full original-grid research cost is reported separately. NP control has all its original warm training charged. No division into invented single-model training times.

Write a compact result table with capacity, actual payload/file sizes, selected steps, error by role and response latency. Preserve unfinished P8 audit as an explicit separate outstanding item. Larger-model work proceeds without depending on its acceptance. Ordinary-phone facial/end-to-end performance remains unvalidated; this experiment cannot complete the broad goal or establish Skolkovo eligibility.
