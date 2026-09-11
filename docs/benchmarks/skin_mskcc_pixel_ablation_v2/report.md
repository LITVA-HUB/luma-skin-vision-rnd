# Adaptive direct-skin source ablations v2

These reuse the six-person development set. No calibration/test person was opened.

|Ablation|Seed|Mean DeltaE00|Median|p95|
|---|---:|---:|---:|---:|
|global_mlp|17|3.8113|3.2951|7.9013|
|global_mlp|29|3.7303|3.2437|8.2877|
|global_mlp|43|3.9020|3.3571|8.5492|
|shared_tight|17|3.4905|2.9545|7.2489|
|shared_tight|29|3.5197|3.0923|7.0204|
|shared_tight|43|3.4858|3.0482|7.1764|
|local_mean|17|3.9570|3.4841|7.5807|
|local_mean|29|3.7770|3.3333|7.6092|
|local_mean|43|3.8869|3.5400|7.1924|
|local_tight|17|4.0014|3.9262|7.8264|
|local_tight|29|3.9911|3.6766|7.5569|
|local_tight|43|3.9713|3.7638|7.6465|

The capacity-matched global MLP has938755parameters; each vote model924932.
Local-only votes lose useful context. Tenfold tighter robust iterations do not provide a reliable gain.
These negative mechanisms remain evidence, not discarded runs. Standard error-head/calibration and independent test remain next steps.
All24best/final checkpoints replay exactly, with independent scalar DeltaE00 checks.
