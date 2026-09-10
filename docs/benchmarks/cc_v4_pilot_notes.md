# V4 integration receipts and errata

Pilot cc_v4_transport_pilot_s17:2epochs, real source train1126/validation119, field warmup not yet enabled. The run only verifies integrated forward/backward/checkpoint/evaluation, not action-field learning or a benchmark gain. Its source/model/config/GT-cache identities are in the immutable run artifact manifest.

The separate20repetition pilot benchmark `experiments/runs/ccv4_transport_pilot_s17_benchmark.json` was inadvertently launched while the first full posterior training process was active. **All of its latency numbers are CONTENDED/INVALID for comparison and excluded from reported model timing.** It verifies only benchmark-script execution and observed25/51/103 query accounting. Preserve the file; definitive100repeat benchmarks run after all training ends, one process at a time. The first full posterior run's wall-time includes this brief contention; its optimizer steps/data/training protocol are unchanged. Report training time as elapsed measurement, not a controlled throughput comparison.

No external test/camera GT or facial data were decoded in either pilot. No V4 output establishes skin DeltaE.
