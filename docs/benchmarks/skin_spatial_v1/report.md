# Real-photo spatial and recurrent graph source experiment

54predeclared fits, all retained. Values are native instrument-referenceDeltaE00.
Each entry averages three separately trained seed scores; this is not an ensemble.
Source VAL was used for checkpoint selection. Camera transfer is source exploration,
not a new independent unseen-camera or ordinary facial-phone benchmark.

| Protocol | Method | Mean | Mean at80% | p95 | Seed means |
|---|---|---:|---:|---:|---|
|mixed|plain|3.4855|3.4300|7.1819|3.4851, 3.5127, 3.4587|
|mixed|conv1|3.4756|3.4063|7.1453|3.4870, 3.5036, 3.4361|
|mixed|conv3|3.4901|3.4532|7.2079|3.4654, 3.5320, 3.4728|
|mixed|graph1|3.6142|3.4955|7.5631|3.6740, 3.5900, 3.5787|
|mixed|graph3|3.6314|3.5191|7.9576|3.6812, 3.6312, 3.5818|
|mixed|graph3_scrambled|3.5430|3.5256|7.3092|3.5470, 3.5457, 3.5363|
|from_SLR|plain|5.6063|5.6837|10.9750|5.3482, 5.4913, 5.9794|
|from_SLR|conv1|6.3139|6.0779|12.2734|5.6139, 7.0013, 6.3263|
|from_SLR|conv3|6.6501|6.1535|13.1755|5.9482, 6.8641, 7.1381|
|from_SLR|graph1|6.6681|6.6331|13.1034|6.3747, 6.9272, 6.7025|
|from_SLR|graph3|6.3611|5.9568|13.7110|6.2429, 6.1289, 6.7117|
|from_SLR|graph3_scrambled|6.4279|6.1696|12.2903|5.4941, 7.2294, 6.5603|
|from_ipod|plain|6.1399|6.0788|10.7483|5.8650, 6.3605, 6.1943|
|from_ipod|conv1|5.9123|5.8197|10.5396|5.8414, 6.2994, 5.5962|
|from_ipod|conv3|5.7738|5.7106|10.5538|5.6855, 6.0182, 5.6176|
|from_ipod|graph1|5.8708|5.8860|10.3710|6.0947, 5.8317, 5.6860|
|from_ipod|graph3|5.8068|5.7015|10.4904|6.0462, 5.8499, 5.5243|
|from_ipod|graph3_scrambled|5.9265|5.9050|10.8005|6.0038, 6.0226, 5.7530|

80%ranking uses uncalibrated nativeLab patch-vote dispersion. It is not an
expected-error head, a calibrated accept threshold, or a validated coverage guarantee.
Full100/95/90/80/70/60%metrics are in every run result; all integer coverage points
are in [the curve CSV](uncalibrated_risk_coverage.csv).

## Matched mechanism comparisons

- mixed: graph1 minus conv1 = +0.1387DeltaE00; lower in 0/3seeds.
- mixed: graph3 minus conv3 = +0.1413DeltaE00; lower in 0/3seeds.
- mixed: graph3 minus plain = +0.1459DeltaE00; lower in 0/3seeds.
- mixed: graph3 minus graph3_scrambled = +0.0884DeltaE00; lower in 0/3seeds.
- from_SLR: graph1 minus conv1 = +0.3543DeltaE00; lower in 1/3seeds.
- from_SLR: graph3 minus conv3 = -0.2890DeltaE00; lower in 2/3seeds.
- from_SLR: graph3 minus plain = +0.7548DeltaE00; lower in 0/3seeds.
- from_SLR: graph3 minus graph3_scrambled = -0.0668DeltaE00; lower in 1/3seeds.
- from_ipod: graph1 minus conv1 = -0.0415DeltaE00; lower in 1/3seeds.
- from_ipod: graph3 minus conv3 = +0.0330DeltaE00; lower in 2/3seeds.
- from_ipod: graph3 minus plain = -0.3331DeltaE00; lower in 2/3seeds.
- from_ipod: graph3 minus graph3_scrambled = -0.1197DeltaE00; lower in 2/3seeds.

## Compute accounting

| Method | Stored parameters | Active parameters | Max fit+selection VRAM MiB |
|---|---:|---:|---:|
|plain|993287|924932|108.62|
|conv1|993287|993028|117.40|
|conv3|993287|993028|125.05|
|graph1|993287|990727|119.41|
|graph3|993287|990727|127.56|
|graph3_scrambled|993287|990727|126.66|

Unused modules preserve exactly matched initialization/storage across arms; active
parameters are the honest capacity comparison. Conventional spatial controls have
about0.23%more active parameters than graph variants. Plain has fewer parameters.
No pretrained weights, spectral pseudo-labels, camera identity or extra training data.
Only two already verified source cache files are loaded.

## Verification

162exact color arrays and 162risk arrays replayed;
9504independent scalarDeltaE00 cases and 324coverage rows checked.
Maximum metric discrepancy 4.88e-15.
Source hashes, fitting-only target scales, patient/camera boundaries, identical
initial states per seed, and same-camera epoch selection verified.

[Protocol](../../research/skin_spatial_protocol_v1.md), [audit](audit.json),
[decision](../../research/skin_spatial_next_decision.md).
