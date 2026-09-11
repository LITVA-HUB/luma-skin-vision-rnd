# Training-only spatial branch: real skin source follow-up

36predeclared follow-up fits, with zero branch steps at EVERY selection/evaluation endpoint.
The earlier post-hoc discovery was on these same source roles: this is not an independent
test or an unbiased estimate after architecture search. Targets are actual instrument
nativeLab. No camera ID, spectral pseudo-label, extra image or pretrained weight.

Each number averages three separately trained seed scores; not an ensemble.

| Protocol | Method | MeanDeltaE00 | p95 | Seed means |
|---|---|---:|---:|---|
|mixed|graph_always|3.6394|7.1259|3.5921, 3.6626, 3.6634|
|mixed|conv_always|5.2915|10.8496|5.0627, 5.1969, 5.6149|
|mixed|graph_drop|3.4493|7.1375|3.4517, 3.4374, 3.4589|
|mixed|conv_drop|3.4777|7.2189|3.4505, 3.5157, 3.4669|
|mixed|original_plain|3.4855|7.1819|3.4851, 3.5127, 3.4587|
|mixed|ordinary_conv3|3.4901|7.2079|3.4654, 3.5320, 3.4728|
|mixed|capture_plain_mse|3.4771|7.0733|3.4957, 3.5284, 3.4072|
|mixed|capture_mixture_mse|3.4406|6.9316|3.3783, 3.4698, 3.4736|
|from_SLR|graph_always|5.8339|10.9445|4.6711, 6.0218, 6.8088|
|from_SLR|conv_always|6.6652|12.6219|6.8186, 6.3140, 6.8629|
|from_SLR|graph_drop|5.4829|10.7564|5.7274, 5.1249, 5.5964|
|from_SLR|conv_drop|6.0488|12.1188|5.7965, 6.3821, 5.9676|
|from_SLR|original_plain|5.6063|10.9750|5.3482, 5.4913, 5.9794|
|from_SLR|ordinary_conv3|6.6501|13.1755|5.9482, 6.8641, 7.1381|
|from_SLR|capture_plain_mse|5.8301|11.2459|5.6802, 6.0602, 5.7499|
|from_SLR|capture_mixture_mse|5.0193|9.7825|4.9975, 5.2417, 4.8187|
|from_ipod|graph_always|4.9736|9.4600|5.2933, 4.8977, 4.7298|
|from_ipod|conv_always|9.7375|15.8605|10.2993, 9.1095, 9.8036|
|from_ipod|graph_drop|5.7700|10.1397|5.7413, 5.6008, 5.9681|
|from_ipod|conv_drop|5.6434|9.7481|5.2833, 5.7227, 5.9242|
|from_ipod|original_plain|6.1399|10.7483|5.8650, 6.3605, 6.1943|
|from_ipod|ordinary_conv3|5.7738|10.5538|5.6855, 6.0182, 5.6176|
|from_ipod|capture_plain_mse|5.5089|9.4018|5.3087, 5.6826, 5.5353|
|from_ipod|capture_mixture_mse|5.9635|11.0896|5.7634, 6.4146, 5.7124|

Historical controls above are LOCALLY REPRODUCED, with code/data bindings rechecked.
Capture-plainMSE and capture-mixtureMSE use approximately0.929Mparameters and the
same source data, resolution and80epoch budget; their stronger results are retained.
Always versus drop differs only in branch usage during fitting (.5enabled per batch
for drop). All four methods deploy the same924,932parameter plain inference core.
Training auxiliaries remain in the checkpoint; stored count993,287is not active size.

## Uncalibrated risk diagnostic

| Protocol | Method | Mean at80% | Training active parameters | Max fit+selection MiB |
|---|---|---:|---:|---:|
|mixed|graph_always|3.6342|990727|127.56|
|mixed|conv_always|5.4543|993028|125.05|
|mixed|graph_drop|3.4315|990727|127.16|
|mixed|conv_drop|3.4888|993028|124.65|
|from_SLR|graph_always|5.8116|990727|123.23|
|from_SLR|conv_always|6.4832|993028|121.64|
|from_SLR|graph_drop|5.4922|990727|123.23|
|from_SLR|conv_drop|5.6943|993028|120.72|
|from_ipod|graph_always|4.8452|990727|125.23|
|from_ipod|conv_always|9.9902|993028|122.72|
|from_ipod|graph_drop|5.7012|990727|125.23|
|from_ipod|conv_drop|5.5595|993028|122.72|

Dispersion ranking is uncalibrated and may worsen accepted-image error. All fixed
coverage endpoints are in run results, with full [risk curves](uncalibrated_risk_coverage.csv).
No expected-error or guaranteed acceptance claim follows.

## Verification

108color and108risk arrays exactly replayed;
6336independent scalar cases/216coverage rows pass.
18always-arm final parameter sets exactly equal the prior spatial fits.
This verifies that their training trajectory was unchanged; only inference and
same-camera checkpoint selection differ. Fit-only target scales and subject/camera
boundaries pass. No MSKCC TEST/CAL or UMINHO held-out data were loaded.

[Protocol](../../research/skin_train_branch_protocol_v1.md), [audit](audit.json),
[decision](../../research/skin_spatial_next_decision.md).
