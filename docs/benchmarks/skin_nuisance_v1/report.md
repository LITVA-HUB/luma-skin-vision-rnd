# nuisance: real skin source mechanism screen

All36predeclared fits retained. Values average three separate seed scores, not an
ensemble. Same source data/80epoch budget; native instrumentLab scored withCIEDE2000.
The source roles and camera families have already been examined during research:
this is not an independent test or ordinary facial-phone validation.

| Protocol | Method | MeanDeltaE00 | p95 | Seed means |
|---|---|---:|---:|---|
|mixed|learned|3.6394|7.1259|3.5921, 3.6626, 3.6634|
|mixed|fixed_grid|3.6128|7.2335|3.5904, 3.6336, 3.6144|
|mixed|global|3.7441|7.5480|3.7177, 3.7498, 3.7647|
|mixed|bias|5.5085|10.6669|5.0731, 5.7739, 5.6786|
|mixed|historical_plain|3.4855|7.1819|3.4851, 3.5127, 3.4587|
|mixed|capture_plain|3.4771|7.0733|3.4957, 3.5284, 3.4072|
|mixed|capture_mixture|3.4406|6.9316|3.3783, 3.4698, 3.4736|
|mixed|training_only_graph|3.6394|7.1259|3.5921, 3.6626, 3.6634|
|from_SLR|learned|5.8339|10.9445|4.6711, 6.0218, 6.8088|
|from_SLR|fixed_grid|5.2823|10.1647|4.7810, 5.8392, 5.2268|
|from_SLR|global|5.2972|9.9990|5.2410, 5.0996, 5.5512|
|from_SLR|bias|6.4028|12.2688|6.5107, 6.1860, 6.5117|
|from_SLR|historical_plain|5.6063|10.9750|5.3482, 5.4913, 5.9794|
|from_SLR|capture_plain|5.8301|11.2459|5.6802, 6.0602, 5.7499|
|from_SLR|capture_mixture|5.0193|9.7825|4.9975, 5.2417, 4.8187|
|from_SLR|training_only_graph|5.8339|10.9445|4.6711, 6.0218, 6.8088|
|from_ipod|learned|4.9736|9.4600|5.2933, 4.8977, 4.7298|
|from_ipod|fixed_grid|5.1586|9.8748|5.8467, 4.9717, 4.6575|
|from_ipod|global|5.4993|10.2016|5.4342, 5.6902, 5.3735|
|from_ipod|bias|8.5210|13.8889|8.3623, 8.2083, 8.9925|
|from_ipod|historical_plain|6.1399|10.7483|5.8650, 6.3605, 6.1943|
|from_ipod|capture_plain|5.5089|9.4018|5.3087, 5.6826, 5.5353|
|from_ipod|capture_mixture|5.9635|11.0896|5.7634, 6.4146, 5.7124|
|from_ipod|training_only_graph|4.9736|9.4600|5.2933, 4.8977, 4.7298|

Historical rows are LOCALLY REPRODUCED compatible source controls; bindings were
rechecked. Strong previous results are retained, rather than comparing only to weak
members of the current family.

## Risk and compute

| Protocol | Method | Mean at80% | Nominal inference params | Nominal training params | Max fit+selection MiB |
|---|---|---:|---:|---:|---:|
|mixed|learned|3.6342|924932|990727|127.56|
|mixed|fixed_grid|3.5503|924932|990468|112.66|
|mixed|global|3.7410|924932|990468|112.66|
|mixed|bias|5.8001|924932|990468|112.13|
|from_SLR|learned|5.8116|924932|990727|123.23|
|from_SLR|fixed_grid|5.2196|924932|990468|108.73|
|from_SLR|global|5.2712|924932|990468|108.73|
|from_SLR|bias|6.2313|924932|990468|108.70|
|from_ipod|learned|4.8452|924932|990727|125.23|
|from_ipod|fixed_grid|5.0826|924932|990468|110.74|
|from_ipod|global|5.3890|924932|990468|110.74|
|from_ipod|bias|8.4898|924932|990468|110.70|

Risk is uncalibrated patch-vote dispersion, not expected error or an accept guarantee.
All100/95/90/80/70/60%metrics are in run results; [full curves](uncalibrated_risk_coverage.csv).

All inference uses the same plain core. Fixed/global operators are image-independent
but their latent residuals are image-dependent. Bias has only256functional offset
dimensions despite its nominal65,536matrix parameters.9learned-control final weight
sets exactly match the previous training-only graph experiment.

A declared [test-only precision amendment](test_precision_amendment/amendment.json)
checks mathematical constant preservation inFP64 rather than an overly tightFP32
absolute tolerance. Original test/lock archived. Model, data and fitting unchanged.

## Verification

108color and108risk arrays exactly replayed;
6336independent scalar cases/216coverage rows checked.
Fitting-only target scales, identical per-seed initializations and same-camera epoch
selection verified. TEST/CAL and UMINHO held-out endpoints were not used.

[Protocol](../../research/skin_nuisance_protocol_v1.md), [audit](audit.json),
[decision](../../research/skin_representation_next_decision.md).
