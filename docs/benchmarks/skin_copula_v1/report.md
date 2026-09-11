# copula: real skin source mechanism screen

All36predeclared fits retained. Values average three separate seed scores, not an
ensemble. Same source data/80epoch budget; native instrumentLab scored withCIEDE2000.
The source roles and camera families have already been examined during research:
this is not an independent test or ordinary facial-phone validation.

| Protocol | Method | MeanDeltaE00 | p95 | Seed means |
|---|---|---:|---:|---|
|mixed|none|3.4801|6.9917|3.5003, 3.5000, 3.4399|
|mixed|rgb_hist|3.6507|7.6661|3.6433, 3.6643, 3.6444|
|mixed|copula|3.6460|7.6904|3.6209, 3.6419, 3.6752|
|mixed|rank_only|5.0899|10.8992|5.1248, 5.0504, 5.0944|
|mixed|historical_plain|3.4855|7.1819|3.4851, 3.5127, 3.4587|
|mixed|capture_plain|3.4771|7.0733|3.4957, 3.5284, 3.4072|
|mixed|capture_mixture|3.4406|6.9316|3.3783, 3.4698, 3.4736|
|mixed|training_only_graph|3.6394|7.1259|3.5921, 3.6626, 3.6634|
|from_SLR|none|5.7770|11.2446|6.1093, 5.3233, 5.8983|
|from_SLR|rgb_hist|5.6347|10.5217|5.5596, 5.5686, 5.7759|
|from_SLR|copula|6.0339|12.2945|6.1230, 5.7561, 6.2225|
|from_SLR|rank_only|8.8349|18.5104|8.9509, 9.3180, 8.2359|
|from_SLR|historical_plain|5.6063|10.9750|5.3482, 5.4913, 5.9794|
|from_SLR|capture_plain|5.8301|11.2459|5.6802, 6.0602, 5.7499|
|from_SLR|capture_mixture|5.0193|9.7825|4.9975, 5.2417, 4.8187|
|from_SLR|training_only_graph|5.8339|10.9445|4.6711, 6.0218, 6.8088|
|from_ipod|none|5.9897|10.4261|5.8611, 5.8138, 6.2943|
|from_ipod|rgb_hist|7.9333|14.6224|8.1890, 8.0095, 7.6014|
|from_ipod|copula|5.5702|11.0199|5.3783, 6.1041, 5.2283|
|from_ipod|rank_only|6.1987|13.4579|6.1335, 6.5183, 5.9444|
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
|mixed|none|3.4601|1072644|1072644|112.68|
|mixed|rgb_hist|3.6186|1072644|1072644|112.68|
|mixed|copula|3.7053|1072644|1072644|112.68|
|mixed|rank_only|5.3884|1072644|1072644|112.68|
|from_SLR|none|5.8585|1072644|1072644|107.51|
|from_SLR|rgb_hist|5.6146|1072644|1072644|107.51|
|from_SLR|copula|6.2065|1072644|1072644|107.51|
|from_SLR|rank_only|8.2677|1072644|1072644|107.51|
|from_ipod|none|5.9122|1072644|1072644|110.46|
|from_ipod|rgb_hist|8.3218|1072644|1072644|110.46|
|from_ipod|copula|5.5179|1072644|1072644|110.46|
|from_ipod|rank_only|5.7064|1072644|1072644|110.46|

Risk is uncalibrated patch-vote dispersion, not expected error or an accept guarantee.
All100/95/90/80/70/60%metrics are in run results; [full curves](uncalibrated_risk_coverage.csv).

Copula and RGB-histogram context have exactly matched architecture and capacity.
None has a constant branch input, so nominal parameter equality does not imply equal
informative capacity. Rank-only intentionally deletes absolute-color information.
1,230independent count-CDF histogram and1,536patch-profile checks are exact.
The16strict-monotone input probes have zero histogram change; channel mixing and
clipping do change it. This is not arbitrary-camera invariance or an accuracy test.

## Verification

108color and108risk arrays exactly replayed;
6336independent scalar cases/216coverage rows checked.
Fitting-only target scales, identical per-seed initializations and same-camera epoch
selection verified. TEST/CAL and UMINHO held-out endpoints were not used.

[Protocol](../../research/skin_copula_protocol_v1.md), [audit](audit.json),
[decision](../../research/skin_representation_next_decision.md).
