# Equal-weight source combinations

Post-discovery source exploration. Fixed same-seed two-model nativeLab averaging;
no tuned weights. Three seed scores averaged below, not a six-model ensemble.
All members use their own same-camera validation checkpoint. No independent test.
Two networks cost about twice one network; no ensemble novelty claim.

| Protocol | Pair | MeanDeltaE00 | At80% | p95 | Seed means |
|---|---|---:|---:|---:|---|
|mixed|mixture_graph_always|3.4043|3.4411|6.8769|3.3035, 3.4913, 3.4181|
|mixed|mixture_graph_drop|3.3694|3.3123|6.7629|3.3232, 3.3843, 3.4007|
|mixed|mixture_conv_drop|3.3855|3.3626|6.9525|3.3383, 3.3942, 3.4240|
|mixed|mixture_plain|3.3680|3.2749|6.7946|3.3497, 3.3820, 3.3723|
|mixed|plain_graph_always|3.4136|3.3561|6.9739|3.3620, 3.4608, 3.4181|
|from_SLR|mixture_graph_always|5.0697|5.3043|9.9307|4.4972, 5.3626, 5.3494|
|from_SLR|mixture_graph_drop|5.0123|5.2217|10.0444|5.0950, 4.8850, 5.0570|
|from_SLR|mixture_conv_drop|5.2698|5.3972|10.4060|5.2046, 5.4771, 5.1278|
|from_SLR|mixture_plain|5.1702|5.1650|9.9795|5.2431, 5.2800, 4.9874|
|from_SLR|plain_graph_always|5.2237|5.4085|10.2161|4.8164, 5.6897, 5.1650|
|from_ipod|mixture_graph_always|5.2806|5.4846|10.0112|5.4262, 5.4201, 4.9955|
|from_ipod|mixture_graph_drop|5.7810|5.8445|10.4380|5.7219, 5.8938, 5.7272|
|from_ipod|mixture_conv_drop|5.7114|5.7515|10.2254|5.4546, 5.9353, 5.7442|
|from_ipod|mixture_plain|5.6523|5.6836|10.1399|5.4673, 5.9243, 5.5652|
|from_ipod|plain_graph_always|5.1284|5.3470|9.4521|5.2316, 5.1516, 5.0021|

80%ranking uses uncalibrated inter-model disagreement; it may worsen error.
This is not a learned expected-error head or coverage guarantee. All45combinations,
270fixed-coverage rows and independent scalar checks are retained in [results](summary.json).
[Protocol](../../research/skin_branch_combination_protocol_v1.md),
[decision and limits](../../research/skin_spatial_next_decision.md).
