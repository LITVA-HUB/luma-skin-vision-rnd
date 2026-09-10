# Samsung/Oppo benchmark after documented HDF5 name repair

**All30 frozen source-only methods rerun;79/88 references scorable, all88 NT inputs readable.**

This is an additive file-format repair after v1 results were observed, not a new untouched test. Original v1 locks, predictions and results remain preserved. Original author HDF5 reader takes the first key; the independent repair permits sole camera/MIS names only in verified camera RGB files. All other processing, reference requirements, models, calibration and grouping are unchanged. Previously readable inputs and valid references are byte-identical numerically.

Dataset: Beyond RGB, original CC BY4.0; single quantized demosaiced camera-RGB NT image.37 Oppo and42 Samsung references meet the unchanged gray-patch policy;9 do not. No camera identity, WT capture, spectrum or CCM is a model input. Training was real SimpleCube++ only. No ordinary JPEG/HEIC/iPhone/skin/DeltaE validation. All numbers are REPRODUCED LOCALLY under our custom protocol, not published author numbers.

|Family (means across3 seeds; no ensemble)|Pooled meanВ°|Pooled risk80В°|Samsung risk80В°|Oppo risk80В°|
|---|---:|---:|---:|---:|
|v2_direct|4.838|4.498|3.803|5.210|
|v2_sog|4.367|3.951|4.064|3.817|
|gw_ridge1|4.430|4.311|4.630|3.995|
|direct_hgb7|4.658|4.576|3.971|5.272|
|v5_point|4.899|untrained|untrained|untrained|
|v5_posterior_random|4.740|5.039|4.387|5.838|
|v5_action_random|4.733|4.902|4.232|5.792|
|v5_transport_random|4.947|5.106|4.448|5.914|
|v5_transport_policy|4.775|4.907|4.173|5.893|
|v5_transport_gradient|4.734|5.051|4.526|5.549|
|gray_world|6.240|5.717|5.779|5.372|
|max_rgb|13.907|12.744|13.678|11.495|
|shades_gray|4.806|4.766|4.835|4.880|
|gray_edge|4.900|5.134|4.768|5.707|

Paired scene-bootstrap differences (SoG minus comparator; negative favors SoG):

|Comparator|Full mean95% intervalВ°|Risk80 95% intervalВ°|
|---|---|---|
|v2_direct|[-1.3245567509095824, 0.39106533345816485]|[-1.531731187086911, 0.3219256208148086]|
|gw_ridge1|[-0.4894383691072508, 0.36501096920149434]|[-1.0090260667291566, 0.236300173496546]|

|V2 SoG nominal coverage|Mean accepted errorВ°|Accepted/scorable|Accepted/planned|
|---|---:|---:|---:|
|100%|4.367|79/79|79/88|
|95%|4.228|75/79|75/88|
|90%|4.077|71/79|71/88|
|80%|3.951|63/79|63/88|
|70%|3.742|55/79|55/88|
|60%|3.658|47/79|47/88|

Frozen source-calibrated nominal80% thresholds:

|Method|Accepted/scorable|Actual coverage%|MeanВ°|
|---|---:|---:|---:|
|ccv2_direct_large_g0_s17|50/79|63.3|4.321|
|ccv2_direct_large_g0_s29|69/79|87.3|4.573|
|ccv2_direct_large_g0_s43|53/79|67.1|4.604|
|ccv2_sog_large_g0_s17|36/79|45.6|3.384|
|ccv2_sog_large_g0_s29|50/79|63.3|3.368|
|ccv2_sog_large_g0_s43|45/79|57.0|3.945|
|gw_ridge1|39/79|49.4|3.772|
|direct_hgb7|44/79|55.7|4.312|

Reference exclusion reasons: `{"Fewer than two usable gray references": 9}`. Scene counts by scorable phone views: `{"2": 37, "1": 5, "0": 2}`. All model_refused counts are0 on scorable inputs. Quality filtering does not certify constant light across separate NT/WT captures, and excludes9/88 planned measurements.

Independent FP64 cosine/arccos rescoring checks576 means, max difference1.79e-13 degrees. Complete all-seed, per-camera recovery/reproduction metrics, tails, curves and source thresholds are in results.json and all_methods.csv. Source80% calibration is not target80% acceptance.

No new latency/VRAM measurement. Original V2:3.034M parameters,12.324MB checkpoint,4.181ms RTX4060 model-only at128px;762MiB train/28MiB inference PyTorch allocations under original scope. V5:3.097M/12.599MB, deployment latency unmeasured. This validates a limited normalization/reliability component, not measured facial skin color or universal camera support.

Scientific decision: retain stronger V2 transfer control, preserve V5 failures, and test V6 combination only on source development. Fourier ridge provides an independent representation-control hypothesis. These observed phone captures must not become a validation set for new methods; confirm improvements on a separate locked sample. Novelty and Skolkovo classification remain unverified.
