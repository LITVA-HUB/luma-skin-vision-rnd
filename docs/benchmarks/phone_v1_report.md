# First locked Samsung/Oppo transfer benchmark

**Measured custom public-data transfer screen. The newer V5 physical critic loses to the older V2 anchored model on phones.**

All30 methods were fixed before decoding reserved test inputs. A checkpoint-container loader bug was corrected with a separately frozen receipt before any model prediction/error; original executable and lock remain preserved. All results below are REPRODUCED LOCALLY, not author-reported numbers.

Dataset: Beyond RGB original CC BY4.0, Samsung Galaxy S21 Plus and Oppo Find X5 Pro,44 paired official field TEST scenes/88 NT images. Train source: real SimpleCube++ only. Reference policy retained72/88 inputs:37 Samsung and35 Oppo;16 unscorable references are NOT successful model refusals. Nine are strict HDF5-key rejections and seven fail gray-reference quality. Two of72 scorable rows also have unreadable NT keys and are mandatorily refused by every model: the full-population mean includes finite neutral fallback diagnostics, NOT accepted100% operation. Actual maximum coverage is70/72; separate alias-format investigation is required. No estimator saw phone GT, camera metadata, WT chart capture or spectrum. Single quantized demosaiced camera-RGB input; not ordinary JPEG/HEIC. No commercial skin accuracy claim.

|Family (mean of3 seeds for CNNs; no ensemble)|Pooled meanР’В°|Pooled risk80Р’В°|Samsung risk80Р’В°|Oppo risk80Р’В°|
|---|---:|---:|---:|---:|
|v2_direct|5.137|4.380|3.782|5.178|
|v2_sog|4.662|3.822|3.899|3.795|
|gw_ridge1|5.798|4.445|4.481|4.242|
|direct_hgb7|5.386|4.599|4.056|5.030|
|v5_point|5.168|untrained|untrained|untrained|
|v5_posterior_random|4.949|4.861|4.285|5.442|
|v5_action_random|4.913|4.731|4.092|5.358|
|v5_transport_random|5.159|4.971|4.332|5.556|
|v5_transport_policy|4.993|4.807|4.252|5.457|
|v5_transport_gradient|4.962|4.927|4.567|5.267|
|gray_world|6.568|5.834|5.849|5.948|
|max_rgb|13.843|12.656|13.124|12.171|
|shades_gray|5.147|4.915|4.693|5.146|
|gray_edge|5.263|5.125|4.543|5.710|

Pooled V2 SoG risk80 improves12.8% versus matched direct C+, but Samsung regresses. The paired scene-bootstrap95% difference interval is[-1.533,+0.378] degrees versus C+ and[-1.103,+0.168] versus GW+ridge. Both include zero: no statistically conclusive dominance from this small, filtered screen. V5 source-development risk gains do not transfer to these phone captures.

Nominal80% means57/72=79.17% of scorable inputs, but only57/88=64.77% of all planned inputs. Per-device accepted counts are29/37 Samsung and28/35 Oppo. Reported source thresholds below are separate: they were never fitted to phones.

|V2 SoG fixed coverage|Mean errorР’В°|Accepted/scorable|Accepted/planned|
|---|---:|---:|---:|
|100%|4.328|70/72|70/88|
|95%|4.284|68/72|68/88|
|90%|4.037|64/72|64/88|
|80%|3.822|57/72|57/88|
|70%|3.714|50/72|50/88|
|60%|3.716|43/72|43/88|

Frozen source-calibrated nominal80% thresholds (actual phone coverage):

|Method|Accepted/scorable|Actual coverage%|MeanР’В°|
|---|---:|---:|---:|
|ccv2_direct_large_g0_s17|44/72|61.1|4.348|
|ccv2_direct_large_g0_s29|60/72|83.3|4.418|
|ccv2_direct_large_g0_s43|46/72|63.9|4.386|
|ccv2_sog_large_g0_s17|31/72|43.1|3.695|
|ccv2_sog_large_g0_s29|46/72|63.9|3.385|
|ccv2_sog_large_g0_s43|39/72|54.2|3.566|
|gw_ridge1|34/72|47.2|3.949|
|direct_hgb7|37/72|51.4|4.486|

Reference exclusion reasons: `{"Fewer than two usable gray references": 7, "ValueError: Unexpected camera HDF5 key": 9}`. Scenes grouped by number of scorable phone views: `{"2": 34, "1": 4, "0": 6}`. Quality filtering is an engineering policy locked on TRAIN loader captures, not an author official evaluation protocol or an instrument uncertainty guarantee. Separate WT/NT capture illumination consistency remains an assumption.

Independent FP64 cosine/arccos reconstruction checked576 full/coverage means; largest discrepancy1.94e-13 degrees. All90 per-device/method records, tails, recovery metrics, complete risk curves, frozen source thresholds and every seed are in results.json/all_methods.csv. V5 point-only has no trained selective head.

No phone-specific latency measured. Reused V2 measurements:3.034M parameters,12.324MB checkpoint,RTX4060 model-only4.181ms at128px,training762MiB and inference28MiB PyTorch allocations; original timing scope preserved. V5:3.097M parameters,12.599MB checkpoint; V5 deployment latency remains NOT MEASURED.

For Luma: real evidence for the photometric normalization/reliability component on two previously unseen phone models, with explicit failure cases. It does not validate facial skin color, DeltaE00, iPhone, arbitrary ISP pipelines, clinical/cosmetics outcomes or production-safe calibration. TheSkolkovo innovation wording remains a candidate positioning statement.

Next: V6 factorial canonical-frame+physical-evidence experiment, declared before phone errors and trained on source development only. Preserve these phone outputs as observed evaluation, not a model-selection set. Reserve another disjoint acquisition for future confirmation. In parallel investigate a histogram/Fourier representation without a spatial CNN and decision sets instead of a single illuminant; neither is claimed novel merely by implementation.

Reproduction: original method_lock.json plus loader_repair_v1_1.json bind artifacts. Run cc_phone_benchmark.py prepare using the lock on a fresh cache, then cc_phone_benchmark_v1_1.py predict/evaluate with --lock-sha256 f47778497a76782da288fe6740d8ef12388fa080ead9a08db618b7a6de393021. Outputs refuse overwrite. cc_phone_report.py independently verifies means and creates this report. Data acquisition and reference protocol remain separately documented.
