# Direct skin color from local pixels: source screen v1

**SOURCE VALIDATION ONLY**:264images,66sites,6people;966training images/24people.
The10-person final test and6-person calibration groups remain unopened. All scores below used for development.
Local input pipeline: verified original JPEG -> central skin crop ->128x128RGB -> model -> native instrument Lab. No camera ID or author-derived color features at inference.

|Model|Seed|Mean DeltaE00|Median|p95|Parameters|
|---|---:|---:|---:|---:|---:|
|cnn|17|3.8139|3.1696|8.0428|1520931|
|cnn|29|4.0029|3.5096|7.9454|1520931|
|cnn|43|4.1147|3.7380|8.6781|1520931|
|votes_mean|17|3.4888|2.9278|7.1438|924932|
|votes_mean|29|3.5074|3.1112|7.2062|924932|
|votes_mean|43|3.4964|3.0540|7.2758|924932|
|votes_huber3|17|3.4892|2.9394|7.1414|924932|
|votes_huber3|29|3.5156|3.0999|7.1849|924932|
|votes_huber3|43|3.4956|3.0529|7.2567|924932|

Three-model ensembles (not single-model deployment):

|Architecture|Mean DeltaE00|Median|p95|Mean at80%, density|Mean at80%, seed disagreement|
|---|---:|---:|---:|---:|---:|
|cnn|3.6152|3.2097|7.1517|3.3885|3.4225|
|votes_mean|3.4588|3.0118|7.1105|3.2426|3.4491|
|votes_huber3|3.4611|3.0028|7.1375|3.2453|3.4417|

All18best/final checkpoints replayed exactly; independent scalar CIEDE2000 audit passed.
All3matched seed initializations are bitwise identical. Correlated patch estimates can render robust iteration inactive; see mechanism diagnostics in summary.json.
Risk scores are diagnostic rankings, not calibrated expected skin-color error. A full matched C+ error-head/calibration comparison remains unperformed.
Two histogram MLP controls hit their fixed optimizer iteration limit and lost simpler controls; warnings retained in controls.json.
No unseen-camera, ordinary phone-face, independent final-test, novel-method or export accuracy claim follows from this source screen.
The patch architecture uses established set aggregation/confidence weighting/Huber mechanisms. Numerical improvement is not itself proof of novelty.
