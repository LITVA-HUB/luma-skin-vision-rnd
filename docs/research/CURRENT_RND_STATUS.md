# Current R&D status — public real-data milestone, 2026-09-10

**Real public-ground-truth experiments now exist. Special-mixture advantage is NOT ESTABLISHED. Physical facial skin accuracy remains NOT MEASURED.**

The proprietary instrument-paired facial dataset is unavailable under the user's hard constraint. It is not a blocker for this public-component track. Historical synthetic state is preserved at commit `2685bf0`/tag `milestone/synthetic-only-2026-09-10` and [its status snapshot](SYNTHETIC_MILESTONE_STATUS.md). Proposed's synthetic loss to A2 remains in [negative results](negative_results.md).

## Implemented and measured

- Original-license audit: SimpleCube++ 2234 real images, CC BY4.0; publisher checksum verified. Sony 30 author-redistributed INTEL-TAU pilot, upstream CC BY-SA4.0, evaluation only. No restricted data or third-party pretrained weights used.
- Linear 16-bit loader, black/saturation handling and target exclusion. Publisher splits and capture-date-disjoint estimator/validation/risk/calibration groups. Data hashes verified before evaluation.
- Gray World, Max RGB, Shades of Gray, Gray Edge, standard MobileNetV3-small C/C+, matched mixture Proposed, context/disagreement/combined risk-head ablations. Three fixed seeds for official split; separate camera-held-out run. Eight 60-epoch GPU trainings total.
- Recovery/reproduction metrics, full risk-coverage curves, six fixed coverages, tails, source-frozen thresholds, grouped bootstrap and hardware records. No manufactured ΔE00.

Official SimpleCube++ test 462: seed-average mean reproduction error 3.573° Shades of Gray,2.166° C+,2.085° Proposed. At 80% diagnostic coverage:1.678° C+,1.556° ordinary estimator with combined risk features,1.586° Proposed. Mixture gain is not established; paired confidence intervals include no improvement.

External Sony 30: Gray World 4.073°, C+8.859°, Proposed 5.520°; failure to beat the strongest comparator on a different sensor. 550D→600D: Proposed 2.848°/risk 80 1.891° versus C+2.983°/2.637°, but same sensor type and one seed limit interpretation. Full evidence: [public report](../benchmarks/public_benchmark_report.md).

## Not validated

The official random train/test has capture-date overlap. Full INTEL-TAU archive access remains unavailable; the external pilot is small and author-selected. Source calibration does not guarantee acceptance reliability under shift. No instrumented skin Lab, skin-specific repeatability, cosmetics outcome, arbitrary smartphone ISP robustness or patent novelty is validated. Existing facial API remains conservative.

Public-model quantization/TensorRT/ONNX optimization was deferred because special-mechanism superiority is absent. Historical synthetic export is preserved, not relabeled as public-data deployment evidence.

## Next R&D decision

Retain the ordinary compact illuminant estimator plus combined risk features as the reference. Deprioritize the mixture. Obtain an original-provenance, modest-size, scene-disjoint multi-camera subset from cleared INTEL-TAU/other permissive data; test camera-independent rejection and physically grounded surfaces where actual references permit. Do not tune the reported test sets. Continue public-data R&D; facial instrument validation remains a later, separate gate.
