# Public-real-data evidence for technical project discussion

This addendum supersedes synthetic-only maturity statements in earlier planning documents. It does not establish legal eligibility or approval.

| Question | Concrete evidence | Limit |
|---|---|---|
| Implemented component? | src/luma_skin_vision/cc: loader, estimators, learned error selection, experiment runner | Not integrated into original Luma app |
| Real ground truth? | Publisher checksum, measured neutral-cube illuminant vectors, frozen 462-image test | Neutral-cast angular error, no physical cheek Lab |
| Does ML add value? | Three seeds: compact estimator 2.166° vs Shades of Gray 3.573° | Official scene/date overlap; generic baseline value |
| Does Proposed beat strong control? | Risk 80: Proposed 1.586° vs control 1.556°; confidence intervals include no gain | NO established selective advantage |
| Camera generalization? | 550D→600D holdout plus 30 external Sony examples | Same-sensor Canon; Sony 5.520° loses to Gray World 4.073° |
| Local deployment plausible? |966,760 stored parameters; training under 1GB allocated; model-only ~3.3ms | Full CPU expert preprocessing is larger; no production SLA |
| Data rights recorded? | Inventory, rights ledger, attribution and manifests | Distribution duties separate; no external release |
| Facial-specific validation? | Future physical protocol and conservative API gates | No measured skin accuracy or cosmetic outcome |

Review package: [benchmark](../benchmarks/public_benchmark_report.md), [protocol](../research/public_protocol_v1.md), [prior art](../research/public_color_prior_art.md), [decision](../research/public_hypothesis_decision.md), [status](../research/CURRENT_RND_STATUS.md). Evidence includes per-image predictions/errors, source/data/checkpoint hashes, eight training histories, ablations, calibration transfer, bootstrap and hardware logs.

Defensible technical statement: the team measured the future engine's photometric normalization/reliability core against real established color-constancy references and identified camera-transfer failures. This does not validate the full facial product or establish innovation/IP eligibility.
