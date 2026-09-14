# ChromaSeed-W: decision after expanded regularization search

2026-09-13. **Verified progress, broad goal active.** Previous P progress was checked against its manifest/receipt and current terminal process state. W fitting, final audit and real runtime measurements have now completed. No W process remains. Improved real ordinary-phone facial quality is still unproved, so this is not full goal completion.

## Evidence that changes the next action

Expanded alpha0.0001–10 while preserving every P numerical operation and all previous alpha controls.8 748 readouts in12 shared banks,26 608 coupled solves including24 664 iterative,112.58s main workflow. All2 916 P controls independently reproduced exactly.26 numerical tests and the analytic-tensor/SVD audit pass.

All15 inner family scores improved, but13 corresponding outer scores worsened. Normalized W gives5.5447/9.5273/8.9189 instead of P5.4387/8.5970/8.7050. W shared weights5.4995/9.5260/8.6863 also lose to P shared5.3901/8.6548/8.3869 in all roles. A few corrections beat the newly weakened W reference, which must not be presented as superiority over previous stronger methods.

Chosen penalties are no longer on the lower boundary:0.001/0.003/0.01/0.03. Width also shifts to2 in mixed/forward. Both iterative families now select positive steps; forward fixed correction selects4, midpoint forward/reverse selects16, but forward midpoint executes5 solves before rejection. Extra training alone does not yield the requested quality. Reverse midpoint111.33ms fit versus P shared20.50ms from its earlier timing, without a quality win; these timings came from separate runs on the same host. All models remain20 284 numeric bytes and approximately16.4µs for a prepared-feature CPU query.

## Next useful question, planned and not launched

Do the internal winning settings depend strongly on individual fit-side people? Diagnose this on saved person-held-out OOF predictions before another fitting search: leave one person out of the selection scores, resample people, measure configuration/rank switches and uncertainty of paired inner score gaps. Distinguish selection instability from a systematic shift between cameras/people. Seed changes must not stand in for independent people.

Write a separate diagnostic protocol before calculation. Keep this W source lock, choices and outer scores immutable. Do not pick a new conservative rule or threshold by scanning the now-exposed W outer table. If a subsequent selection-rule experiment is justified, register its rule and evaluation order separately. No new model or promotion follows merely from a favorable retrospective diagnostic.

This direction addresses model selection/generalization, not computational speed: compactness and prepared-feature latency are already measured. Ordinary-phone facial evidence remains unavailable here. Useful local diagnostics remain, so the goal is not blocked; it also is not achieved. The full goal still requires high real quality, not just low internal error and rapid optimization.

## Resume facts

Run `experiments/runs/chromaseed_weak_ridge_v1`. Training session46551/PID33824 terminal exit0. Initial audit60251 terminal exit0; it was then extended with independent exact checks of all P control arrays and rerun as76553, also exit0. Final audit includes all those checks. Runtime returned exit0 directly, after audit completion; no competing heavy jobs.

Source SHA256 `c16e607e05cc9750ccc3275c16d482c2fb863c693d920ab366b0e0200c1d33e8`; selection SHA256 `c455b980d26f39ec6f5230819aaf06eead19332b9ff44d455029827fbf13cef4`. Preserve W core/train/test/protocol and all inherited P/KE/KF/R/palette locks. No delegation, publication, data acquisition or old held-out partition use occurred. Only original TRAIN; no images/tokens loaded.

[Report](../benchmarks/chromaseed_weak_ridge_v1/report.md) · [Verification](../benchmarks/chromaseed_weak_ridge_v1/verification.json) · [Card](../architecture/chromaseed_weak_ridge_model_card.md) · [Goal ledger](chromaseed_active_goal.md).
