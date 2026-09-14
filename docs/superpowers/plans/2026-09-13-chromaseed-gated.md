# ChromaSeed-G implementation plan

Local autonomous model experiment, no delegation. [Protocol](../../research/chromaseed_gated_v1_protocol.md).

- [x] Numerical tests and compact gated readout/core predictor:23 primary tests and3 separate NumPy-only consumer tests pass.
- [x] Frozen168-configuration banks, all inner selection before all final scoring;72 exact P base controls and864 routed fallback copies verified.
- [x] Independent direct/analytic/SVD audit, including72 selected refits and12 active-path probes.
- [x] Actual selected and active-path fit/inference timings:112 complete fits including warmups, all payloads exact; all-family report, model card and final verification.

All primary/audit/runtime/report processes are terminal. Mixed gain is modest and exploratory; known single-camera training roles fall back exactly. The broader goal stays active. [Decision and next gate-stability diagnostic](../../research/chromaseed_gated_next_decision.md).
