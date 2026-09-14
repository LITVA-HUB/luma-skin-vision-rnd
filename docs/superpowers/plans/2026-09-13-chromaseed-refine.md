# ChromaSeed-R implementation plan

User explicitly requested autonomous choices and batch experiments. The prospective protocol is the design record; no further design approval is needed.

1. Write independent numerical tests for bank optimization, model isolation, gates and halting.
2. Implement fit-only preprocessing/analytic anchor and five small network banks.
3. Implement resumable inner-fit, selection, final-fit and held-evaluation stages with source locks and row hashes.
4. Verify CPU tests and CUDA synthetic throughput, then freeze before real data fits.
5. Run the registered 8192-update bank study; monitor meaningful progress, record resources and failures.
6. Independently recompute metrics/selection and measure model-only inference with actual stopping.
7. Publish only local aggregate reports and model cards; preserve negative findings and keep the broader research goal active where requirements remain unmet.

All seven steps completed on 2026-09-13. Additional storage study: 90 evaluations and a reproducible stopping-threshold diagnosis. Final relevant suite: 65 passed. See `docs/research/chromaseed_refine_next_decision.md`; this implementation plan is complete while the broader research goal remains active.
