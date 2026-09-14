# ChromaSeed-K implementation plan

Autonomous experimentation is explicitly authorized. The user goal remains broader than this series.

1. Write independent tests for RKHS/Nyström algebra, nested landmarks and residual bounds.
2. Implement compact predictors and genuine incremental inference with separate teacher-approximation diagnostics.
3. Implement reusable fit banks, inner-only choices, final refits/evaluation and the bounded blend control.
4. Verify synthetic CPU/GPU performance and source-lock the real-data run.
5. Execute the full registered original-TRAIN-only study.
6. Audit independently, measure actual model latency/fit cost, document candidate quality/size tradeoffs and any unresolved limitations.

All six steps completed2026-09-13. Actual primary run `experiments/runs/chromaseed_kernel_v1`, complete workflow replay `experiments/runs/chromaseed_kernel_replay_timing_v1`. Independent audit passed,150 selected model execution paths verified,11 numerical tests passed,300 selected/prediction archives replayed with zero array drift. Report: `docs/benchmarks/chromaseed_kernel_v1/report.md`; next decision: `docs/research/chromaseed_kernel_next_decision.md`. This is progress toward the broader active goal, not goal completion. No study process remains.
