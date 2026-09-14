# HR verification implementation plan

> **For agentic workers:** Use superpowers:executing-plans. No subagent delegation is authorized.

**Goal:** Implement the complete independent HR audit, actual runtime/construction measurements and final report consumers while preserving the live numerical experiment.

**Architecture:** Shared pure validation/selection/cost helpers and a primary-process guard support independent audit, runtime and reporting entry points. The audit may probe a frozen completed-inner snapshot; full final work requires a terminal successful primary.

**Tech stack:** Existing Python/NumPy/PyTorch environment; CPU verification, later isolated CUDA construction replay.

**Spec:** docs/research/chromaseed_head_range_verification_v1_protocol.md

## Tasks

- [x] Add tests for process gating, stable selection, complete pass comparison, result identity and recovery accounting.
- [x] Implement independent full audit without editing frozen AS/HR/P1/P2 sources.
- [x] Implement runtime and construction replay with exact payload comparisons.
- [x] Implement source-backed report/seal and read-only verification.
- [x] Run helper tests and an explicitly partial completed-inner probe; verify the live-primary guard.
- [x] Freeze new verification sources and document what remains unexecuted until primary terminal success.

Do not create the final verification.json during preparation. Do not run another GPU workload while HR is live. No final quality gain is asserted by completing this plan.

Preparation completed 2026-09-14, while primary session13677/PID42200 remained live. Four new Python consumers, their test file and protocol are frozen by `experiments/runs/chromaseed_head_range_v1/verification_v1/protocol.json`, SHA3c5f4feed96c18e39cc5ec750163425778b9ee2397633bd4f4860dcb44b147d9. All185 original source bindings remain unchanged. Preserve these new bound files.

14tests passed1.73s; lint passed. Tests include an actual child-process lifecycle and CPU-export final-audit path: corrupting an intermediate answer is caught even with unchanged final answers. All three primary entry points reject the still-running HR job before writes; the benchmark guard detects both the primary and its venv wrapper. No psutil/environment dependency was added.

Final partial probe session42012 TERMINAL0,20.338s: four completed inner banks (dynamic5m/wide, dynamic_small/linear, patch5m/linear, soft5m/wide; mixed/fold0),72models/16,416OOFvectors, maxLab7.610323443429934e-6. Receipt verification_v1/probes/e56b1604476bbb77/result.json SHAa98d4e96f54657de1fb40c4517f27c3d1f70d8b532ea0f7663b1f35751891d76; its source bindings match the final frozen consumers. Earlier draft probe a14e3c7b0e98b785 is preserved as superseded development evidence.

Full audit,189CPU timings,42full construction replays and final report/seal are implemented but NOT EXECUTED. After primary tool session13677 actually terminates with exit0 and PID42200 is dead, run audit.py, then runtime.py, then report.py, sequentially under OMP/MKL/OPENBLAS_NUM_THREADS=1. Do not launch the actual full sequence while primary is active. The broad quality goal remains active.
