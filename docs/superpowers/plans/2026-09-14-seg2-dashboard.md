# Seg2 dashboard integration

Previous turn: verified progress — Seg2 implementation/registration and real CPU gate completed. Current HR session13677/PID42200 is confirmed live. No second GPU work, CPU latency benchmark, delegation, dependency change or frozen ML edit.

Design follows the already accepted Russian training dashboard. Add Seg2 after P3 and before Seg1. Existing autonomous authorization covers implementation and local dashboard restart; no additional approval is needed. Execute inline with the established brainstorming, writing-plans and TDD workflow.

Flow under test: `/#overview` → open Seg2 details → filter data arm and seed → inspect the six-run queue and validation chart → refresh while preserving the controls.

## Source contract

Read-only standard-library monitor; never import training code, torch, image files, NumPy arrays or weights. Source: `D:/Luma-RnD/skin_face_transfer_v1` registration, preflight receipts, immutable job/completion, mutable progress, per-checkpoint metadata, trajectory receipts, selection and final verification. Observe actual recorded PID; do not treat the immutable running job as proof that training continues. HR and P3 predecessor seals remain required.

Six trajectories, 5,976 updates each, 35,856 total. A run counts as saved only with matching recipe fields, a valid history and its receipt. Fresh current steps must agree with the current trajectory, completed preceding runs, total counts and PID. No extrapolation of steps. Rate/ETA come only from consistent fresh Seg2 training telemetry; CPU preflight cost and old-model speed cannot substitute. Hide estimates on unknown/dead/stale/malformed state.

Show registration counts (LaPa 15,914/1,692/2,000; CelebA 24,112/2,992/2,822), frozen batch32, 4,416,673 parameters, fixed budget windows 1,494/2,988/5,976. Per-checkpoint validation IoU can appear as progress evidence; final held quality only after a matching verification seal and summary. Segmentation/apparent-image color must not become a physical skin-color accuracy percentage.

## Implementation

- [x] Add `apps/training-dashboard/face_transfer_monitor.py` and tests covering queue/gates, paired run counts, PID/state checks, malformed input, step/ETA constraints, validation metadata and post-training stages. Eighteen new cases, including a reproduced/fixed malformed nested preflight failure.
- [x] Add a consistent responsive `FaceTransfer.jsx`, a local CSS file, source counts, four progress metrics, six-run filters and per-source validation traces. Uses the existing API polling without another dependency.
- [x] Wire `/api/status` and the overview; document source/metric semantics in README.
- [x] Relevant tests: 41 passed in 1.06s; targeted ruff clean. Final Vite build: 29 modules, JS281.10kB/84.06kB gzip. All 63 Seg2 and 283 P3 bindings rehashed unchanged; Seg2 production job absent.
- [x] Actual bundled Playwright1.62.1 / installed Edge headless, GPU disabled: 13 flow/state checks passed at 1440x1100 and 390x844, no console errors/overlays/page overflow. Filter6→3→1, selected graph/expanded section survive refresh; old P3 queue72 and navigation work. Ten future states use isolated network fixtures. Screenshots inspected, including actual waiting desktop/mobile and clearly labeled fixture chart. QA evidence outside repo: `C:/Users/dimal/.codex/visualizations/2026/09/13/01a0982c-a74f-7ff0-b456-128a38a117c6/seg2-dashboard-qa/checks.json` and companion PNG/check.cjs.
- [x] Confirmed old listener37504 by exact command path, restarted only the dashboard. Actual new listener6276/wrapper25748; HR42200 remains live. Actual API: Seg2waiting_hr,CPU6passed,0/6,40026TRAIN,no warnings. LatestHR114/168,ETA9907.9s (~2h45m, excludesverification). Opening browser panel was queued by the app for this task; actual rendered checks used Edge independently.

No model quality or full objective completion claim is part of this dashboard change. Native HR/P3 and Seg2 CUDA/training/evaluation remain pending.
