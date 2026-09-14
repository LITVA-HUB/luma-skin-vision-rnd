# Luma training dashboard implementation plan

User requested a beautiful, informative Russian web dashboard showing current training speed, remaining time and statistics. Build this in the current task; no delegation, publication or training restart.

**Design:** use the generated full-screen concept `apps/training-dashboard/design/concept.png`. White canvas, neutral pale sidebar210px, emerald accent#178265, charcoal text#17211d, thin borders#e6ebe7. Sans-serif Segoe UI/Arial, headings28–30px, metrics32px, body14px, controls13px. Keep the open four-column metric strip, two-column chart/current panel, package table and quality section. Navigation: Обзор, Пакеты, Модели, Журнал. On mobile collapse the rail into a top navigation and stack chart/current panels.

**Architecture:** isolated React/Vite frontend, Python standard-library HTTP server and read-only telemetry collector. Serve built frontend and API on127.0.0.1:8766. No dependency or source changes to the frozen research environment. No training controls, external uploads, images, datasets or weights opened. Read only JSON receipts, selection/review/results summaries and nvidia-smi.

**Data contract:** `/api/status` returns package schedule, actual completed counts, current estimated progress, duration/ETA estimates, actual saved loss traces, quality summaries with evidence labels, completion events and fresh GPU telemetry. Poll every3seconds with one request in flight. Missing readings are null, never synthetic zero/100%. Check actual process liveness before claiming training is running. The primary writer saves detailed traces only at bank completion; current step is explicitly estimated and never reaches100% before receipt exists. ETA accounts for all63inner+21final banks, architecture-specific measured duration and unknown final checkpoint range. Final-mode choices update when selections.json appears. Loss curves are minibatch objective, not accuracy/ΔE00.

- [x] Add meaningful collector tests for84-item accounting, stopped/stale worker, checkpoint-dependent final duration, no fake current progress and malformed JSON.
- [x] Implement `apps/training-dashboard/monitor.py` and `server.py`, with cached JSON reads and GPU sampling that never imports torch or changes a training process.
- [x] Implement small React modules for shell, metrics, chart/current panel, packages, quality and events; real filters, row selection, chart tabs and CSV download.
- [x] Build, start hidden local server, verify JSON against source receipts and GPU query, and verify UI in IAB; native IAB supported every required check.
- [x] Compare concept and rendered screenshot with view_image, check desktop/mobile, navigation/filter/chart interactions, no console errors, and stale/offline states. Preserve screenshots outside the repo; no temporary QA scripts remain.
- [x] Open the finished local dashboard, mark the browser tab deliverable, and prepare URL plus restart launcher for handoff.

Intentional functional deviations from concept: illustrative numbers/curves/rows become actual data; generated completed-final example is discarded because final training has not started. Add role selectors, complete navigation views and detail controls where needed for real data exploration. Logo is a small accessible vector mark consistent with the reference. No generated image is used as the actual interface.

Verification, 2026-09-14: seven telemetry/CSV tests pass (0.22 s); Ruff passes for all three Python files; Vite production build passes. Browser checks cover all four views, filters/search, package details, chart selection and speed tab, model-role empty states, actual CSV download, and recovery after stopping/restarting only the monitor. The training PID 38808 stayed alive. No fresh browser errors after recovery. The launcher is idempotent and preserves the healthy server PID. Final API health is OK, reports live training and GPU telemetry without warnings; all 172 bound research-source hashes match. No frozen training sources or research dependencies changed.

Visual fidelity ledger, checked with view_image against the generated concept at a 1435 × 1096 desktop viewport and 390 × 844 mobile viewport:

- Composition: retained pale 218 px sidebar, open four-metric strip, two-column training/current panel, packages and model-quality section.
- Palette and surfaces: retained white canvas, emerald active states and restrained separators; fine bordered panels match the reference.
- Typography: 32 px desktop heading and metrics, 14 px table text; responsive hierarchy. The generated illustration's slightly larger text was adapted to the functional selectors and evidence captions.
- Icons and logo: consistent small outline icons and an accessible vector Luma mark; no screenshot used as interface.
- Data graphics: six actual saved trajectories and measured speed intervals replace the smooth illustrative curves. Provisional progress and ETA are explicitly approximate; no fabricated completed-final row.
- Interaction: added selectors, filters, details and CSV without changing the main visual order. The extra working controls and evidence notes extend full-page height beyond the concept.
- Responsive layout: medium widths stack training/current panels; mobile uses a top navigation, two-column metric strip and stacked panels. No page-level horizontal overflow; wide tables scroll within their own region. Temporary viewport override reset before handoff; normal window is 858 × 1318.

Screenshots: `C:/Users/dimal/.codex/visualizations/2026/09/13/01a0982c-a74f-7ff0-b456-128a38a117c6/training-dashboard/desktop.png` and `mobile.png`. Serve `http://127.0.0.1:8766` with `apps/training-dashboard/start-dashboard.cmd`; the monitor is local to this computer.
