# ChromaSeed-X Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans locally. No sub-agent delegation is authorized in this worktree. Execute each numerical task with its stated evidence checks.

**Goal:** Measure whether fit-only projected color statistics yield a smaller, faster and accurate trained skin-color component.

**Architecture:** Preserve all frozen A/G dependencies. Add a variable-dimension kernel input projection and NumPy-only predictor; reuse the analytical joint/static readout math. Original TRAIN only, per-person folds,13 representations, all policies frozen before final fitting.

**Tech Stack:** Existing Python3.12.6 environment, NumPy2.5.3/SciPy1.18.1, CPU one thread; no dependency install.

**Spec:** [Registered protocol](../../research/chromaseed_projection_v1_protocol.md).

## Global constraints

Preserve every old source/verification/data binding. No legacy validation/calibration/test, raw image acquisition, publication or agents. The earlier active-goal turn is verified progress. A full quality claim on ordinary-phone faces remains unproved.

## Task 1: mathematically faithful projection and portable predictor

Create scripts/chromaseed_projection.py, scripts/chromaseed_projection_numpy.py and tests/test_chromaseed_projection.py. `projection(z,w,d,tau)` returns fit-only projection arrays/info; `project(model,x)` implements stored arithmetic; `fit_bank(...)` stores all current configurations, `fit_single(...family,seed,representation,alpha)` reproduces one. A `Predictor(model)` consumes one color36 vector without training imports.

- [x] Write toy failing tests for covariance/SVD, normalization, width/operator, raw equality, guarded selection and consumer contracts. Initial collection failed on missing module as expected.
- [x] Implement and pass numerical tests, including actual serialized payload parity and full-fit/bank parity.12primary tests1.66s before freeze; final combined15tests1.72s.

## Task 2: frozen experiment

Create scripts/chromaseed_projection_train.py. Register13 representations and controls, use separate experiments/runs/chromaseed_projection_v1. Freeze59 primary source paths plus exact A/G input bindings before fitting.

- [x] Validate source/parent inputs, run all nine inner banks, reproduce raw controls exactly, freeze24 choices.
- [x] Finish three final banks, then111×33 evaluations. PrimaryPID36304/session37948 exit0;3,744solutions/1,872aliases separated.

## Task 3: independent audit

Create scripts/chromaseed_projection_reference.py and scripts/chromaseed_projection_audit.py plus focused reference tests where needed. Direct kernel and weighted SVD projector checks must not use the primary projector or Gram solver as an oracle.

- [x] Validate every split, control, alias, OOF/policy/final prediction; independently refit72 selected instances+12 positive probes. Audit72869 exit0.
- [x] Explain any numerical disagreements without changing frozen primary sources or weakening quality conclusions. Projection operator-relative3.17e-8; independent native-Lab difference2.29e-5; full36 rotation near-tie documented.

## Task 4: measured cost and reviewable evidence

Create runtime/report/verification scripts and docs/benchmarks/chromaseed_projection_v1 artifacts, architecture card and next decision. Update only mutable status/goal/README/AGENTS after numerical work.

- [x] Measure111 actual consumers and112 complete parity-checked fits with no heavy concurrent worker. Runtime exit0.
- [x] Report all policies and adverse outcomes; inspect figures and run relevant final checks. Final verification receipt binds completion once; read-only rechecks preserve it. Full goal remains active.
