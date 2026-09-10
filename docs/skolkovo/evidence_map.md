**Current CC v2 update:** [camera-transfer evidence addendum](cc_v2_evidence_addendum.md) records measured matched-C+ improvement, source/Canon failures, competitive cheap controls, ONNX parity and hardware evidence. Facial colorimetry and novelty remain unvalidated. Earlier milestone text below is preserved as history.

# Public-real-data update

The [V5/phone progress addendum](cc_v5_phone_progress.md) records paired training
verification, the negative matched-query recurrence test, phone data rights and
reference preparation. Its development-only results must remain separate from
the V2 independent camera-transfer evidence and future facial validation.

The active maturity/evidence map is [public_evidence_addendum.md](public_evidence_addendum.md). Real illumination-reference experiments now exist; physical facial color is still unvalidated. Historical synthetic-stage mapping below is retained, not the current blocker list.

---

Historical synthetic/facial-stage material follows.

# Technical evidence map — 2026-09-10

The [official applicant page](https://sk.ru/applicants-actions/) describes formal completeness review, innovation-priority review and substantive expert review. Its substantive criteria are innovation, commercialization potential, theoretical feasibility and relevant team competence. The map below is our evidence plan, not a statement that any criterion has been satisfied.

| Reviewer question | Current evidence | Missing evidence / completion condition |
| --- | --- | --- |
| What technical property could differ from prior art? | Targeted primary-source comparison in research/prior_art.md; narrow provisional mechanism in novelty_hypothesis.md | Full closest-paper extraction, patent search; no novelty conclusion yet |
| What experimental result supports the difference? | Synthetic pipeline tests only, identified as such | Locked instrumented test set, best matched C+, equal-coverage ΔE00/tail risk with subject-level confidence intervals, ablations and device/light holdouts |
| Is the project physically plausible? | Explicit bounded operating domain and abstention; documented reference protocol | Instrument repeatability, registered cheek measurement accuracy, failure cases; unknown-JPEG inverse ambiguity may impose irreducible error |
| Does the prototype execute with available resources? | Repository commands and actual logs referenced by CURRENT_RND_STATUS.md | Real training/benchmark runs; 8 GB target verified with measured memory; no borrowed paper latency |
| Can a compact engine preserve useful behavior? | Modular train/export/inference interfaces | Native/export equivalence and task accuracy after FP16/INT8, recalibration where needed; compactness target is not evidence |
| Is there a concrete customer problem? | Cosmetics recommendation quality hypothesis and intended integration contract | Founder-supplied user problem evidence, willingness to pay, partner interviews and measured impact |
| Can the team execute? | Required roles identified below | Names, employment/contract relationships, relevant track records and time commitments supplied by founder; no biographies invented |
| Can the company legally commercialize artifacts? | Separate code/weight/data inventory and IP map | Exact artifact clearance, contributor rights, participant consent/privacy review and legal entity documents |

Required roles: CV/ML research engineer; colorimetry specialist; data-collection coordinator; backend/product integration engineer; commercial lead; legal/IP support when needed. One qualified person may cover multiple roles, but ownership and availability must be documented.

Evidence package index: `docs/research/`, `docs/data/`, `docs/benchmarks/`, `docs/architecture/`, `docs/ip/`, lockfile, immutable run manifests, split/data hashes and test outputs. Protect private photographs; provide aggregated results and controlled reproducibility access rather than public raw-face release.

All final efficacy fields are NOT MEASURED: held-out participants N, camera pipelines M, illumination conditions K, proposed/baseline ΔE00, relative change, confidence interval, production parameter count, model size, batch-1 latency and VRAM. Engineering smoke measurements must remain labeled SYNTHETIC and cannot populate efficacy fields.


## 2026-09-11 phone and broader-search evidence

The first source-only Samsung/Oppo evaluation is measured, including a documented
HDF5 name repair and preserved original results: [report](../benchmarks/phone_v1_alias_report.md).
V2 remains stronger than V5 on phone selective risk; no universal/skin claim.
[Mechanism-search ledger](../research/aggressive_search_2026_09_11.md) and
[V6 execution](../research/cc_v6_execution_status.md) preserve combination failures
and investigate non-CNN Fourier regression and correction decision sets.
