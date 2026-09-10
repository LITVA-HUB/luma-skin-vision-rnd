# Technology overview — technical draft, 2026-09-10

Luma Selective Skin Color Engine is a standalone R&D project for non-medical regional facial color measurement from smartphone photographs. The intended output is protocol-defined CIELAB D65/2° plus a calibrated decision to accept, request another capture, or declare the input unsupported. It does not identify people, infer ethnicity, diagnose disease or establish cosmetic undertone.

The scientific problem is ambiguity caused by lighting, camera spectral response, ISP processing, geometry and unusable skin regions. The proposed investigation combines region measurement suitability, bounded correction hypotheses and held-out residual prediction. Smartphone colorimetry, regional calibration and rejection already exist; the narrow candidate contribution requires measured improvement over matched alternatives. See [prior art](../research/prior_art.md) and [novelty hypothesis](../research/novelty_hypothesis.md).

Current maturity: local prototype infrastructure and synthetic engineering verification are tracked in [current status](../research/CURRENT_RND_STATUS.md). Real instrumented data, repeatability, color accuracy, selective utility and compact production performance remain NOT MEASURED. Synthetic output is not technical efficacy evidence.

The eventual integration is a separately versioned engine/API consumed by Luma before explicit user confirmation. The original application is PROVIDED CONTEXT, not independently verified source. Foundation/concealer recommendations also require measured product colors and user-outcome validation beyond skin measurement.

Official Foundation criteria cover innovation, commercial potential, theoretical feasibility and team competence. This draft organizes evidence around those criteria; it does not establish approval or eligibility. [Official applicant process](https://sk.ru/applicants-actions/), checked 2026-09-10.
