# Conditional skin-color model provenance

The distribution model, fitting code, quadrature decisions and verification
scripts were implemented locally in this repository. They build on the existing
local CaptureColor backbone. No external model weights or implementation code
were incorporated in this branch.

Underlying concepts are established: mixture density networks (Bishop1994) and
conditional risk minimization (Goh/Jaillet2016-2017). Citations are recorded in
the [research review](../research/prior_art.md). Local implementation does not
establish patent novelty, freedom to operate, or a defensible performance claim.

Training/evaluation uses the original MSKCC CC-BY dataset with its instrument Lab
references. Participant images, identifiers and weights remain local. No new
dataset rights are inferred. This branch does not bundle ISSA/CIE constants;
the preceding material experiment retains its separate CC BY-SA obligations.

Measured evidence is exploratory and negative for universal superiority. The
Gaussian mixed-source risk signal is an unverified candidate for further matched
testing, not an invention claim or commercial accuracy guarantee.
