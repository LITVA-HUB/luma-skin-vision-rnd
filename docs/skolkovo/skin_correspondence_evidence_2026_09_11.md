# Real skin-color diagnostic and falsification evidence

Implemented and measured locally: source-only triplicate reference audit on
314 sites, 27 frozen model endpoints, and 36 paired-objective training runs.
Every trained system uses one image for inference and scores native Lab against
real instrument references. This is not an illuminant-angle benchmark.

Measured negative finding: weakening paired-view agreement does not produce a
universal accuracy improvement. Increasing agreement slightly improves known-
camera source mean but worsens selective error and one transfer direction.
No mechanism is promoted for deployment or patent novelty based on this cycle.

Reference spread is material, but is not a certified instrument uncertainty or
an irreducible floor. The publication's capture conditions include initial white
balance calibration and contact/polarization modes. Ordinary facial phone use
remains unvalidated. The previous independent test is unchanged and exposed.

Original data rights remain the documented MSKCC CC-BY grant; no additional data
or external model weights were acquired. Training code is a local algebraic
ablation of standard regression/consistency, not an invention claim. No raw
participant images, identifiers or weights enter Git.

[Measured results, opponent check and next decision](../research/skin_correspondence_next_decision.md).
