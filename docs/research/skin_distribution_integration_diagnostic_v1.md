# Post-hoc integration sensitivity diagnostic

After all36frozen fits, order2/order3 inference differs enough to question
numerical approximation. Keep all weights, the original candidate set and the
primary results unchanged. This diagnostic is not a new trained method and is
not selected as a replacement primary endpoint.

For all18Gaussian/MDN fits, evaluate the same candidates with a fixed scrambled
Sobol Gaussian rule: seed71131,2048three-dimensional points plus their negatives
(4096nodes/component); nested512points plus negatives (1024nodes/component).
Components are integrated separately and combined with their predicted weights.
No true Lab participates in decisions. Predictions/risks are evaluated against
the already-used source references only after inference. No new held-out access.

Report every fit: color error at full and all fixed coverages; candidate-choice
changes; expected-risk differences between rules; quadrature regret of the old
order3decision under the4096rule. Neither Sobol level is an exact-integral or
continuous-optimum guarantee. No hyperparameter search or density refitting.

Independent checks: unit-weight mass, antithetic first moments, empirical normal
covariance, selected scalar CIEDE2000 point evaluations and final full/coverage
aggregations. Preserve evidence if more accurate integration still loses.
