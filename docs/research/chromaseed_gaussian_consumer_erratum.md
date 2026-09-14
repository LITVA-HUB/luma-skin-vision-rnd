# TG consumer interface erratum

2026-09-13, found during NR preflight before NR source freeze. The sealed TG model card says "One row or a batch is accepted". That is incorrect for `chromaseed_gaussian_numpy.Predictor.__call__`: it accepts exactly one finite color36 vector with shape(36,), returning Lab3. The separate `chromaseed_gaussian.predict(model,x)` batch helper accepts N×36. To batch the actual consumer, iterate over rows.

TG's numerical audit and timing already used the one-row consumer correctly; its weights, errors, timing samples, receipt19a14cd570957c02a1dcf2781169d174b1549efeb1fab915276591fdfde3f098 and frozen sources remain unchanged. NR's initial consumer tests mistakenly passed batches and raised the expected ValueError. Tests were corrected to call the existing interface per row; no implementation behavior or numeric tolerance changed. This separate correction preserves the original sealed artifact and will be bound as an NR input.

[Original TG model card](../architecture/chromaseed_gaussian_model_card.md) · [Actual consumer source](../../scripts/chromaseed_gaussian_numpy.py) · [TG receipt](../benchmarks/chromaseed_gaussian_v1/verification.json).
