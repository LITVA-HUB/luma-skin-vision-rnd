# User-authorized capacity increase: at most200,000 neural parameters

LATEST: the subsequent stable39-input correction screen uses188,035 added
parameters /1,117,332 total and no reference bank. Nine encoder andnine head
fits show an internal mean6.1082 ->5.6560 improvement, with camera transfer
pending and OOF-superiority hypothesis rejected. Same1,129,297 cap remains.
[Current result](../benchmarks/skin_crossfit_correction_v1/report.md).

STATUS UPDATE: the candidate head below is now implemented and trained in
skin_neural_reference_v1,27 matched fits /9 frozen cores. Code verifies193,795
added /1,123,092 total parameters, with no universal accuracy win. The earlier
candidate description below records the decision before fitting; its then-
unimplemented status is historical. Current evidence is in
[the measured report](../benchmarks/skin_neural_reference_v1/report.md).

User instruction: "разрешаю увеличить количество параметров на 200 000".
Applied to the current single compact CaptureColor/SkinRepresentation neural
base,929,297 parameters. Next prototype cap:1,129,297 total neural parameters.
This is not permission to enlarge every ensemble member without reporting the
total deployed system. No new accuracy follows from the budget authorization.

The next proposed color adapter and ordinary C+ adapter must receive the same
base representation, training data, initialization policy and training budget,
with approximately equal active parameter counts. Preserve the untouched base
as a third control, and preserve every failed synthetic/real-data result.

A concrete feasible head budget (unverified candidate, not implemented yet):
input551 =512 learned context +36 image descriptors +3 predicted color values.
Layers551->256->192->16->3 have193,795 parameters including biases:
141,312 +49,344 +3,088 +51. Total with current core:1,123,092;6,205 remain under
the authorized cap. An ordinary residual head can use exactly these layers as
C+. A proposed16-dimensional reference-affinity embedding plus3 correction
gates could share the same learned parameter count. Merely changing the role
of these outputs does not establish novelty or an accuracy advantage.

Any reference-bank descriptors, cached learned embeddings, native Lab values,
scales and non-neural fitted coefficients must be counted separately in stored
scalars, complete file size and runtime. Do not advertise the neural weight
count as the entire deployed payload. No inference camera ID or query reference
is allowed. Avoid consuming budget solely to reach a round parameter count.

Before fitting: confirm exact active counts by code, independent-person bank
construction and equal initial color predictions. Freeze a small matched
source experiment. Error heads still require person-excluded color residuals;
an encoder trained on the held person cannot be disguised as an OOF encoder.
Measure native skin DeltaE00, both camera directions, tails and calibrated
coverage before any production/export claim.
