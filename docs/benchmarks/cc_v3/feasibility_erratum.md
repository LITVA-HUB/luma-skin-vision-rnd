# Initial two-epoch feasibility artifact

`experiments/runs/ccv3_frame_feasibility_s17` is a declared two-epoch source-only GPU feasibility run. Preserve its original source snapshot, weights and history. It is not a full architecture comparison.

Independent runner review found that its untrained epoch0 history serializes three **unmeasured training losses as0.0**. These fields must be interpreted as NOT MEASURED, not perfect loss. The initial validation predictions were actually computed; the validation-based checkpoint rule and measured GPU feasibility are unaffected. The runner now writes null for epoch0 training summaries, verified with a failing-first regression, before any full120epoch run.

Measured feasibility only:1,215,535parameters; best validation reproduction4.34427° at epoch2 versus4.46980° untrained; validation support100%; peak PyTorch allocated643.09MiB including233.45MiB cached source pixels/GT. This short run supplies no independent generalization, calibrated risk or architecture superiority claim.
