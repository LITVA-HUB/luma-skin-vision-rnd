# HR runtime WDDM host recovery v1

The actual original runtime session 54282 terminated with exit 1. No response or recipe receipt was created. Its unchanged `full_runtime_v1.log` records rejection of 23 GPU-context PIDs before CUDA initialization or timing. The primary session 13677 and independent audit session 81929 previously terminated with exit 0. The completed audit SHA-256 is `283dad925b9f8ed36b269418fc935854d0fb3b2d396e26e4f0b760e0fb9e90df`. No quality choice is reopened.

## Diagnosed cause and scope

On this RTX 4060 / Windows WDDM / 610.88 host, the NVIDIA compute-apps query includes Explorer, DWM, Task Manager, browser and other desktop application contexts. Current processes were inspected using CIM. Two actual one-second GPU Engine samples showed approximately 0.3% per DWM/Codex 3D process and no nonzero compute engine activity. A context's existence is not a measurement of sustained competing training. NVIDIA documents C+G as both context types, not as a pure-graphics exemption: [NVIDIA-SMI](https://docs.nvidia.com/deploy/nvidia-smi/). Windows engine accounting covers CUDA as well as graphics APIs: [Microsoft DirectX engineering](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/).

This supplement changes only the host gate and adds execution provenance. All original sources, training, predictions, metrics, INNER selections, 0.002/1e-6 tolerances, response timing function and complete recipe function remain unchanged. Ordinary desktop applications are not closed. This is a local desktop benchmark, not a claim of fully isolated hardware.

## Host rule fixed before any runtime measurement

Freeze a baseline snapshot, code/tests/protocol, successful targeted checks, initial failure log, original source/verification locks, audit, selection and results. The GPU UUID and driver must match thereafter. Permitted desktop instances have an explicitly inspected name plus exact PID and process creation time. Only the finite desktop name list in the new guard can be registered. A disappeared desktop instance is acceptable; a replacement PID or unknown GPU context is rejected. Names alone do not grant an exemption. The currently executing runtime PID is the only GPU process excluded from activity totals.

Reuse the original Python research-worker/ancestor check without relaxing its unknown-command rejection. A regression test additionally demonstrated that the old substring rule missed `skin_face_transfer_run.py`; the new gate rejects every other Python `skin_` research invocation, including that entry point. Collect two distinct one-second `GPU Engine(*)/Utilization Percentage` samples before setup and before each of the 189 responses and 42 full recipes (232 checks). Store all counters, statuses, timestamps and process metadata. Require valid finite counters and compute-engine coverage. Sum non-current-process utilization by physical engine across processes: each engine must be at most 10%, and each compute/CUDA engine at most 0.5%, in both samples. A new active counter PID also requires registered identity even if NVIDIA omitted it. Missing/error telemetry is a refusal, not zero activity. No retries or threshold adjustment follow from a failed production gate.

The checks precede each timed operation; they do not establish continuous isolation throughout a recipe. Sampling and verification overhead are outside the existing response/build timers and are separately recorded; invocation wall time includes applicable sampling overhead. The first failed invocation's complete elapsed time was not recorded and remains null, not zero. No training-cost or latency improvement is inferred from changing this gate.

## Adapter and evidence

The new host adapter replaces `runtime.require_quiet_host`, augments returned source bindings and receipt binding metadata, and augments the final runtime artifact inventory. It does not replace `response`, `replay`, `fit`, `upstream_fit`, `Predictor`, `predict_torch`, `compare` or `exact`; object identities are checked. Each response/recipe receipt includes the supplemental protocol hash and effective sources. Every saved successful snapshot, launch record, source, test, protocol and failure-log binding is included in the final runtime artifact map, which the unchanged final seal and downstream P3 code already verify transitively.

Run exactly one adapted runtime attempt. Existing successful receipts or another launch prevent this zero-measurement recovery from being reused. A failure preserves all evidence and requires a separately justified recovery version; it never silently reruns training or rewrites a measurement. A separate `verify` must replay all 232 host decisions and check bindings after actual runtime exit 0 and confirmed dead PID. `report` first performs that check, then uses the original report writer with an appended disclosure of the WDDM conditions and an additional binding to the completed runtime-attempt receipt. Original numerical tables are preserved.

Commands are separate actual processes using the existing R&D Python, UTF-8 and OMP/MKL/OPENBLAS threads 1:

1. `scripts/chromaseed_head_range_host_recovery.py freeze`
2. `scripts/chromaseed_head_range_host_recovery.py run`
3. After actual terminal success and dead PID, `scripts/chromaseed_head_range_host_recovery.py report`

P3 and Seg2 remain pending the full HR seal. They import the original stricter host gate too; their eventual launch needs its own recorded use of the frozen new guard, without altering their already registered numerical sources. This supplement does not silently patch later stages or authorize concurrent jobs.
