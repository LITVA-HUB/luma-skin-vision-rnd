# Fused AdamW feasibility while AS trains

Status: CPU-only synthetic feasibility completed; GPU speed/capture and production training remain untested. Separate from the frozen AS architecture screen, which continues unchanged on RTX4060. User authorization includes efficient training, kernels and autonomous experiments. This prototype is not connected to any primary runner.

Evidence motivating inspection: AS synthetic BF16 did not accelerate the current six-slot implementation. First completed mixed-fold0 six-slot2048-step banks took156.14s (patch5m) and444.08s (soft5m) with roughly equal parameter counts. These observations do not establish which kernel dominates time. Existing BankAdamW performs separate elementwise clipping, moment, decay and parameter-update operations. A repository search across scripts/tests/research found no fused AdamW implementation. Installed PyTorch2.8.0+cu128's torch.optim.adamw functional API exposes fused=True; its local optimizer sources list CPU and CUDA support. Availability is not speed evidence.

Bounded plan:

1. Write CPU synthetic tests for independent slot rates, clipping, moments, decay, zero gradients and exact reset/replay before implementation.
2. Implement a separate adapter using built-in fused AdamW on row views of the unchanged flat bank. Keep the leading bank dimension, parameter storage and batch/gradient computation intact. No packages/compiler, no new data or GPU calls.
3. Compare against independent FP64 AdamW equations and the current optimizer. Record all numerical differences; mathematical equivalence does not imply bitwise equality or equal long-run model quality.
4. Only after AS primary/audit/runtime work is terminal, consider a separately registered GPU profile and matched training/replay experiment. Do not claim acceleration or substitute this optimizer in the frozen AS run.

CPU probes must use small synthetic tensors and one thread. The active AS audit/runtime/report executables remain deferred. Preserve AS source hashes, sampling, schedule, all controls and results. No speed, GPU-capture or model-quality acceptance is implied by CPU feasibility.

Measured CPU evidence: four tests passed in1.59s. An initially missing-module collection failure preceded implementation; the completed suite covers independent FP64 equations, per-slot rates/clipping/decay/moments, exact reset, unchanged parameter storage and an actual four-pass network with coupled gradients. The separate probe passed512 prescribed-gradient steps and64 coupled-model steps. No CUDA context was initialized. Receipt: ../benchmarks/chromaseed_fused_optimizer_cpu_probe/probe.json, with hashes of the prototype/tests/current Bank source and installed PyTorch implementation files.

The fused update is **not bitwise identical** to the original optimizer, even at step1. Maximum parameter difference over512 prescribed steps:4.76837158203125e-7; moment difference5.960464477539063e-8; second moments equal in this probe. In the64-step four-pass network, maximum parameter difference5.960464477539063e-8 and normalized-output difference2.9802322387695312e-8. Original tolerances passed; they were not changed. These finite synthetic probes cannot bound long-run GPU training drift. A future study must explicitly test capture/reset, full-bank reproduction, speed and quality before adoption; it cannot replace AS's frozen optimizer as a transparent drop-in.
