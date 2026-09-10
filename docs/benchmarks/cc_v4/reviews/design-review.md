# Independent V4 scientific design review

Reviewed 2026-09-10. Scope: scientific/novelty/physics claims and meaningful controls; no model-code review, implementation, training, data access or performance verification.

Reviewed snapshots:

- `docs/research/cc_v4_universal_method.md`, SHA256 `bfaa6e2c60dad9bfc1162fc92d1b7eb2305d889ec00ebd2bdc8825862c680183`.
- `docs/research/cc_v4_spec.md`, SHA256 `5bbce09aee93c0af60cb2ab1bab7ae145208434b61ff440b6931b242b05bcccf`.

Verdict: **one load-bearing mechanism/control issue in the reviewed snapshots; accepted design amendment resolves it conceptually, pending integration into the executable contract. One additional control is needed before attributing gains to sequential refinement.** No scientific objection to proceeding with a properly labelled source-only development screen after the amendment.

## 1. Original log-chroma transport is exactly an affine reparameterization

**P1 for interpreting mechanism/novelty; not a numerical implementation defect.** The snapshot specification line 13 and method lines 69–80 compare a router receiving corrected log chroma and relative action with a router whose action is fixed at the initial proposal. This isolates action-dependent weighting, but does not isolate a distinctive transported-evidence operation.

Let `u=z-a0`, `v=s-a0`, `d=a-a0`. Proposed color/action inputs are `[u-d,v-d,d]`, while a generic action-conditioned router could receive `[u,v,d]`. For any first-layer weights,

```text
Wz(u-d) + Ws(v-d) + Wd*d
= Wz*u + Ws*v + (Wd-Wz-Ws)*d.
```

The mapping is invertible, and all later activations/weights can remain unchanged. Thus the two router function classes are exactly the same. Ordinary weight decay or initialization could create optimization differences, but the original construction does not establish a structurally different physical transport mechanism. The action-independent posterior comparison cannot detect this confound.

**Accepted amendment from root, assessed independently:** replace each corrected log-chroma pair with two centered components of its corrected chromaticity simplex:

```text
p(z,a) = softmax([zR-aR, 0, zB-aB])
phi(z,a) = [3*pR-1, 3*pB-1]

transport router: [f,k,phi(z,a),phi(s,a),d]
action control:   [f,k,phi(z,a0),phi(s,a0),d]
posterior:        [f,k,phi(z,a0),phi(s,a0),0]
```

This nonlinear action transform cannot generally be absorbed into a single first affine layer. Its two retained components determine the third, so it does not introduce additional information. A sufficiently capable generic action MLP can approximate the same transform. The defensible hypothesis is therefore an explicit physical inductive bias or numerical-conditioning benefit at matched finite capacity, not increased information, universal expressive power, or new physics.

The physical interpretation is sound for positive local channel means/RMS under global diagonal division: normalize the corrected three-channel statistic by its channel sum. Mean and RMS both commute with a positive channelwise multiplier. The identity is restricted to that operation and valid unfloored statistics; clipped/zero cells and numerical flooring require explicit treatment. It does not imply that the learned encoder, proposal head, or entire predictor is exactly equivariant.

Keep the `action` mode as an essential control, with the same active relative-action inputs and module graph as transport. It should receive the same losses, sampling, schedule, initialization protocol, search and teacher enhancements. The posterior remains essential for testing whether action-dependent weights themselves improve over a coherent analytic decision risk. Neither control replaces the other. Retain the original criticism in the record; do not retroactively describe the affine prototype as a novel transport architecture.

## 2. More search stages also mean more candidate evaluations

**P2 for the interpretation of iterative refinement.** Specification line 26 and method line 86 use 25/51/103 queries for 1/2/4 stages. This is a valid deployed latency–accuracy experiment, but better four-stage error would not by itself show a benefit from sequential reconsideration: it also spends more queries and visits a different candidate set.

Before attributing gains to adaptation across stages, compare the same frozen critic against a predeclared nonadaptive candidate design with the same query count and comparable action domain, centered on the original point. Report actual latency as well as query count; duplicated/clamped candidates and batching can change costs. Retain true candidate-oracle error and selected error for each policy, since candidate coverage and ranking error are separate causes. This comparison need not block the initial source screen, provided its first report calls 1/2/4 a compute–accuracy curve rather than evidence of a new reasoning capability.

The wording that the action is repeatedly updated, the encoder runs once, and there is no recurrent hidden-state model is accurate. Retaining the previous center proves only nonincreasing predicted risk, not nonincreasing real reproduction error. The documents explicitly acknowledge this; preserve that distinction in reports and user-facing claims.

## Load-bearing safeguards already present

The reviewed design correctly distinguishes bounded weighted physical costs from one coherent posterior when weights depend on actions. It retains the analytic posterior null, angular versus squared-sine expectation distinction, GT-independent action sampling, original-scene sample accounting, an uncalibrated first screen, held-out selected-action reliability fitting, and source-only configuration selection. The old validation set is explicitly development evidence rather than a fresh benchmark.

Scope and novelty language are appropriately restrained: global positive diagonal geometry does not establish arbitrary ISP robustness, unknown surface/skin colorimetry, or recovery of unobservable reflectance. Candidate scoring, iterative correction, distillation, and derivative supervision are attributed to prior work. Latent local proposals are not asserted to be measured local illuminants. Teacher/local-feature stages remain planned ablations, and a generic compact model with the same training enhancements is required later.

No further load-bearing scientific finding was identified in this bounded review. Source-paper licensing and implementation correctness require their separate existing audits; this report does not certify either or predict accuracy.
