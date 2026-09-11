# Pixel source ablations v2 — frozen after v1 source screen

This is adaptive research on the SAME six validation people, not a new test.
The original80-epoch three-seed9run screen is retained. Pixel cache, subject
roles, label conventions, optimizer, budget and selection are unchanged.

Observed: mean-pooling patch-vote seeds give3.4888/3.5074/3.4964 meanDeltaE00,
CNN3.8139/4.0029/4.1147. The three-step Huber5 variants are essentially tied
with plain confidence pooling. At the selected checkpoints only0.01–0.70% of
individual votes exceed the5Lab-unit threshold. This explains weak intervention,
but does not prove tightening it helps. There is no novel-method victory.

Freeze four new arms, seeds17/29/43,80epochs each, before running them:

1. Capacity-matched global-color MLP:36locally extracted color features,
   Linear36->512->768->512->256->3 with SiLU between layers. About0.94M
   parameters, same data/loss/optimizer/budget. Tests whether v1 gain is merely
   larger capacity or training recipe rather than a useful patch distribution.
2. Shared-context votes,3Huber steps with delta0.5 (tenfold smaller). Exactly
   the v1 network parameters. Tests whether the old intervention was inactive.
3. Local-only color votes with global-context confidence, no iteration.
   Reuse the identical local/context/vote modules and parameter count. Compute
   Lab votes with the512context inputs set to zero; compute confidence using
   the full context inputs through the same vote module. Context can choose
   plausible patches but cannot directly fill every local color vote with the
   same global answer. No pixel position or camera ID enters the network.
4. Local-only color votes plus3Huber steps with delta0.5. Compare directly
   against arm3 with identical initial weights and data order. Assumption:
   local estimates have enough information to be useful; failure mode: losing
   necessary lighting context creates biased patch votes that robust pooling
   cannot repair. A win requires actual DeltaE00 reduction, not more disagreement.

These are ablations using established mechanisms, not asserted inventions.
Preserve all outcomes. Camera and clinical strata remain known-camera source
development. Error heads/calibration and the ten-person final test stay reserved.
