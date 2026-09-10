# V5 equal-query refinement result: two completed seeds

**Reused119-image source development validation only.** Five trained critic arms, seeds17/29; best checkpoints already selected by V5 training. No new camera data.

All25-query first stages retained the original model point on all119 images, for all10 checkpoints (1190 model/image cases). Consequently the51-query adaptive and fixed searches produced bitwise identical selected actions in this CPU comparison. The observed25-to51-query improvement cannot be credited to feedback/recentering: the first stage did not move. This is specific to these learned critics and grid radii, not a general impossibility of iterative inference.

At103 queries recentering becomes active, but its benefit is small and inconsistent:

| Seed | Critic arm | Adaptive51 mean ° | Fixed51 mean ° | Adaptive103 mean ° | Fixed103 mean ° |
|---|---|---:|---:|---:|---:|
|17|posterior_random|2.7126|2.7126|2.7477|2.7565|
|17|action_random|2.4461|2.4461|2.4180|2.4253|
|17|transport_random|2.3649|2.3649|2.3765|2.3727|
|17|transport_policy|2.4607|2.4607|2.4836|2.4730|
|17|transport_gradient|2.4676|2.4676|2.4922|2.4947|
|29|posterior_random|2.4950|2.4950|2.5052|2.5018|
|29|action_random|2.4039|2.4039|2.3767|2.3726|
|29|transport_random|2.5243|2.5243|2.5912|2.5903|
|29|transport_policy|2.4804|2.4804|2.5263|2.5314|
|29|transport_gradient|2.5212|2.5212|2.5314|2.5297|

Complete recovery/reproduction/tails and raw risk curves at100/95/90/80/70/60 are in summary.json; exact paired actions and risks in NPZ files. One-stage actions match by executable assertion; evaluated query counts25/51/103 match exactly. Repeated centers count for both methods. No latency measurements are made.

Decision: do not claim a recurrence innovation or invest in more passes on this evidence. Complete seed43 and preserve this fixed-search control for the next stronger critic/context experiment. No change to V5 weights or training code was made. Equal-query comparison controls this multiscale search design, not every possible nonadaptive proposal scheme.
