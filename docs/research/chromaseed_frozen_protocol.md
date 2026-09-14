# ChromaSeed frozen-color-basis control

Declared after synthetic-only pretraining completed, while the primary inner fit is running and before any new real outer evaluation. This secondary extension is exploratory. No source renderer or primary training configuration changes.

Question: does the palette-trained hidden layer transfer useful color features when it is frozen, and can a skin readout be learned analytically without backpropagation? This separates representation transfer from initialization/fine-tuning.

Use the same original TRAIN hash, outer roles, three inner person folds, person/site weights and seeds17/29/43. Four bases: saved random initialization, clean_palette, rendered_palette, shuffled_palette. Always use the final2048-step synthetic hidden layer; no skin warmup or skin_long hidden layer. No teacher at inference.

For each fit role, obtain fixed normalized inputs z36 and SiLU(hidden(z))64; concatenate100 features. Compute their mean/std on that fit role only (floor1e-6), solve weighted ridge to fixed-normalized Lab, bias unpenalized. Alpha grid{0.1,1,10}, three inner folds, three seeds. Choose alpha per basis/protocol by mean person DeltaE00 over concatenated inner OOF predictions and seeds. Absorb fit-only centering/scaling into output bias, hidden-output and linear-skip weights so the full deployed model remains exactly2749 scalars/10996 bytes. Hidden weights and biases must be byte-identical to their synthetic/random source.

Freeze all choices/final artifacts before evaluating any outer role. Compare all four frozen bases, and report primary full-fine-tuning results separately. The analytic head has a different fitting algorithm/regularizer; a win cannot be attributed to pretraining unless it also beats the frozen random and shuffled controls. Record synthetic pretraining cost separately; do not count it as zero. Single-fit timing is reported only with its measurement conditions, not as full palette-to-model cost.

Prior art: linear probing, ELM/random features and analytic ridge readouts are known mechanisms. This control establishes no algorithmic novelty or ordinary-phone facial validation. All original TRAIN outer groups are already reused and overlapping; no independent confirmation.
