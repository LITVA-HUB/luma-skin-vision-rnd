# Post-fit information interventions, not new model selection

The frozen 27-fit screen is complete. Primary results are unchanged. No fitting,
checkpoint selection or reserved-endpoint access is permitted in this diagnostic.
Compute selected-TRAIN color errors for every existing final model to characterize
fit/generalization gaps. These are training errors, never independent accuracy.

For all 18 residual-adapter models, remove the residual while retaining the
trained core. For all nine pixel models, additionally replace the encoder RGB
by within-patch permutation or patch-channel mean. Original 18 patch statistics
remain untouched. These are deliberate feature interventions, not valid new
capture protocols or retrained model baselines. Per-patch permutation uses fixed
seed51871 and moves RGB triples together, preserving the color multiset exactly;
mean retains only each patch RGB mean. No cross-patch mixing. Measure actual
held-out skin DeltaE00 and RMS change of predicted Lab from the original model.
Record residual RMS as a diagnostic, not calibrated uncertainty.

If removing spatial information barely affects outputs, the learned branch has
not demonstrated the intended use of local structure. This says nothing universal
about all pixel networks, other budgets, larger data or identifiable skin color.
Recompute all diagnostic color metrics with independent scalar CIEDE2000.
