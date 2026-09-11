# Post-run algebraic explanation, separate from skin DeltaE00 accuracy

For analysis only, decompose site/person-balanced squared Euclidean native-Lab
residual energy. This is DeltaE76 squared geometry, not CIEDE2000, a new target,
a replacement accuracy metric or a measured irreducible floor.

Let m_i be mean target-minus-prediction residual for person i, averaged first
over photos within sites, then over sites. Let mu=mean_i(m_i), v=mean_i||m_i-mu||²,
K=number of people. The excluded-person correction is mu-(m_i-mu)/(K-1).
For fixed strength a, the EXACT change in hierarchically weighted squared
residual energy is:

(-2a+a²)||mu||² + (2a/(K-1)+a²/(K-1)²)v.

Within-person centered residuals cancel in the change. Therefore cross-person
offset correction can hurt when between-person residual variation exceeds the
shared offset benefit. Compute this identity independently on the90 original
source prediction arrays for both frozen strengths, and compare with direct
site/person-balanced squared-error calculation. No model or calibration changes.
These descriptive terms use actual source reference labels, not inference-time
features. They do not establish a pure camera effect or an analytic DeltaE00 law.
