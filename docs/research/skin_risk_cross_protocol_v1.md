# Frozen source color/risk crossing

Use only the existing36distribution-fit source predictions. No fitting, threshold
selection or held-out endpoint access. Nine protocol/seed cases: mixed,SLR,iPod
times17/29/43. The ordinary model is mse_mode; Gaussian is the same-seed single
density. A second ordinary model uses cyclic seeds17->29->43->17, so all three
unordered ordinary pairs occur. All models were selected on source validation.
This is explicitly post-hoc exploratory source evidence, not a calibrated C+ test.

Ten frozen systems per case:
- ordinary / own four-hypothesis dispersion;
- ordinary / Gaussian expected DeltaE00 at that ordinary prediction;
- ordinary / Gaussian covariance centered at that ordinary prediction;
- ordinary / ordinary-versus-Gaussian mean disagreement;
- two ordinary model mean / pair disagreement;
- two ordinary model mean / mean within-model dispersion;
- ordinary+Gaussian mean / Gaussian expected DeltaE00 at that mean;
- ordinary+Gaussian mean / Gaussian covariance centered at that mean;
- ordinary+Gaussian mean / ordinary-versus-Gaussian disagreement;
- Gaussian mean / its own expected DeltaE00.

The same Gaussian uncertainty can be scored at different proposed answers. This
distinguishes covariance from bias/disagreement. Integration uses fixed scrambled
Sobol seed71131,2048points and their negatives =4096Gaussian nodes, exactly the
prior sensitivity rule. This is approximate expected risk, not an error bound.
All mean combinations are fixed50/50 with no validation tuning.

Record full/100,95,90,80,70,60%coverage error and full curves, patient descriptive
intervals, model counts and active parameters. Two-model ordinary control has
approximately the same inference capacity as mixed-objective crossing. A future
C+ head/calibration needs person-held-out residuals and is not claimed here.
Report all cases even when crossing harms transfer. Independent test is unchanged.
