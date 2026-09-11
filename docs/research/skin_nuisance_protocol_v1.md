# Training-only nuisance branch: causal source controls

Fixed before new fitting; adaptive SOURCE follow-up, not independent validation.
36fits: learned/fixed_grid/global/bias x seeds17/29/43 x mixed/from_SLR/from_ipod.
Same original MSKCC source pixels/nativeLab,80epoch budget, site batches,
standardizedMSE, optimizer, identical initialization and zero-branch same-camera
checkpoint selection as the prior training-only experiment. No TEST/CAL use.

Learned reproduces the previous graph_always arm exactly. Fixed_grid replaces
learned affinity/anchor/coupling by the four-neighbor grid, couplinglog(2), and
anchorlog(2)+.05. Global uses a complete graph with total weighted degree3.5log(2)
at every node, equal to the mean grid degree; no image-coordinate dependence.
Both perform three anchored Jacobi updates on local latent features. Operators
are fixed; residuals still depend on image features. Bias replaces the residual
by a constant.05vector before the same learned256x256projection and.5*tanh.
Its functional perturbation has only256degrees of freedom although nominal
active matrix count is65,536; disclose this simpler control rather than implying
equal expressive capacity. No amplitude or hyperparameter search is performed.

All methods retain absolute local features, and deploy exactly the same plain
924,932parameter core. Stored modules are identical; nominal active fitting
counts are disclosed. No learned physical spectrum/illuminant is inferred.
The primary question is whether learned graph structure is necessary for the
one-direction effect. If fixed/global/bias controls account for it, reject a
special graph novelty claim. Compare both directions and the strong previously
reproduced capture-plainMSE and capture-mixtureMSE. No best-camera reporting.

Save best/final/source predictions, independently verify nativeDeltaE00 and all
fixed-coverage dispersion diagnostics. The learned arm must exactly reproduce
the prior graph_always best/final prediction arrays and final weights. Source
roles have already been explored; even a new gain is not independent confirmation.
