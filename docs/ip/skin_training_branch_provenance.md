# Spatial and training-only branch provenance

New implementation written locally with existing PyTorch/NumPy project dependencies.
No externally licensed model weights, foundation model, cloud training, imported
paper code, newly restricted dataset or publication step. Source target/data is
the previously verified original MSKCC image/instrument pairing and unchanged
source cache; UMINHO spectra were not used as training labels in these experiments.

Prior art: graph diffusion, finite quadratic/Jacobi solvers, recurrent residual
networks, stochastic depth and model averaging. References are in prior_art.md.
The observed fitting-only graph effect is a candidate mechanism needing further
isolation, not proof of invention or a patentability opinion. The model still
loses in some source protocols and selective acceptance remains unreliable.

Exact data/code bindings: skin_spatial_v1/source_lock.json,
skin_train_branch_v1/source_lock.json and skin_branch_combination_v1/protocol_lock.json
under docs/benchmarks. Weight/prediction hashes and independent audits are recorded
there. Participant arrays and weights remain local/ignored; no direct identifiers
or images are committed. Data permissions, code provenance and weight use are
separate matters; existing original-source attribution remains required.

The prior independent400-image test is unchanged and exposed. No new independent
skin, ordinary facial-phone or universal-camera accuracy claim is supported.
