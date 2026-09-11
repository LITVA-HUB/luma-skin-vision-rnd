# TRAIN-only audit amendment, before material fits or VALIDATION access

The frozen metadata split is unchanged (ade4ed5). 8,680 TRAIN records from
1,232 labels have complete declared spectral support and valid 400-700nm
reflectances strictly between zero and one after percent conversion.
No reserved or VALIDATION numeric endpoints were decoded.

Source XYZ formulas use each record's declared 360-740nm or 400-700nm support,
but all Lab formulas use the same white from 400-700nm. Independent scalar
replay of 26,040 XYZ values agrees within 3.56e-14. Native Lab replay agrees
within 1.14e-13. This mixed support convention must remain explicit.

For 3,114 TRAIN records with 360-740nm support, dropping the extra measured
bands alone changes mean DeltaE00 by 0.09546 (p95 0.11468). For 5,566 common
400-700nm records the same computation reproduces supplied Lab to numerical
precision. Therefore the fixed material screen will report spectral error on
all records, but its primary DeltaE00 endpoint ONLY on the original 400-700nm
subset. No unseen values or missing tails are imputed. Every configuration
uses the same fixed subset, determined by metadata before validation access.

Add a TRAIN subject-weighted mean-spectrum control to the 15 fixed PCA
configurations. Include exact-input color replay as a numerical diagnostic,
not a trained model. No width/transform is chosen by validation for a new
independent claim. Statistical unit is an original subject label, with the
identity uncertainties already recorded in metadata.json.

Blank shared-formula XML elements required distinguishing shared references
from explicit formula text in the audit reader. All numeric values are replayed
independently of that text. An explicit hue formula references column XO,
outside the data table; hue is not used in XYZ/Lab targets or model metrics.
The original workbook is unmodified. Initial parser failures are part of the
record: optional blank demographic fields and shared formula storage were
handled before any material model fit, without changing measured endpoints.

New external prior art: the 2026 work [The optical origin of the human skin
color 'banana' in CIELAB space](https://pmc.ncbi.nlm.nih.gov/articles/PMC13307969/)
connects chromophore/light-transport models to the ISSA color locus. This
further rules out claiming that a physical skin-color manifold itself is new.
Our proposed use would have to demonstrate an actual image error/rejection
advantage over matched ordinary regularization; that remains unproven.
