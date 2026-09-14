# Luma ChromaSeed-Seg1: spatial skin extraction

The user authorizes autonomous data collection and quality experiments, including larger models on the RTX 4060. Five supplied images have no measured color reference. Their diagnostic overlays show the UCI color-only classifier selecting walls and other objects. This makes semantic facial-skin extraction the next concrete experiment; head-range color-regression work remains pending after this data expansion.

Use the public author-linked LaPa archive as observed (18,168 training images, 2,000 validation, 2,000 test; eight training images fewer than the commonly reported 18,176). Preserve original triplets and all author labels. Keep official test; exclude lower-priority exact-image duplicates and conservative filename-prefix overlaps. Subject independence is not established. Decode train and validation only before selection. Original test bytes can be extracted and hashed without model evaluation.

Train a U-Net from scratch, width 24 (target under five million parameters), GroupNorm and SiLU, input 192x192. Binary target is LaPa labels 1 or 6 (facial skin and nose). Other classes, including hair and eyes, are excluded. Preserve the original eleven-class masks for future work. This is a segmentation component; it does not predict instrument Lab.

Prospective single-run recipe: seed 20260914, AdamW lr 0.0003, weight decay 0.0001, batch 32 unless actual GPU memory preflight requires reduction, twelve epochs, cosine learning rate ending at 10% of the initial rate. Fixed binary cross-entropy plus soft Dice loss. Training-only horizontal flips, modest exposure/gamma/channel-gain augmentation. Validate every epoch; choose highest mean per-image skin IoU, earliest epoch breaks ties. Prediction threshold is logit zero; no test threshold tuning. Freeze checkpoint and selection before decoding test for evaluation. Report global IoU, Dice, precision, recall, per-image IoU, speed, size and all failures.

Implementation and validation:

- [x] CPU tests for shape/gradient/loss/metrics and exact own-state reload; invalid shapes/labels rejected.
- [x] Prepare audited train/validation arrays, preserve source hashes and duplicate exclusions. Actual retained counts15914/1692, official2000test untouched for evaluation.
- [ ] Wait for AS reconstruction terminal zero, seal AS and verify it read-only before any new GPU job.
- [ ] GPU training-batch preflight with finite loss and gradients, record peak memory; do not run concurrent GPU jobs.
- [ ] Freeze protocol including actual batch and all sources/data hashes; run twelve epochs with progress/ETA and validation-only selection.
- [ ] Freeze selection, evaluate official held images with the same preprocessing, compare the existing frozen UCI baseline on a prospectively hashed 128-image subset if runtime permits.
- [ ] Run supplied photos and DAST as exploratory diagnostics; never invent measured skin-color truth or independent-person accuracy.
- [ ] Record model, dataset and result receipts, update the training dashboard if needed for the new active run.

No originals or participant images in Git, no uploads, no external messages. Dataset license status is recorded separately from scientific usefulness, following the latest user instruction. No AS or previously sealed source is modified.
