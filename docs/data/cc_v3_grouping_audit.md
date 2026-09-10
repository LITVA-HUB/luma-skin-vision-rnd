# CC V3 identity and grouping audit

Date: 2026-09-10
Status: **OBSERVED METADATA / PROPOSED SPLIT METHOD — NO SPLIT FROZEN**

This audit uses the V1 Sony30 manifest, the V2 and V3 frozen manifests, and the completed byte-verification reports. It compares paths, field identifiers, file sizes, CRCs and SHA-256 values. It does not parse a `.wp` or metadata JSON value, decode image pixels, calculate errors, or inspect model predictions. Image dimensions were not needed and image headers were not opened.

## Verified populations

The V3 verification report is `VERIFIED_BYTES_ONLY`: 384 image/GT pairs and 768 files, with no failures. Its path, byte-size, CRC and SHA-256 records exactly match the download report; all 768 expected files are present under the V3 data root. Every V3 path and size/CRC pair also matches its frozen remainder manifest. The equivalent V2 checks hold for its 384 pairs. All 60 V1 Sony30 files currently match their manifest size and SHA-256.

| Population | Image rows | Unique image paths | Unique image SHA-256 | Unique GT paths | Unique GT-file SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 Sony30 | 30 | 30 | 30 | 30 | 29 |
| V2 historical three-camera set | 384 | 384 | 384 | 384 | 342 |
| V3 new three-camera set | 384 | 384 | 384 | 384 | 347 |

V1's one repeated metadata-file hash is shared by Sony fields 008 and 009. V1 stores resized PNG images and metadata JSON; V2/V3 store TIFF images and `.wp` files. Exact hashes across those representations are therefore not semantic reference comparisons.

## Historical overlap

V3 and V2 have zero shared image paths, image SHA-256 values, or GT paths. All 384 V3 image identities are distinct from the 384 V2 image identities. The 30 V1 Sony field identifiers 001–030 are absent from both the V2 and V3 Sony selections; V1 also has no exact image-file or GT-file hash collision with either set.

V2 and V3 do share **46 exact GT-file hashes**, always within the same camera:

| Camera | Shared V2/V3 GT hashes | V2 rows carrying them | V3 rows carrying them |
| --- | ---: | ---: | ---: |
| Canon 5DSR | 17 | 28 | 25 |
| Nikon D810 | 10 | 15 | 16 |
| Sony IMX135 BLCCSC | 19 | 20 | 26 |
| **Total** | **46** | **63** | **67** |

These are repeated reference-file bytes, not duplicate images. They show that path-disjoint selection did not create a reference-hash-disjoint population. Any later report should describe V3 as 384 new image files with 67 rows carrying a reference hash already present in the historical V2 evaluation set. The 128 rows per camera are a preselected acquisition population, not yet a prespecified primary evaluation protocol. A future LOCO lock may define the 103/112/102 reference-history-disjoint rows (317 total) as the proposed primary population and retain all 128 rows per outer camera as a sensitivity population, or choose another fully documented policy before GT values or errors are inspected.

## Reference-hash grouping

Within V3, exact GT-file SHA-256 produces the following proxy groups:

| Camera | Rows | Groups | Singleton groups | Repeated groups | Rows in repeated groups | Largest group | Group-size histogram |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Canon 5DSR | 128 | 114 | 104 | 10 | 24 | 4 | 104×1, 7×2, 2×3, 1×4 |
| Nikon D810 | 128 | 118 | 110 | 8 | 18 | 3 | 110×1, 6×2, 2×3 |
| Sony IMX135 BLCCSC | 128 | 115 | 105 | 10 | 23 | 3 | 105×1, 7×2, 3×3 |
| **Total** | **384** | **347** | **319** | **28** | **65** | **4** | — |

For comparison, V2 has 107/121/114 groups for Canon/Nikon/Sony, respectively, or 342 total. Combining V2 and V3 gives:

| Camera | Combined rows | Combined groups | Singleton groups | Repeated groups | Rows in repeated groups | Largest group |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Canon 5DSR | 256 | 204 | 173 | 31 | 83 | 7 |
| Nikon D810 | 256 | 229 | 213 | 16 | 43 | 8 |
| Sony IMX135 BLCCSC | 256 | 210 | 176 | 34 | 80 | 5 |

There are **zero exact GT-file-hash groups spanning two cameras** in V2, V3, or their union. This absence must not be interpreted as cross-camera scene independence: camera-specific sensor responses and serialization can produce different reference bytes for the same physical scene.

## Physical-scene limits

An equal GT-file hash proves byte equality only. It may indicate repeated white-point metadata across nearby captures, but it does not prove that the images show the same physical scene, subject, illumination event or acquisition session. Conversely, unequal reference hashes do not prove different scenes. Filename field numbers are camera-local identifiers, and no verified cross-camera event mapping is available.

Zero image SHA-256 overlap proves that the stored files are not byte-identical. It does not rule out resized, re-encoded, cropped, burst-adjacent or visually near-duplicate captures. This audit deliberately performs no perceptual comparison. Reference-hash grouping is therefore a conservative leakage proxy and a bootstrap/splitting unit, not a validated physical-scene identifier.

## Proposed deterministic later LOCO split

No row assignment is made or frozen here. Before any V3 errors are calculated, a separate immutable split artifact should implement the following fold-local procedure:

1. Create three outer folds. In each fold, all 128 rows from one camera are the outer test population. No row, image statistic, GT value, prediction, aggregate or calibrated score from that camera may enter estimator fitting, checkpoint/architecture selection, posterior or risk fitting, threshold calibration, normalization, or early stopping for that fold.
2. Keep all V1 and V2 images outside every fitting role. Tag V3 groups whose GT-file hash occurs in V2. A pending conservative option is to exclude all 67 history-linked V3 rows from every new role, leaving 103 Canon, 112 Nikon and 102 Sony rows. If selected before GT/error inspection, those 317 rows can form the proposed primary LOCO population and the complete 128-per-camera acquisition population can remain a predeclared sensitivity analysis. Another option is to use history-linked inner-camera groups only for estimator training while excluding them from validation, risk fitting and calibration. The later LOCO lock must choose one policy; this audit does not freeze it.
3. Define the minimum grouping unit globally by exact GT-file SHA-256. All rows in a group receive one role. Use a global key even though the present audit found no cross-camera group, so a later corrected identity table cannot split a cross-camera group across roles.
4. Assign the remaining inner-camera groups to estimator train, checkpoint validation, risk fitting and calibration using a documented fixed salt and a deterministic hash ordering. Use a deterministic greedy group allocator to approach predeclared row-count proportions separately for each inner camera while never breaking a group. Suggested planning proportions are 70/10/10/10 percent by row count; exact integer targets, salt, role precedence and tie-breaking must be frozen before execution.
5. Generate each fold independently. Hyperparameters and checkpoints for an outer camera may use only that fold's two inner cameras plus the already frozen source-only development decision. Do not pool outer-fold errors to revise another fold. Each evaluated row must come from a checkpoint whose config proves that row's camera was absent from every fitted or selected component.
6. Record the ordered row IDs, image/GT paths, image/GT hashes, proxy group key, historical-reference tag, role, outer camera, split-script hash and parent provenance hashes. Validate zero role overlap by path, image hash and group key before decoding for training. Preserve failed folds and all split artifacts once frozen.

This strategy prevents exact reference-file groups from crossing inner roles and enforces strict outer-camera exclusion. It cannot by itself establish physical-scene-disjoint LOCO evidence; that claim would require authoritative acquisition-session or scene correspondence metadata, or a visual/metadata near-duplicate audit.
