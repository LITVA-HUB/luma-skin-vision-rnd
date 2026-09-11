# Public instrument-paired skin sources, checked2026-09-11

This inventory concerns actual skin measurements, not image-derived tone
classes or illuminant-vector proxies. Dataset license and scientific fitness
are independent. No proprietary acquisition or external contact is authorized
by this inventory.

|Original source|Reference and inputs|Rights/access status|Decision|
|---|---|---|---|
|[MSKCC Skin Tone Labeling](https://api.isic-archive.com/doi/mskcc-skin-tone-labeling-dataset/)|Original4879images/64people; instrument-paired1838normal-skin images/474sites/46people after3missing-ID exclusions. SkinColorCatch triplicate native Lab; manufacturer D65/10deg.453clinical close-ups+1385dermoscopic; CanonT6i and iPodTouch7, study-calibrated capture.|CLEARED FOR R&D AND COMMERCIAL MODEL TRAINING under original CC-BY attribution grant; version unspecified on landing page. Every metadata record CC-BY. No author weights adopted. Fullbundle5,855,692,278B; acquired only instrument-paired original images:2,128,062,766B, all1838S3 MD5 checks passed.|Direct DeltaE00 source-validation pilot completed; ten-person test unopened. Controlled close-ups, not everyday facial selfies. See skin_mskcc_summary_v1 benchmark and skin_mskcc_protocol_v1.|
|[He2021](https://zenodo.org/records/5532176)|Measured skin XYZ with matched regional RAW/JPG RGB;200 training sites/40 people,100 test sites/20 people; also95 silicone and140 chart patches. Canon6D MarkII, controlled polarized LED setup; CM-2600d SCI/SAV. No full images in deposit.|CLEARED FOR R&D AND COMMERCIAL MODEL TRAINING: original metadata CC BY4.0.395,693bytes acquired, publisher MD5 and local SHA256 checked.|Use the author subject split for a bounded RGB-to-XYZ baseline. No camera-independent/full-image result. Missing exact numeric white/SPD prevents our current physical DeltaE00 claim.|
|[Liu2021 companion spectral charts](https://zenodo.org/records/5730472)|380 chart reflectances and paired RAW RGB under two lights; numeric3500K/6500K SPD. This is chart data, not skin.|CLEARED FOR R&D AND COMMERCIAL MODEL TRAINING: original CC BY4.0.206,045bytes acquired and checked.|Calibration correspondence probe only. Its RAW and reconstructed XYZ do not match the He chart; DO NOT transfer its white reference to skin.|
|[ENCoDE](https://physionet.org/content/encode-skin-color/1.0.0/)|128 people, instruments plus iPhoneSE2020/Pixel4a images at body sites including forehead. Paper reports1,211 iPhone and1,227 Android released images. Missing readings exist.|DO NOT USE until approved credentialed access: PhysioNet Health Data License/DUA1.5.0 and required training. No account/DUA/access obtained. Not commercially cleared here.|Strong candidate for a direct multi-device skin endpoint once lawful access and conditions are resolved. No files downloaded.|
|[DAST original authors](https://github.com/dasec/DAST-SkinTone-database)|Face portraits under exposure changes and colorimeter readings. Full dataset requires author contact; repository exposes small examples.|LICENSE UNCLEAR: no original general dataset grant verified; no example images downloaded.|Study protocol only. Do not assume publicly visible ZIP grants model-training rights.|
|[CHROMA-FIT author lab](https://research.fit.edu/idl/publications/)|About2,300 images/209 people, indoor/outdoor; paper describes forehead/forearm instrument measurements.|LICENSE UNCLEAR: original access/license not verified in this pass.|Instrument-paired candidate; no download/training or external contact.|
|[ISSA](https://pmc.ncbi.nlm.nih.gov/articles/PMC11930939/)|Measured skin spectra; potential physical reflectance prior, not a verified paired everyday-camera facial image release.|LICENSE UNCLEAR in this pass: publication openness is not a substitute for original dataset terms.|Metadata/prior-art study only. No acquisition or use.|
|[UMINHO-HSFD](https://figshare.com/collections/_/7163569)|29 measured hyperspectral faces, reflectance arrays and RGB rendered from those measurements.|LICENSE UNCLEAR in this pass: item-level licenses not verified.|Candidate for measured spectral reconstruction; rendered RGB would be labelled as derived input, not real phone capture. No acquisition.|

The [CIE1931 observer table](https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer)
is a separate standard data resource under original CC BY-SA4.0, verified by its
[metadata](https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv_metadata.json).
Its24,021bytes and both original hashes were checked. Used only in the failed
companion-spectrum correspondence diagnostic, not to infer new skin labels.
Attribution: CIE2019, DOI10.25039/CIE.DS.xvudnb9b. Retain the table's share-alike
terms for adaptations; it is not a model-weight license.

Author/source metadata, acquisition receipts, workbook header/identity audit and
negative correspondence check are in [provenance](provenance/skin_public_2026_09_11/).
The original workbooks remain in local data storage, unmodified and unpublished.
