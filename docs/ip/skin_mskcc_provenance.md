# MSKCC source pilot rights and technical provenance

Original release DOI10.34970/962049, Memorial Sloan Kettering Cancer Center.
Original landing page, all4879metadata records and checked image API records
specify CC-BY. The landing page does not identify the license version; do not
invent a4.0 designation. Retain attribution and provenance for adaptations.
Dataset, article, code and weights are separate rights layers. In particular,
the2026prior-art article license or author's pretrained model provenance is
not inherited by this original dataset. No author weights/code were imported.

Local code implements standard ridge, polynomial, MLP and neighbor controls;
direct instrument Lab regression is established prior art, not a patent claim.
Candidate paired-view reliability research remains unverified. No external
publication, upload of participant images, author communication or legal
classification was performed. Raw images and participant mappings remain local,
outside Git. Committed provenance consists of source page and aggregate/hash
receipts; benchmark arrays with pseudonymous record IDs remain local.

The API-declared image size was found to differ from the actual original S3
object size. The initial strict download stopped; the corrected acquisition
verifies each original JPEG against HTTP Content-Length and its single-part
S3 ETag MD5, retains API-vs-object discrepancy and computes SHA256. It does not
claim the API declared size was correct or that images were silently repaired.
