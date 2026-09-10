# Research data handling

Face images are sensitive research data even though the model performs no identity recognition. Use explicit research consent covering capture, color measurements, ML development, who can access data, retention, withdrawal and any downstream use. Jurisdiction-specific legal text and retention periods require the appropriate local review; this document does not assert legal compliance.

- Keep contact details/consent linkage separate from opaque ML subject IDs; restrict re-identification keys to the authorized collection coordinator.
- Store original images, instrument exports and manifests on controlled local storage with encryption/access controls and backups appropriate to the approved protocol. No participant data was created or uploaded in this iteration.
- Keep all photographs out of source control. `.gitignore` excludes `data/private/` and generated toy data, but cannot prevent committing an arbitrarily named sensitive path: review staged file lists before every commit.
- Do not upload images to image generators, LLMs, hosted annotation services or unrelated analytics. Current telemetry is local experiment metadata only.
- Strip identifying free-text notes, filenames, GPS and unnecessary EXIF from copies used for modeling; preserve an access-controlled capture record when technical metadata is needed. Do not infer ethnicity or identify subjects from faces.
- Version dataset withdrawals and regenerate frozen dataset/split hashes; never silently mutate a test set while retaining an old evaluation claim. Include derived models/exports in the withdrawal and retention decision procedure.
- Publish aggregate evidence and consented non-identifying illustrations only after authorization. No public release is part of the present task.

Any API profile update requires explicit user confirmation. `UNSUPPORTED` and `RETAKE` must not update a profile. Development artifacts in this repository have no approved operating domain, so the current API exposes no real measurement.
