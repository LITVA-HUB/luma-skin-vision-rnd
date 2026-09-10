# Fresh-target loader review

Date: 2026-09-10. **STATIC AND METADATA-ONLY CPU REVIEW. No target image decoding, GT-value inspection, model prediction, training or error evaluation.** Core source files were not edited.

## Final verdict: PASS; reported P2 resolved

The initial review found that `prepare_cc_v2_fresh.py` recorded current selection-manifest hashes without enforcing the preselected set. Root now imports `FROZEN` from the downloader and compares the complete `{filename: SHA-256}` mapping at lines 34–36. Missing manifests, additional matching-pattern manifests and changed bytes all cause rejection. This executes before output directory creation (line 37), sample processing, image/GT payload reads (line 50), and decoding (line 58). The originally reported substitution gap is closed.

Independent metadata-only checks used temporary copies of the three selection manifests and a nonexistent payload directory:

- Missing manifest: rejected before output creation.
- Extra manifest matching the selection glob: rejected before output creation.
- Modified manifest bytes: rejected before output creation.
- Exact frozen set: preflight accepted; deliberately stopped by a test sentinel at output-directory creation, before any payload access.

The decoder was mocked to fail if called; **decoder calls = 0**. The current source metadata hashes match all three `FROZEN` constants. Focused Ruff check: **All checks passed**. These checks were executed without changing production code or committed tests.

## Retained preprocessing assessment

The script selects the processed 1080p TIFF format and accepts only uint16 three-channel decoded arrays. It reverses OpenCV BGR order to RGB, applies no repeated camera black-level subtraction, camera CCM, gamma conversion or chart rectangle, and requires three finite positive `.wp` components before scale normalization. This is consistent with the publisher's distinction between processed TIFFs with normalized RGB references and the separate Bayer packages with two-component `[R/G,B/G]` references. The paper explicitly says chart-bearing reference images are absent from the distributed database. [Official author paper, Sections III–IV](https://pure.au.dk/ws/files/301638552/INTEL_TAU_A_Color_Constancy_Dataset.pdf), [publisher dataset description](https://researchportal.tuni.fi/en/datasets/intel-tau/).

The 0.98 rejection threshold, square area resize, per-image exposure normalization and clipping are the project's fixed preprocessing choices, not a claimed reproduction of the paper's evaluation. Their full raw-image path is not the scope of the model's exact cached-tensor diagonal-equivariance proof. Actual file encoding/range, valid-image counts and derived outputs remain unverified until the authorized post-lock preparation. Mirror-to-original byte identity and scene clustering limitations remain as documented in `cc_v2_camera_acquisition.md`.

Reviewed SHA-256:

- `scripts/prepare_cc_v2_fresh.py`: `c777cbfea5ecfe630b0931405fe4876e1e3f3bbc818230a0640aa24686d0c16a`.
- `scripts/download_cc_v2_fresh.py`: `998fa1b44abb7a71bd74a4d311f7bb755192460085e7fb7437340307153de740`.
