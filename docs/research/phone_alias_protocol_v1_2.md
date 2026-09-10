# Additive HDF5 name repair, after observing v1 results

Preserve v1 locks/predictions/results. Unexpected sole internal `MIS` dataset
names occur inside author samsung.h5/oppo.h5 files; dimensions are camera-sized
HWC RGB, not the16-channel multispectral array. Original pinned author reader
H5Image.py lines62-64 reads the first HDF5 dataset without matching its name to
the camera. Source: https://raw.githubusercontent.com/shirawerman/Beyond-RGB/0a3ed93e37f9f1b67fb23781c5e6ab34631020af/example%20code/H5Image.py
Consulted as format documentation; author code is not imported/executed.

This ADDITIVE engineering repair accepts exactly one dataset named the expected
camera or MIS, only in hash-verified original camera-specific file paths and
only with the original HWC float32 three-channel requirement. Multi-dataset or
16-channel files remain rejected. Pixel scale, orientation, chart quality,
masking, every model/head/threshold, scene selection and metrics are unchanged.
Only v1 rows rejected for this key-name reason are numerically re-read. All
unaffected arrays/GT/decisions are copied byte-for-byte from verified v1 cache.

Freeze this executable and addendum before calculating repaired errors.
Predictions read no GT. Report all30 outcomes and both original/repaired counts.
This is not a fresh confirmatory test: original72-row outcomes were already
observed. Do not select weights or soften gray-reference quality on this basis.
