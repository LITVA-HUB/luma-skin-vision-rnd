"""Synthetic alignment/metric checks only; these are not real C audit results."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c_audit", ROOT / "scripts/error_floor/c_oof_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    rows = []
    for person in range(6):
        for site in range(5):
            for image in range(2):
                target = [40. + person * 7 + site, 7. + site, 12. + site]
                reps = [np.array(target) + np.array([j * .2, j * .1, -j * .1]) for j in (-1, 0, 1)]
                rows.append({"image": f"fake{len(rows):04d}", "patient": f"person{person}",
                             "site": f"site{person}_{site}", "device": "fixture_camera",
                             "mode": "fixture_mode", "image_type": "fixture_type", "anatomic_site": "fixture_location",
                             "target": target, "repetitions": np.asarray(reps).tolist()})
    n = len(rows)
    target = np.array([r["target"] for r in rows])
    pred = target.copy()
    pred[:, 0] += np.linspace(.05, 2.8, n)
    capture = {k: np.array([r[k] for r in rows]) for k in ("image", "patient", "site", "device", "mode", "image_type", "anatomic_site")}
    capture.update(target=target, repetitions=np.array([r["repetitions"] for r in rows]),
                   original_mean_lab=target - 3, original_median_lab=target - 2,
                   original_median_rgb=np.tile([.3, .2, .1], (n, 1)),
                   color36=np.arange(n * 36).reshape(n, 36) / (n * 36),
                   width=np.full(n, 128), height=np.full(n, 128),
                   diagnostic=np.stack([np.linspace(.2, .9, n), np.linspace(0, .1, n), np.linspace(0, .02, n)], axis=1),
                   diagnostic_names=np.array(["brightness_mean_encoded_RGB", "original_high_clip_pixel_fraction", "original_low_clip_pixel_fraction"]))
    data = {"prediction": pred, "reference": target, "row": np.arange(n),
            "anonymous_group": np.array([r["patient"] for r in rows])}
    return rows, data, capture


def align(data, rows):
    return audit.align_oof(data, rows, "prediction", "reference", "row", "anonymous_group", expected_n=len(rows))


class AuditChecks(unittest.TestCase):
    def test_permutation_alignment_and_top50_invariance(self):
        rows, data, capture = fixture()
        perm = np.random.default_rng(717).permutation(len(rows))
        shuffled = {k: v[perm] for k, v in data.items()}
        pred, target, groups = align(shuffled, rows)
        np.testing.assert_array_equal(pred, data["prediction"])
        np.testing.assert_array_equal(target, data["reference"])
        renamed = groups.copy()
        for i, value in enumerate(sorted(set(groups))):
            renamed[groups == value] = f"alias{i}"
        shuffled["anonymous_group"] = renamed[perm]
        align(shuffled, rows)
        capture_permutation = {k: v[perm] if k != "diagnostic_names" else v for k, v in capture.items()}
        restored = audit.align_capture(capture_permutation, rows)
        direct, _ = audit.analyze(rows, data["prediction"], data["reference"], capture, bootstrap_draws=5)
        recovered, _ = audit.analyze(rows, pred, target, restored, bootstrap_draws=5)
        self.assertEqual(direct["overall"], recovered["overall"])
        self.assertEqual(direct["top50"], recovered["top50"])
        self.assertEqual(len(recovered["top50"]), 50)
        self.assertEqual(len({r["row_id"] for r in recovered["top50"]}), 50)
        self.assertEqual(recovered["top50"][0]["row_id"], "R0060")
        self.assertTrue(all("fake" not in str(r) and "person0" not in str(r) for r in recovered["top50"]))

    def test_delta_e_for_pure_lightness_has_independent_formula(self):
        rows, data, _ = fixture()
        target, pred = data["reference"], data["prediction"]
        mid_l = (pred[:, 0] + target[:, 0]) / 2
        sl = 1 + .015 * (mid_l - 50) ** 2 / np.sqrt(20 + (mid_l - 50) ** 2)
        expected = (pred[:, 0] - target[:, 0]) / sl
        result = audit.metrics(pred, target, np.array([r["patient"] for r in rows]))
        self.assertAlmostEqual(result["delta_e00"]["mean"], expected.mean(), places=12)
        self.assertAlmostEqual(result["delta_e00"]["p95"], np.quantile(expected, .95), places=12)
        self.assertEqual(result["mae_lab"]["a"], 0.)

    def test_anonymized_capture_anchored_to_original_rows(self):
        rows, _, capture = fixture()
        n = len(rows)
        anonymized = {k: v.copy() for k, v in capture.items()}
        pa = {p: f"P{i+1:02d}" for i, p in enumerate(sorted({r["patient"] for r in rows}))}
        sa = {s: f"S{i+1:03d}" for i, s in enumerate(sorted({r["site"] for r in rows}))}
        anonymized["row_index"] = np.arange(n)
        anonymized["image"] = np.array([f"I{i+1:04d}" for i in range(n)])
        anonymized["patient"] = np.array([pa[r["patient"]] for r in rows])
        anonymized["site"] = np.array([sa[r["site"]] for r in rows])
        perm = np.random.default_rng(42).permutation(n)
        shuffled = {k: v[perm] if k != "diagnostic_names" else v for k, v in anonymized.items()}
        restored = audit.align_capture(shuffled, rows, anonymized=True)
        np.testing.assert_array_equal(restored["color36"], capture["color36"])
        shuffled["patient"][0] = "BAD"
        with self.assertRaises(ValueError): audit.align_capture(shuffled, rows, anonymized=True)

    def test_alignment_fail_closed(self):
        rows, data, capture = fixture()
        for kind in ("duplicate_index", "wrong_target", "mixed_people", "split_patient", "one_based_index", "missing_key", "nan_prediction"):
            changed = {k: v.copy() for k, v in data.items()}
            if kind == "duplicate_index": changed["row"][0] = 1
            if kind == "wrong_target": changed["reference"][0, 0] += 1
            if kind == "mixed_people": changed["anonymous_group"][10] = changed["anonymous_group"][0]
            if kind == "split_patient": changed["anonymous_group"][0] = "newID"
            if kind == "one_based_index": changed["row"] += 1
            if kind == "missing_key": del changed["reference"]
            if kind == "nan_prediction": changed["prediction"][0, 0] = np.nan
            with self.subTest(kind=kind), self.assertRaises(ValueError): align(changed, rows)
        changed = {k: v.copy() for k, v in capture.items()}
        changed["patient"][0] = "wrongID"
        with self.assertRaises(ValueError): audit.align_capture(changed, rows)


if __name__ == "__main__":
    unittest.main()
