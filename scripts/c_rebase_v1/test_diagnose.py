"""Numerical tests for algebraic audit identities, never model-performance tests."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location('audit', Path(__file__).with_name('diagnose.py'))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class DecompositionTest(unittest.TestCase):
    def setUp(self):
        self.pred = np.array([[3.,0.,0.],[5.,0.,0.],[1.,0.,0.],[4.,1.,0.],[4.,3.,0.],[9.,0.,0.]])
        self.target = np.array([[2.,0.,0.],[2.,0.,0.],[2.,0.,0.],[3.,0.,0.],[3.,0.,0.],[8.,0.,0.]])
        self.sites = np.array(['s1','s1','s1','s2','s2','s3'])
        self.people = np.array(['p1','p1','p1','p1','p1','p2'])

    def test_exact_identity_and_privileged_oracle_exclusions(self):
        result, grouped = audit.decomposition(self.pred, self.target, self.sites, self.people, seed=3, draws=100)
        np.testing.assert_allclose(list(result['mse_Lab'].values()), [14/6,10/6,0])
        within = np.array(list(result['within_site_prediction_variation_mse_Lab'].values()))
        bias = np.array(list(result['site_mean_prediction_bias_mse_Lab'].values()))
        np.testing.assert_allclose(within+bias, list(result['mse_Lab'].values()))
        np.testing.assert_allclose(grouped[:3], [[3.,0.,0.]]*3)
        self.assertEqual(result['oracle_evaluations'][1]['coverage'], 5/6)
        self.assertEqual(result['oracle_evaluations'][2]['coverage'], 5/6)
        np.testing.assert_allclose(result['mean_cross_term_Lab'], 0, atol=1e-12)

    def test_refuses_different_targets_within_site(self):
        changed = self.target.copy(); changed[0,0] += 1
        with self.assertRaises(ValueError):
            audit.decomposition(self.pred, changed, self.sites, self.people, seed=3, draws=100)


if __name__ == '__main__':
    unittest.main()
