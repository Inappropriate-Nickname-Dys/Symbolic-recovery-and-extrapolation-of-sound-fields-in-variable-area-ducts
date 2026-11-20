"""Physical-reference and enhanced-search regression checks; no Julia in unit CI."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from acoustic_technique.supplemental import common, data, evaluation, search, pipeline

REPO = Path(__file__).resolve().parents[1]


class NonlinearScienceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations = data.prepare_nonlinear()

    def test_implicit_reference_agrees_with_independent_series(self):
        theta = 2 * np.pi * np.arange(256) / 256
        for sigma in [0, 0.16, 0.32, 0.6, 0.8]:
            with self.subTest(sigma=sigma):
                p = data.implicit(sigma, theta)
                self.assertLess(np.max(abs(p - np.sin(theta + sigma * p))), 1e-11)
                self.assertLess(np.max(abs(p - data.series(sigma, theta, 400))), 1e-8)

    def test_training_validation_and_extrapolation_stay_separate(self):
        d = self.observations
        self.assertEqual(d["Xtrain"].shape, (2560, 2))
        self.assertEqual(d["Xval"].shape, (1536, 2))
        self.assertEqual(d["Xtest"].shape, (24576, 2))
        self.assertEqual(d["Xid"].shape, (3072, 2))
        self.assertTrue(set(d["sigma_train"]).isdisjoint(d["sigma_valid"]))
        self.assertLess(d["Xtrain"][:, 0].max(), 0.32)
        self.assertGreater(d["Xtest"][:, 0].min(), 0.32)

    def test_derivatives_and_phase_sensitive_harmonic_reference(self):
        d = self.observations
        p = d["ytest"]
        ps, pt = data.reference_derivatives(d["Xtest"], p)
        m = evaluation.nonlinear_metrics(d, p, derivatives=(ps, pt))
        self.assertLess(m["burgers_residual_absolute_rms"], 1e-13)
        h = m["harmonics"]
        self.assertLess(np.max(abs(h["cosine"])), 1e-10)
        self.assertLess(max(h["relative_coefficient_error_by_order"]), 1e-6)

    def test_noise_uses_train_rms_and_leaves_test_clean(self):
        j = pipeline.make_job("noise_performance", "test", 2, 3600, {}, 40)
        with patch.object(data, "clean_data", return_value=copy.deepcopy(self.observations)):
            d = data.build_observations(j)
        expected = np.sqrt(np.mean(d["ytrain_clean"] ** 2)) * 0.01
        self.assertAlmostEqual(d["noise_std"], expected, places=14)
        self.assertTrue(np.array_equal(d["ytest"], self.observations["ytest"]))
        self.assertFalse(np.array_equal(d["ytrain"], d["ytrain_clean"]))
        self.assertFalse(np.array_equal(d["yval"], d["yval_clean"]))


class EnhancedProtocolChecks(unittest.TestCase):
    def test_nonlinear_defaults_are_the_latest_unrestricted_configuration(self):
        j = pipeline.make_job("confirmation_final", "nested80_long", 100, 3600, {})
        c = search.configuration(j, Path("/unused-unit-test"))
        self.assertEqual(c["warmup_maxsize_by"], 0)
        self.assertEqual(c["constraints"], {})
        self.assertEqual(c["nested_constraints"], {})
        self.assertEqual(
            (c["maxsize"], c["maxdepth"], c["populations"], c["population_size"]), (80, 30, 64, 128)
        )
        self.assertEqual(c["timeout_in_seconds"], 3600)
        self.assertEqual(c["unary_operators"], ["sin", "cos", "exp", "cosh"])
        self.assertIsNone(c["early_stop_condition"])
        final = pipeline.PLAN["final_configuration"]
        for key, value in final["overrides"].items():
            self.assertEqual(c[key], value)

    def test_linear_noise_and_latest_nonlinear_do_not_share_old_constraints(self):
        j = pipeline.make_job("simple_noise", "conical_g1", 0, 180, {}, 40, "conical", 1)
        c = search.configuration(j, Path("/unused-unit-test"))
        self.assertEqual(c["maxsize"], 40)
        self.assertEqual(c["nested_constraints"]["sin"]["sin"], 0)
        self.assertEqual(j["noise_seed"], 471010)

    def test_validation_selection_does_not_follow_ood_errors(self):
        candidates = [
            dict(complexity=5, validation_mse=1e-5, ood_mse=100),
            dict(complexity=8, validation_mse=1.02e-5, ood_mse=0),
            dict(complexity=1, validation_mse=None, ood_mse=0),
        ]
        self.assertEqual(common.select(candidates), candidates[0])

    def test_resume_does_not_reset_the_ten_hour_deadline(self):
        parent = REPO / "runs/unit_test_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(parent)) as tmp:
            root = Path(tmp)
            common.dump(
                root / "provenance/nonlinear_start.json",
                dict(start_unix=100, deadline_unix=36100, profile="adaptive"),
            )
            with patch.object(pipeline, "ROOT", root), patch.object(
                pipeline, "discover", return_value=None
            ) as discover:
                self.assertFalse(pipeline.nonlinear("adaptive"))
                discover.assert_called_once_with("adaptive", 36100)


if __name__ == "__main__":
    unittest.main()
