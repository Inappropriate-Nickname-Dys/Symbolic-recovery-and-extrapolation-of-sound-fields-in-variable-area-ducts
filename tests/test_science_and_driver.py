"""Check physical labels, independent certification and the executable study plan."""

import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
import sympy as sp
from acoustic_technique.acoustic_core import (
    T,
    analytic,
    generate,
    parameters,
    refine,
    solve,
    evaluate_expr,
)
from acoustic_technique.structure_certificate import certify

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("study_driver", REPO / "reproduce.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ScientificChecks(unittest.TestCase):
    def test_numerical_labels_match_all_four_analytic_references(self):
        x = np.linspace(0, 1, 101)
        for kind in ["uniform", "conical", "exponential", "catenoidal"]:
            with self.subTest(kind=kind):
                p = parameters(kind, 1)
                solution = solve(p)
                self.assertTrue(solution.success)
                self.assertLess(np.max(abs(solution.sol(x)[0] - analytic(x, p))), 1e-8)

    def test_observations_and_validation_are_local_and_disjoint(self):
        for scheme in ["uniform", "random"]:
            with self.subTest(scheme=scheme):
                d = generate(parameters("conical", 1), n=40, sampling=scheme, seed=2)
                self.assertEqual(d["Xtrain"].shape, (40, 1))
                self.assertEqual(d["Xval"].shape, (240, 1))
                self.assertLessEqual(max(d["xtrain"]), 0.4)
                self.assertLessEqual(max(d["xval"]), 0.4)
                self.assertGreater(np.min(abs(d["xtrain"][:, None] - d["xval"][None, :])), 1e-12)

    def test_training_constant_refinement_recovers_frequency(self):
        X = np.linspace(0, 4 * np.pi, 160)[:, None]
        y = np.cos(X[:, 0])
        expression = sp.cos(sp.Float(1.01) * T)
        fitted = refine(expression, X, y)
        self.assertLess(np.max(abs(evaluate_expr(fitted, X) - y)), 1e-10)

    def test_certificate_rejects_wrong_frequency_despite_correct_syntax(self):
        p = parameters("uniform", 1)
        good = certify("cos(t)", p)
        wrong = certify("cos(1.01*t)", p)
        self.assertTrue(good["coefficients_match_theory_within_tolerance"])
        self.assertFalse(wrong["coefficients_match_theory_within_tolerance"])


class DriverChecks(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((REPO / "config.json").read_text())

    def test_full_plan_preserves_all_14_manifests_and_447_jobs(self):
        plan = driver.build_plan(self.config, "all")
        called = [s["command"][3] for s in plan if s["command"][2] == "acoustic_technique.dispatch"]
        manifests = sorted(p.name for p in (REPO / "experiments/manifests").glob("*.json"))
        self.assertEqual(len(called), 14)
        self.assertEqual(sorted(called), manifests)
        self.assertLess(
            called.index("baseline_manifest.json"), called.index("polish_manifest.json")
        )
        self.assertLess(
            called.index("joint_baseline_manifest.json"),
            called.index("joint_refined_manifest.json"),
        )
        self.assertEqual(
            sum(len(json.loads((REPO / "experiments/manifests" / n).read_text())) for n in called),
            447,
        )

    def test_final_audit_and_pruning_precede_summary(self):
        labels = [s["label"] for s in driver.build_plan(self.config, "post")]
        self.assertLess(labels.index("final_audit"), labels.index("prune_negligible_constants"))
        self.assertLess(labels.index("prune_negligible_constants"), labels.index("summarize"))
        self.assertLess(labels.index("supplementary_figures"), labels.index("archive_audit"))
        self.assertEqual(labels[-1], "actual_result_check")

    def test_joint_appendix_keeps_900_second_16_thread_run_separate(self):
        jobs = json.loads((REPO / "experiments/manifests/joint_extended_manifest.json").read_text())
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["stage"], "focused_extended")
        self.assertEqual(jobs[0]["seconds"], 900)
        self.assertEqual(jobs[0]["threads"], 16)


if __name__ == "__main__":
    unittest.main()
