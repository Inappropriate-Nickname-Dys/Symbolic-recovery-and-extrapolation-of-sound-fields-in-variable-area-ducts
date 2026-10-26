"""Supplementary solver provenance, initial conditions, joint OOD and roots."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import time
from pathlib import Path
import numpy as np
import sympy as sp
from .acoustic_core import (
    ROOT,
    T,
    KINDS,
    parameters,
    solve,
    analytic,
    dump,
    evaluate_expr,
    metrics,
    location_metrics,
)
from .structure_certificate import harmonic_coefficients, close_check


def read(path):
    return json.loads(path.read_text())


def joint_family(e, eta, tolerance=1e-6):
    checks = {}
    try:
        numerator, denominator = sp.fraction(sp.cancel(e))
        dp = sp.Poly(sp.expand(denominator), T, eta)
        if dp.total_degree() > 1:
            raise ValueError("nonlinear joint denominator")
        scale = float(dp.coeff_monomial(T))
        if abs(scale) < 1e-15:
            raise ValueError("zero phase coefficient in denominator")
        checks["denominator_eta_coefficient"] = close_check(
            float(dp.coeff_monomial(eta)) / scale, 1.0, tolerance
        )
        checks["denominator_constant"] = close_check(
            float(dp.coeff_monomial(1)) / scale, 0.0, tolerance
        )
        npoly = sp.Poly(sp.expand(numerator / scale), eta)
        if npoly.degree() > 1:
            raise ValueError("nonlinear eta numerator or phase")
        for name, coef, a, b in [
            ("eta_numerator", npoly.nth(1), 1.0, 0.0),
            ("constant_numerator", npoly.nth(0), 0.0, 1.0),
        ]:
            if coef == 0:
                h = dict(alpha=0.0, beta=0.0, frequency=1.0, common_frequency_within_tolerance=True)
            else:
                h = harmonic_coefficients(coef, tolerance)
            if not h["common_frequency_within_tolerance"]:
                raise ValueError("multiple independent frequencies")
            checks[name + "_cosine"] = close_check(h["alpha"], a, tolerance)
            checks[name + "_sine"] = close_check(h["beta"], b, tolerance)
            checks[name + "_frequency"] = close_check(h["frequency"], 1.0, tolerance)
        return dict(
            family_structure_matched=True,
            coefficients_match_theory_within_tolerance=all(c["passed"] for c in checks.values()),
            parameter_checks=checks,
            tolerance=tolerance,
        )
    except Exception as exc:
        return dict(
            family_structure_matched=False,
            coefficients_match_theory_within_tolerance=False,
            parameter_checks=checks,
            tolerance=tolerance,
            unclassified_reason=str(exc),
        )


def main():
    solver = []
    for kind in KINDS:
        for index in range(3):
            p = parameters(kind, index)
            times = []
            for _ in range(3):
                start = time.monotonic()
                s = solve(p)
                times.append(time.monotonic() - start)
            x = np.linspace(0, 1, 2001)
            solver.append(
                dict(
                    **p,
                    method="DOP853",
                    rtol=1e-11,
                    atol=1e-13,
                    success=s.success,
                    message=s.message,
                    nfev=s.nfev,
                    accepted_steps=len(s.t) - 1,
                    solve_seconds_median=float(np.median(times)),
                    repeated_provenance_check=True,
                    max_ode_analytic_error=float(np.max(abs(s.sol(x)[0] - analytic(x, p)))),
                )
            )
    dump(ROOT / "experiment_00_generator" / "solver_diagnostics.json", solver)
    identities = []
    for experiment in ["experiment_01_recovery", "experiment_03_sampling"]:
        for f in (ROOT / experiment).glob("*/result.json"):
            r = read(f)
            e = sp.sympify(r["selected"]["refined"], locals={"t": T})
            k = r["parameters"]["kappa"]
            canonical = sp.cancel(e)
            dump(
                f.parent / "canonical_expression.json",
                dict(
                    expression=str(canonical),
                    original=str(e),
                    method="algebraic cancellation only; continuous extension at removable singularities; no theoretical coefficient replacement",
                ),
            )
            p0 = float(evaluate_expr(e, np.zeros((1, 1)))[0])
            dp0 = float(evaluate_expr(sp.diff(e, T), np.zeros((1, 1)))[0])
            certificate = f.parent / "structure_certificate.json"
            c = read(certificate) if certificate.exists() else {}
            identities.append(
                dict(
                    experiment=experiment,
                    job=r["job"],
                    stage=r["stage"],
                    kind=r["parameters"]["kind"],
                    initial_pressure_error=abs(p0 - 1),
                    initial_derivative_error_over_kappa=abs(dp0),
                    webster_relative_residual_rms=r["metrics"].get("webster_relative_residual_rms"),
                    family_structure_matched=c.get("family_structure_matched"),
                    coefficients_match_theory_within_tolerance=c.get(
                        "coefficients_match_theory_within_tolerance"
                    ),
                    exact_identity_proved=c.get("exact_identity_proved"),
                    parameter_checks=c.get("parameter_checks"),
                )
            )
    dump(ROOT / "docs" / "pressure_initial_conditions_and_certificates.json", identities)
    joint = []
    ETA = sp.Symbol("eta", real=True)
    target = (ETA * sp.cos(T) + sp.sin(T)) / (ETA + T)
    for f in (ROOT / "experiment_04_joint").glob("*/result.json"):
        r = read(f)
        e = sp.sympify(r["selected"]["refined"], locals={"t": T, "eta": ETA})
        d = np.load(f.parent / "data.npz")
        pred = np.load(f.parent / "predictions.npz")
        residual = sp.diff(e, T, 2) + 2 * sp.diff(e, T) / (ETA + T) + e
        cert = dict(
            exact_identity_proved=bool(sp.simplify(sp.trigsimp(e - target)) == 0),
            exact_webster_ode_proved=bool(sp.simplify(sp.trigsimp(residual)) == 0),
            exact_p0_proved=bool(sp.simplify(e.subs(T, 0) - 1) == 0),
            exact_dp0_proved=bool(sp.simplify(sp.diff(e, T).subs(T, 0)) == 0),
        )
        cert.update(joint_family(e, ETA))
        canonical = sp.cancel(e)
        dump(
            f.parent / "canonical_expression.json",
            dict(
                expression=str(canonical),
                original=str(e),
                method="algebraic cancellation only; export the continuous extension when factors cancel",
                exact_identity_proved=bool(sp.simplify(sp.trigsimp(canonical - target)) == 0),
            ),
        )
        etas = np.array(
            [8.0, 10.0, 12.0, 14.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 31.0, 34.0, 37.0, 40.0]
        )
        initial = evaluate_expr(e, np.column_stack([np.zeros_like(etas), etas]))
        derivative = evaluate_expr(sp.diff(e, T), np.column_stack([np.zeros_like(etas), etas]))
        cert["max_initial_pressure_error"] = float(np.max(abs(initial - 1)))
        cert["max_initial_phase_derivative_error"] = float(np.max(abs(derivative)))
        tests = {}
        nodes = []
        for region in ["geometry_ood", "joint_ood"]:
            X, y, yp = d[region + "_X"], d[region + "_y"], pred["PySR_" + region]
            for name, mask in [("low_eta", X[:, 1] <= 14), ("high_eta", X[:, 1] >= 31)]:
                tests[region + "_" + name] = metrics(y[mask], yp[mask])
        # xi=t/(10*pi) recovers exactly the predefined [0.4,1] OOD boundary.
        Xg, yg, pg = d["geometry_ood_X"], d["geometry_ood_y"], pred["PySR_geometry_ood"]
        Xj, yj, pj = d["joint_ood_X"], d["joint_ood_y"], pred["PySR_joint_ood"]
        for eta in np.unique(Xg[:, 1]):
            mg, mj = Xg[:, 1] == eta, Xj[:, 1] == eta
            x = np.r_[Xg[mg, 0], Xj[mj, 0]] / (10 * np.pi)
            y = np.r_[yg[mg], yj[mj]]
            yp = np.r_[pg[mg], pj[mj]]
            nodes.append(dict(eta=float(eta), metrics=location_metrics(x, y, yp, 10 * np.pi)))
        out = dict(
            seed=r["seed"],
            stage=r["stage"],
            certificate=cert,
            split_ood_metrics=tests,
            nodes_and_extrema=nodes,
            note="independent posterior verification; never used for candidate selection",
        )
        dump(f.parent / "independent_verification.json", out)
        joint.append(out)
    dump(ROOT / "experiment_04_joint" / "independent_verification_summary.json", joint)
    print(
        "solver", len(solver), "1D certificates", len(identities), "joint", len(joint), flush=True
    )


if __name__ == "__main__":
    main()
