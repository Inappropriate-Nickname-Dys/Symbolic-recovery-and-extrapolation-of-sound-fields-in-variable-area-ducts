"""Certify discovered parametric families without reading any OOD metrics.

This is deliberately independent of acoustic_core/run_search.  A successful
floating-parameter check is not an exact identity proof.  Expressions outside
this conservative grammar are left unclassified, rather than called failures
of mathematical recoverability. Run on the Linux server in the gat environment.
"""

import argparse
import json
import math
from pathlib import Path
from datetime import datetime, timezone
import sympy as sp

from .paths import ROOT

T = sp.Symbol("t", real=True)


def scalar(value):
    if sp.sympify(value).free_symbols:
        raise ValueError("coefficient depends on a variable")
    value = complex(sp.N(value, 18))
    if abs(value.imag) > 1e-13 or not math.isfinite(value.real):
        raise ValueError("coefficient is not a finite real number")
    return float(value.real)


def affine(argument):
    poly = sp.Poly(sp.expand(argument), T)
    if poly.degree() > 1:
        raise ValueError("phase or envelope argument is not affine")
    return scalar(poly.nth(1)), scalar(poly.nth(0))


def close_check(observed, expected, tolerance):
    error = abs(observed - expected)
    scale = max(1.0, abs(expected))
    return dict(
        observed=observed,
        expected=expected,
        absolute_error=error,
        scaled_error=error / scale,
        allowed_absolute_error=tolerance * scale,
        passed=bool(error <= tolerance * scale),
    )


def harmonic_coefficients(expression, tolerance):
    """Algebraically extract a finite linear sum of affine sin/cos terms."""
    atoms = sorted(expression.atoms(sp.sin, sp.cos), key=str)
    if not atoms:
        raise ValueError("no harmonic term")
    generators = [sp.Dummy("harmonic_%d" % i) for i in range(len(atoms))]
    polynomial = sp.Poly(sp.expand(expression.xreplace(dict(zip(atoms, generators)))), *generators)
    if polynomial.total_degree() > 1:
        raise ValueError("harmonics are multiplied or nested")
    constant = polynomial.coeff_monomial(1)
    # A nonzero background is not silently rounded into the target family.
    if constant.is_zero is not True:
        raise ValueError("nonzero non-harmonic background: " + str(constant))
    terms = []
    alpha = beta = 0.0
    for atom, generator in zip(atoms, generators):
        coefficient = scalar(polynomial.coeff_monomial(generator))
        if coefficient == 0:
            continue
        frequency, phase = affine(atom.args[0])
        if frequency == 0:
            raise ValueError("zero-frequency harmonic")
        sign = 1.0 if frequency > 0 else -1.0
        if atom.func == sp.cos:
            a = coefficient * math.cos(phase)
            b = -sign * coefficient * math.sin(phase)
        else:
            a = coefficient * math.sin(phase)
            b = sign * coefficient * math.cos(phase)
        alpha += a
        beta += b
        terms.append(
            dict(
                atom=str(atom),
                coefficient=coefficient,
                frequency=abs(frequency),
                phase=phase,
                cosine_coefficient=a,
                sine_coefficient=b,
            )
        )
    if not terms:
        raise ValueError("empty harmonic expansion")
    frequency = terms[0]["frequency"]
    same_frequency = all(
        abs(term["frequency"] - frequency) <= tolerance * max(1.0, frequency) for term in terms
    )
    return dict(
        alpha=alpha,
        beta=beta,
        frequency=frequency,
        terms=terms,
        common_frequency_within_tolerance=same_frequency,
        frequency_spread=max(z["frequency"] for z in terms) - min(z["frequency"] for z in terms),
    )


def certify(expression, parameters, tolerance=1e-6):
    expression = sp.sympify(expression, locals={"t": T})
    if expression.free_symbols - {T}:
        raise ValueError("unexpected free variables")
    kind = parameters["kind"]
    kappa = float(parameters["kappa"])
    mu = float(parameters["mu"])
    b = float(parameters["b"])
    ratio = mu / kappa
    q_ratio = math.sqrt(1.0 - ratio * ratio)
    eta = kappa * b
    expr = sp.cancel(sp.together(expression))
    checks = {}
    observed_envelope = {}
    if kind == "uniform":
        harmonic = expr
        expected_alpha, expected_beta, expected_frequency = 1.0, 0.0, 1.0
        family = "A*cos(w*t)+B*sin(w*t)"
    elif kind == "conical":
        numerator, denominator = sp.fraction(expr)
        slope, offset = affine(denominator)
        if slope == 0:
            raise ValueError("missing affine conical denominator")
        shift = offset / slope
        if shift <= 0:
            raise ValueError("conical denominator has invalid positive-domain pole")
        harmonic = sp.cancel(numerator / slope)
        checks["denominator_eta"] = close_check(shift, eta, tolerance)
        observed_envelope = dict(
            denominator_slope=slope, denominator_offset=offset, normalized_shift=shift
        )
        expected_alpha, expected_beta, expected_frequency = eta, 1.0, 1.0
        family = "[A*cos(w*t)+B*sin(w*t)]/(t+eta_hat)"
    elif kind == "exponential":
        exponentials = sorted(expr.atoms(sp.exp), key=str)
        if len(exponentials) != 1:
            raise ValueError("requires exactly one exponential envelope")
        envelope = exponentials[0]
        rate, offset = affine(envelope.args[0])
        harmonic = sp.simplify(expr / envelope)
        # Move the constant exp(offset) into harmonic amplitudes.
        harmonic *= sp.exp(sp.Float(offset, 18))
        checks["exponential_rate"] = close_check(rate, -ratio, tolerance)
        observed_envelope = dict(rate=rate, offset=offset)
        expected_alpha, expected_beta, expected_frequency = 1.0, ratio / q_ratio, q_ratio
        family = "exp(d*t)*[A*cos(w*t)+B*sin(w*t)]"
    elif kind == "catenoidal":
        hyperbolics = sorted(expr.atoms(sp.cosh), key=str)
        if len(hyperbolics) != 1:
            raise ValueError("requires exactly one reciprocal-cosh envelope")
        envelope = hyperbolics[0]
        rate, offset = affine(envelope.args[0])
        harmonic = sp.cancel(expr * envelope)
        checks["cosh_rate"] = close_check(abs(rate), ratio, tolerance)
        checks["cosh_offset"] = close_check(offset, 0.0, tolerance)
        observed_envelope = dict(rate=rate, offset=offset)
        expected_alpha, expected_beta, expected_frequency = 1.0, 0.0, q_ratio
        family = "[A*cos(w*t)+B*sin(w*t)]/cosh(h*t+z)"
    else:
        raise ValueError("unknown geometry: " + str(kind))
    decomposition = harmonic_coefficients(sp.expand(harmonic), tolerance)
    checks["cosine_coefficient"] = close_check(decomposition["alpha"], expected_alpha, tolerance)
    checks["sine_coefficient"] = close_check(decomposition["beta"], expected_beta, tolerance)
    for i, term in enumerate(decomposition["terms"]):
        checks["frequency_%d" % i] = close_check(term["frequency"], expected_frequency, tolerance)
    passed = decomposition["common_frequency_within_tolerance"] and all(
        check["passed"] for check in checks.values()
    )
    # Only the literal, algebraically equivalent uniform target is proved here.
    exact = bool(kind == "uniform" and sp.simplify(expr - sp.cos(T)) == 0)
    initial = dict(
        value_error=scalar(expr.subs(T, 0)) - 1.0,
        derivative_in_t_error=scalar(sp.diff(expr, T).subs(T, 0)),
    )
    return dict(
        grammar_matched=True,
        family_structure_matched=True,
        parametric_family=family,
        coefficients_match_theory_within_tolerance=bool(passed),
        exact_identity_proved=exact,
        status=(
            "exact_identity_proved"
            if exact
            else (
                "parameterized_family_recovered"
                if passed
                else "family_found_but_coefficients_do_not_match"
            )
        ),
        envelope=observed_envelope,
        harmonic_decomposition=decomposition,
        parameter_checks=checks,
        initial_condition_errors=initial,
        proof_scope=(
            "Exact SymPy simplification of cos(t) identity."
            if exact
            else "Algebraic family extraction plus floating coefficient checks; "
            "not a proof of exact floating-expression identity."
        ),
    )


def run(experiment, tolerance):
    directory = ROOT / experiment
    records = []
    for path in sorted(directory.rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        selected = result.get("selected", {})
        expression = selected.get("refined")
        if expression is None or "parameters" not in result:
            continue
        certificate = dict(
            job=result.get("job", path.parent.name),
            parameters=result["parameters"],
            expression=expression,
            source_result=str(path.relative_to(ROOT)),
            tolerance=tolerance,
            tolerance_definition="abs(error)<=tolerance*max(1,abs(expected))",
            evidence_inputs="selected.refined and parameters only; no training or OOD metrics",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        try:
            certificate.update(certify(expression, result["parameters"], tolerance))
        except Exception as error:
            certificate.update(
                grammar_matched=False,
                family_structure_matched=False,
                coefficients_match_theory_within_tolerance=False,
                exact_identity_proved=False,
                status="not_classified_by_conservative_grammar",
                reason=str(error),
            )
        output = path.parent / "structure_certificate.json"
        output.write_text(
            json.dumps(certificate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        records.append(
            dict(
                job=certificate["job"],
                kind=result["parameters"]["kind"],
                stage=result.get("stage"),
                seed=result.get("seed"),
                index=result["parameters"].get("index"),
                status=certificate["status"],
                parameterized_recovery=certificate["coefficients_match_theory_within_tolerance"],
                exact_identity_proved=certificate["exact_identity_proved"],
                certificate=str(output.relative_to(ROOT)),
            )
        )
    summary = dict(
        experiment=experiment,
        tolerance=tolerance,
        certification_method="symbolic family extraction; theory coefficient checks; no OOD metrics",
        total=len(records),
        parameterized_recovered=sum(r["parameterized_recovery"] for r in records),
        exact_identity_proved=sum(r["exact_identity_proved"] for r in records),
        records=records,
    )
    output = directory / "structure_certificates_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False))
    return summary


def self_check():
    """Check false-positive rejection as well as equivalent phase forms."""
    parameters = dict(kind="uniform", kappa=10 * math.pi, mu=1.4, b=0.65)
    cases = [
        ("uniform", "cos(t)", True, True),
        ("uniform", "1.1*cos(t)", False, False),
        ("uniform", "cos(1.01*t)", False, False),
        ("uniform", "cos(t)+0.000001", False, False),
        (
            "conical",
            "20.444822962087091*cos(t-0.048931661804077299)/(t+20.420352248482676)",
            True,
            False,
        ),
        (
            "conical",
            "20.444822962087091*cos(t-0.048931661804077299)/(t+21.420352248482676)",
            False,
            False,
        ),
        (
            "exponential",
            "-1.0009944289680812*exp(-0.044563384065494593*t)*sin(0.99900655893711476*t-1.6153744737593463)",
            True,
            False,
        ),
        ("catenoidal", "cos(0.9990065589370325*t)/cosh(0.044563384065941715*t)", True, False),
        ("catenoidal", "cos(0.9990065589370325*t)/cosh(0.044563384065941715*t+0.1)", False, False),
    ]
    for kind, expression, expected_recovery, expected_exact in cases:
        parameters["kind"] = kind
        try:
            certificate = certify(expression, parameters)
            recovered = certificate["coefficients_match_theory_within_tolerance"]
            exact = certificate["exact_identity_proved"]
        except ValueError:
            recovered = exact = False
        assert recovered == expected_recovery, (kind, expression, "recovery", recovered)
        assert exact == expected_exact, (kind, expression, "exact", exact)
    print(json.dumps(dict(self_check="passed", cases=len(cases))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment_01_recovery")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    run(args.experiment, args.tolerance)
