"""Webster IVP data and evaluation. Run on Linux, never Windows Python."""

import json
import time
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, brentq, linear_sum_assignment
from scipy.interpolate import CubicSpline

from .paths import ROOT

KINDS = ["uniform", "conical", "exponential", "catenoidal"]
T = sp.Symbol("t", real=True)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def parameters(kind, index):
    return dict(
        kind=kind,
        index=index,
        kappa=float((8 + 2 * index) * np.pi),
        mu=[0.8, 1.4, 2.0][index],
        b=[0.35, 0.65, 1.0][index],
    )


def log_slope(x, p):
    kind, mu, b = p["kind"], p["mu"], p["b"]
    if kind == "uniform":
        return np.zeros_like(x)
    if kind == "conical":
        return 2 / (b + x)
    if kind == "exponential":
        return np.zeros_like(x) + 2 * mu
    return 2 * mu * np.tanh(mu * x)


def area(x, p):
    if p["kind"] == "uniform":
        return np.ones_like(x)
    if p["kind"] == "conical":
        return (1 + x / p["b"]) ** 2
    if p["kind"] == "exponential":
        return np.exp(2 * p["mu"] * x)
    return np.cosh(p["mu"] * x) ** 2


def analytic(x, p):
    k, mu, b = p["kappa"], p["mu"], p["b"]
    q = np.sqrt(k * k - mu * mu)
    if p["kind"] == "uniform":
        return np.cos(k * x)
    if p["kind"] == "conical":
        return (b * np.cos(k * x) + np.sin(k * x) / k) / (b + x)
    if p["kind"] == "exponential":
        return np.exp(-mu * x) * (np.cos(q * x) + mu / q * np.sin(q * x))
    return np.cos(q * x) / np.cosh(mu * x)


def solve(p, end=1.0, rtol=1e-11, atol=1e-13):
    def fun(x, y):
        return [y[1], -float(log_slope(x, p)) * y[1] - p["kappa"] ** 2 * y[0]]

    return solve_ivp(
        fun, (0, end), [1.0, 0.0], method="DOP853", rtol=rtol, atol=atol, dense_output=True
    )


def generate(p, n=160, sampling="uniform", seed=0):
    rng = np.random.default_rng(seed)
    if sampling == "uniform":
        xtr = np.linspace(0, 0.4, n)
    else:
        xtr = np.r_[0.0, np.sort(rng.uniform(0, 0.4, n - 2)), 0.4]
    # Validation excludes every training coordinate; independent points inside ID only.
    xv = (np.arange(240) + 0.371) / 240 * 0.4
    for j in range(len(xv)):
        if np.min(abs(xtr - xv[j])) < 1e-12:
            xv[j] += 1e-7
    x = np.linspace(0, 1, 2001)
    sol = solve(p)
    if not sol.success:
        raise RuntimeError(sol.message)
    yp = sol.sol(x)[0]
    err = float(np.max(abs(yp - analytic(x, p))))
    if err >= 1e-8:
        raise AssertionError("ODE/analytic discrepancy " + str(err))
    return dict(
        x=x,
        y=yp,
        xtrain=xtr,
        ytrain=sol.sol(xtr)[0],
        xval=xv,
        yval=sol.sol(xv)[0],
        Xtrain=(xtr * p["kappa"])[:, None],
        Xval=(xv * p["kappa"])[:, None],
        X=(x * p["kappa"])[:, None],
    )


def evaluate_expr(expr, X):
    variables = [T] if X.shape[1] == 1 else [T, sp.Symbol("eta", real=True)]
    f = sp.lambdify(variables, expr, "numpy")
    with np.errstate(all="ignore"):
        y = np.asarray(f(*X.T), dtype=float)
    return np.broadcast_to(y, (len(X),)).copy()


def refine(expr, X, y):
    # No oracle constants. Optimize only constants present in the discovered tree.
    consts = sorted(expr.atoms(sp.Float), key=str)
    if not consts or len(consts) > 18:
        return expr
    coeff = sp.symbols("c:" + str(len(consts)))
    template = expr.xreplace(dict(zip(consts, coeff)))
    variables = [T] if X.shape[1] == 1 else [T, sp.Symbol("eta", real=True)]
    f = sp.lambdify([*variables, *coeff], template, "numpy")
    start = np.array([float(c) for c in consts])

    def residual(c):
        with np.errstate(all="ignore"):
            pred = np.broadcast_to(np.asarray(f(*X.T, *c), dtype=float), y.shape)
        return np.nan_to_num(pred - y, nan=1e10, posinf=1e10, neginf=-1e10)

    try:
        fit = least_squares(
            residual,
            start,
            jac="3-point",
            max_nfev=1200,
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            x_scale="jac",
        )
        if np.mean(residual(fit.x) ** 2) <= np.mean(residual(start) ** 2):
            return template.subs(dict(zip(coeff, [sp.Float(v, 17) for v in fit.x])))
    except (ValueError, OverflowError, TypeError, FloatingPointError):
        pass
    return expr


def metrics(y, yp):
    if not np.all(np.isfinite(yp)):
        return dict(mae=None, nrmse=None, max_abs=None, finite=False)
    d = yp - y
    rms = np.sqrt(np.mean(y * y))
    return dict(
        mae=float(np.mean(abs(d))),
        nrmse=float(np.sqrt(np.mean(d * d)) / max(rms, 1e-15)),
        max_abs=float(np.max(abs(d))),
        finite=True,
    )


def location_metrics(x, y, yp, kappa, endpoint_tolerance=0.0):
    if not np.all(np.isfinite(yp)):
        return dict(finite=False)

    # Same cubic interpolation and bracket rules for every method.
    def locations(z):
        s = CubicSpline(x, z)
        roots = s.roots(extrapolate=False)
        peaks = s.derivative().roots(extrapolate=False)
        return (
            roots[(roots > 0.4 + endpoint_tolerance) & (roots < 1 - endpoint_tolerance)],
            peaks[(peaks > 0.4 + endpoint_tolerance) & (peaks < 1 - endpoint_tolerance)],
            s,
        )

    zr, pr, rs = locations(y)
    zp, pp, ps = locations(yp)

    def match(a, b):
        threshold = np.pi / kappa / 2  # quarter wavelength in xi
        if len(a) == 0 or len(b) == 0:
            return [], len(a), len(b)
        ia, ib = linear_sum_assignment(abs(a[:, None] - b[None, :]))
        pairs = [(int(i), int(j)) for i, j in zip(ia, ib) if abs(a[i] - b[j]) <= threshold]
        return pairs, len(a) - len(pairs), len(b) - len(pairs)

    out = dict(finite=True)
    for name, a, b in [("nodes", zr, zp), ("extrema", pr, pp)]:
        pairs, miss, false = match(a, b)
        errors = [abs(a[i] - b[j]) for i, j in pairs]
        out[name] = dict(
            reference=a.tolist(),
            predicted=b.tolist(),
            missed=miss,
            false=false,
            matched=len(pairs),
            mean_xi_error=float(np.mean(errors)) if errors else None,
            max_xi_error=float(np.max(errors)) if errors else None,
            mean_wavelength_error=float(np.mean(errors) / (2 * np.pi / kappa)) if errors else None,
        )
        if name == "extrema":
            amps = [abs(float(rs(a[i])) - float(ps(b[j]))) for i, j in pairs]
            out[name]["mean_amplitude_error"] = float(np.mean(amps)) if amps else None
    return out


def field_metrics(data, yp, p, expr=None):
    x, y = data["x"], data["y"]
    out = {
        "full": metrics(y, yp),
        "id": metrics(y[x <= 0.4], yp[x <= 0.4]),
        "ood": metrics(y[x > 0.4], yp[x > 0.4]),
    }
    for left, right in [(0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        m = (x > left) & (x <= right)
        out[str((left, right))] = metrics(y[m], yp[m])
    # Predeclared usable-distance criterion: cumulative OOD NRMSE <= 1%.
    good = 0.0
    for right in np.arange(0.45, 1.001, 0.05):
        m = (x > 0.4) & (x <= right + 1e-12)
        met = metrics(y[m], yp[m])
        if met["nrmse"] is None or met["nrmse"] > 0.01:
            break
        good = float(right - 0.4)
    out["usable_ood_length_xi"] = good
    out["locations"] = location_metrics(x, y, yp, p["kappa"])
    if np.all(np.isfinite(yp)):
        if expr is not None:
            p1 = evaluate_expr(sp.diff(expr, T), data["X"]) * p["kappa"]
            p2 = evaluate_expr(sp.diff(expr, T, 2), data["X"]) * p["kappa"] ** 2
        else:
            spl = CubicSpline(x, yp)
            p1 = spl(x, 1)
            p2 = spl(x, 2)
        residual = p2 + log_slope(x, p) * p1 + p["kappa"] ** 2 * yp
        out["webster_relative_residual_rms"] = float(
            np.sqrt(np.mean(residual[2:-2] ** 2)) / (p["kappa"] ** 2 * np.sqrt(np.mean(y * y)))
        )
    # Dense agreement is not a symbolic identity proof.
    if expr is not None:
        trig = expr.has(sp.sin, sp.cos)
        envelope = (
            p["kind"] == "uniform"
            or (p["kind"] == "exponential" and expr.has(sp.exp))
            or (p["kind"] == "catenoidal" and expr.has(sp.cosh))
            or (
                p["kind"] == "conical"
                and any(z.exp.is_negative and z.base.has(T) for z in expr.atoms(sp.Pow))
            )
        )
        out["structure_numerically_consistent"] = bool(
            trig
            and envelope
            and out["full"]["max_abs"] is not None
            and out["full"]["max_abs"] < 1e-6
            and out.get("webster_relative_residual_rms", 1) < 1e-5
        )
        out["exact_identity_proved"] = False
    return out


def generator_experiment():
    folder = ROOT / "experiment_00_generator"
    records = []
    for kind in KINDS:
        for index in range(3):
            p = parameters(kind, index)
            d = generate(p)
            start = time.monotonic()
            tight = solve(p, rtol=2e-13, atol=2e-15)
            elapsed = time.monotonic() - start
            err = float(np.max(abs(d["y"] - analytic(d["x"], p))))
            delta = float(np.max(abs(d["y"] - tight.sol(d["x"])[0])))
            records.append(
                dict(
                    **p,
                    max_ode_analytic_error=err,
                    refinement_difference=delta,
                    tight_seconds=elapsed,
                )
            )
            np.savez_compressed(folder / (kind + "_" + str(index) + ".npz"), **d)
            dump(folder / (kind + "_" + str(index) + ".json"), p)
    dump(folder / "verification.json", records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    generator_experiment()
