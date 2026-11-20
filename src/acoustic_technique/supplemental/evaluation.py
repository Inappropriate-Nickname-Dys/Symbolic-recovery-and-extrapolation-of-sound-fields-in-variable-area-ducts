"""Post-freeze clean tests, harmonic coefficients, and physical residuals."""

from .common import T, SIGMA, THETA, dump, evaluate, prediction_metrics, ROOT
from .data import reference_derivatives, grid, implicit
import numpy as np
import sympy as sp
from scipy.special import jv


def nonlinear_metrics(d, pred, derivatives=None, predict=None):
    X, y = d["Xtest"], d["ytest"]
    s = X[:, 0]
    out = {"ood": prediction_metrics(y, pred), "segments": {}}
    for a, b in [(0.32, 0.48), (0.48, 0.64), (0.64, 0.8)]:
        m = (s > a) & (s <= b + 1e-12)
        out["segments"]["%g_%g" % (a, b)] = prediction_metrics(y[m], pred[m])
    ns = len(d["sigma_test"])
    nt = len(d["theta_test"])
    theta = d["theta_test"]
    P = pred.reshape(ns, nt)
    sigma = d["sigma_test"]
    orders = np.arange(1, 11)
    true = (
        2
        * jv(orders[None, :], sigma[:, None] * orders[None, :])
        / (sigma[:, None] * orders[None, :])
    )
    sine = 2 / nt * P @ np.sin(theta[:, None] * orders[None, :])
    cosine = 2 / nt * P @ np.cos(theta[:, None] * orders[None, :])
    amp = np.sqrt(sine * sine + cosine * cosine)
    finite = bool(np.all(np.isfinite(P)))
    h = dict(
        finite=finite,
        orders=orders.tolist(),
        sigma=sigma,
        sine=sine,
        cosine=cosine,
        amplitude=amp,
        reference_amplitude=true,
        dc=np.mean(P, axis=1),
    )
    if finite:
        for n in [5, 10]:
            denominator = np.sum(true[:, :n] ** 2)
            h["amplitude_error_1_%d" % n] = float(
                np.sqrt(np.sum((amp[:, :n] - true[:, :n]) ** 2) / denominator)
            )
            h["coefficient_error_1_%d" % n] = float(
                np.sqrt(np.sum((sine[:, :n] - true[:, :n]) ** 2 + cosine[:, :n] ** 2) / denominator)
            )
        h["relative_amplitude_error_by_order"] = np.sqrt(
            np.sum((amp - true) ** 2, axis=0) / np.sum(true**2, axis=0)
        )
        h["relative_coefficient_error_by_order"] = np.sqrt(
            np.sum((sine - true) ** 2 + cosine**2, axis=0) / np.sum(true**2, axis=0)
        )
    out["harmonics"] = h
    if derivatives is not None:
        ps, pt = derivatives
        R = ps - pred * pt
        refps, _ = reference_derivatives(X, y)
        absolute = float(np.sqrt(np.mean(R * R)))
        denominator = float(np.sqrt(np.mean(refps * refps)))
        out["burgers_residual_absolute_rms"] = absolute
        out["burgers_residual_normalization_rms"] = denominator
        out["burgers_residual_normalized_rms"] = absolute / max(denominator, 1e-15)
    if predict is not None:
        out["id_test"] = prediction_metrics(d["yid"], predict(d["Xid"]))
        xb = np.column_stack([np.zeros(nt), theta])
        out["inlet_nrmse"] = prediction_metrics(np.sin(theta), predict(xb))["nrmse"]
        a = np.column_stack([sigma, np.zeros(ns)])
        b = np.column_stack([sigma, np.full(ns, 2 * np.pi)])
        out["periodicity_endpoint_rms"] = float(np.sqrt(np.mean((predict(a) - predict(b)) ** 2)))
        waves = {}
        for distance in [0.2, 0.4, 0.5, 0.6, 0.8]:
            xx = np.column_stack([np.full(nt, distance), theta])
            waves[str(distance)] = dict(
                theta=theta, truth=implicit(distance, theta), prediction=predict(xx)
            )
        out["waveforms"] = waves
    return out


def simple_metrics(job, d, pred, expr=None, derivatives=None):
    from ..acoustic_core import field_metrics, location_metrics, log_slope
    from ..structure_certificate import certify

    p = d["parameters"]
    data = {"x": d["x"], "y": d["ytest"], "X": d["Xtest"]}
    out = field_metrics(data, pred, p, expr)
    out["locations"] = location_metrics(
        d["x"], d["ytest"], pred, p["kappa"], endpoint_tolerance=1e-6
    )
    if derivatives is not None:
        p1, p2 = derivatives
        p1 = p1 * p["kappa"]
        p2 = p2 * p["kappa"] ** 2
        r = p2 + log_slope(d["x"], p) * p1 + p["kappa"] ** 2 * pred
        out["webster_relative_residual_rms"] = float(
            np.sqrt(np.mean(r[2:-2] ** 2)) / (p["kappa"] ** 2 * np.sqrt(np.mean(d["ytest"] ** 2)))
        )
        out["residual_differentiation"] = "analytic or automatic differentiation"
    else:
        out["residual_differentiation"] = (
            "symbolic expression" if expr is not None else "cubic spline numerical derivative"
        )
    if expr is not None:
        try:
            certificate = certify(expr, p, 1e-6)
            certificate["noise_structure_match"] = bool(
                certificate["family_structure_matched"]
                and certificate["harmonic_decomposition"]["common_frequency_within_tolerance"]
            )
            certificate["max_scaled_coefficient_error"] = max(
                v["scaled_error"] for v in certificate["parameter_checks"].values()
            )
            certificate["noise_interpretation"] = (
                "Report grammar match and coefficient errors separately; do not demand 1e-6 in noisy experiments."
            )
        except Exception as e:
            certificate = dict(
                noise_structure_match=False,
                coefficients_match_theory_within_tolerance=False,
                status="not_classified_by_conservative_grammar",
                reason=str(e),
            )
        out["structure_certificate"] = certificate
    return out


def evaluate_model(job, d, pred, expr=None, predict=None, derivatives=None):
    if job["scenario"] == "nonlinear_burgers":
        if expr is not None:
            derivatives = (
                evaluate(sp.diff(expr, SIGMA), d["Xtest"]),
                evaluate(sp.diff(expr, THETA), d["Xtest"]),
            )
            predict = lambda x: evaluate(expr, x)
        return nonlinear_metrics(d, pred, derivatives, predict)
    return simple_metrics(job, d, pred, expr, derivatives)


def save_evaluation(folder, metrics):
    """Keep readable scalar metrics separate from the full numerical plotting data."""
    metrics = dict(metrics)
    if "harmonics" in metrics:
        h = metrics["harmonics"]
        arrays = {k: v for k, v in h.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(folder / "harmonics.npz", **arrays)
        metrics["harmonics"] = {k: v for k, v in h.items() if not isinstance(v, np.ndarray)}
    if "waveforms" in metrics:
        arrays = {s + "_" + k: v for s, w in metrics.pop("waveforms").items() for k, v in w.items()}
        np.savez_compressed(folder / "waveforms.npz", **arrays)
    return metrics
