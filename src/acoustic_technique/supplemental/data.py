"""Frozen Webster data and independently checked pre-shock Burgers data."""

from .common import ROOT, CONFIG, dump, sha256
import numpy as np
from scipy.optimize import brentq
from scipy.special import jv


def implicit(sigma, theta):
    theta = np.asarray(theta, dtype=float)
    if sigma == 0:
        return np.sin(theta)
    return np.array(
        [
            brentq(
                lambda p: p - np.sin(t + sigma * p),
                -1.0,
                1.0,
                xtol=1e-13,
                rtol=4 * np.finfo(float).eps,
            )
            for t in theta.ravel()
        ]
    ).reshape(theta.shape)


def series(sigma, theta, n_terms):
    theta = np.asarray(theta, dtype=float)
    if sigma == 0:
        return np.sin(theta)
    n = np.arange(1, n_terms + 1, dtype=float)
    a = 2 * jv(n, n * sigma) / (n * sigma)
    return np.einsum("n,n...->...", a, np.sin(n.reshape((-1,) + (1,) * theta.ndim) * theta))


def reference_derivatives(X, P):
    c = np.cos(X[:, 1] + X[:, 0] * P)
    pt = c / (1 - X[:, 0] * c)
    return P * pt, pt


def grid(sigma, theta):
    s, t = np.meshgrid(sigma, theta, indexing="ij")
    p = np.array([implicit(float(z), theta) for z in sigma])
    return np.column_stack([s.ravel(), t.ravel()]), p.ravel()


def prepare_nonlinear(phases=64, sigma_max=0.8, train_end=0.32):
    rng = np.random.default_rng(CONFIG["data_seed"])
    edges = np.linspace(0, train_end, 65)
    centers = (edges[1:] + edges[:-1]) / 2
    rng.shuffle(centers)
    s_train = np.sort(centers[:40])
    s_valid = np.sort(centers[40:])
    theta = 2 * np.pi * np.arange(phases) / phases
    theta_v = 2 * np.pi * (np.arange(phases) + 0.5) / phases
    s_test = np.linspace(train_end + 1e-8, sigma_max, 96)
    x, y = grid(s_test, 2 * np.pi * np.arange(256) / 256)
    xt, yt = grid(s_train, theta)
    xv, yv = grid(s_valid, theta_v)
    # This extra ID test set is disjoint from both sets of training/validation positions.
    s_id = (np.arange(12) + 0.287) / 12 * train_end
    xi, yi = grid(s_id, 2 * np.pi * np.arange(256) / 256)
    return dict(
        Xtrain=xt,
        ytrain_clean=yt,
        Xval=xv,
        yval_clean=yv,
        Xtest=x,
        ytest=y,
        Xid=xi,
        yid=yi,
        sigma_train=s_train,
        sigma_valid=s_valid,
        sigma_test=s_test,
        theta_test=2 * np.pi * np.arange(256) / 256,
    )


def check_reference():
    folder = ROOT / "reference"
    folder.mkdir(parents=True, exist_ok=True)
    theta = 2 * np.pi * np.arange(1024) / 1024
    records = []
    chosen = 400
    for s in [0.0, 1e-7, 0.16, 0.32, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p = implicit(s, theta)
        error = {str(n): float(np.max(abs(series(s, theta, n) - p))) for n in [100, 200, 400]}
        records.append(
            dict(
                sigma=s,
                max_abs_pressure=float(np.max(abs(p))),
                implicit_residual=float(np.max(abs(p - np.sin(theta + s * p)))),
                series_errors=error,
            )
        )
    assert max(r["max_abs_pressure"] for r in records) <= 1 + 1e-12
    assert max(r["implicit_residual"] for r in records) < 1e-10
    assert max(r["series_errors"][str(chosen)] for r in records) < 1e-8
    d = prepare_nonlinear()
    assert d["Xtrain"].shape == (2560, 2) and d["Xval"].shape == (1536, 2)
    assert len(d["Xtest"]) == 24576
    assert set(d["sigma_train"]).isdisjoint(d["sigma_valid"])
    assert d["Xtest"][:, 0].min() > d["Xtrain"][:, 0].max()
    p = d["ytest"]
    ps, pt = reference_derivatives(d["Xtest"], p)
    report = dict(
        status="passed",
        generation="brentq implicit relation; no PySR target expressions used",
        series_terms_for_crosscheck=chosen,
        checks=records,
        reference_burgers_residual_rms=float(np.sqrt(np.mean((ps - p * pt) ** 2))),
        train_shape=list(d["Xtrain"].shape),
        valid_shape=list(d["Xval"].shape),
        test_shape=list(d["Xtest"].shape),
        data_seed=CONFIG["data_seed"],
        adjustment="Use 400 Fubini terms for verification; retain sigma_max=0.8 if checks pass.",
    )
    np.savez_compressed(folder / "nonlinear_clean.npz", **d)
    dump(folder / "nonlinear_verification.json", report)
    return report


def clean_data(job):
    if job["scenario"] == "nonlinear_burgers":
        if job.get("train_phases", 64) != 64 or job.get("sigma_max", 0.8) != 0.8:
            return prepare_nonlinear(job.get("train_phases", 64), job.get("sigma_max", 0.8))
        with np.load(ROOT / "reference/nonlinear_clean.npz") as z:
            return {k: z[k] for k in z.files}
    from ..acoustic_core import parameters, generate

    p = parameters(job["kind"], job["parameter_group"])
    d = generate(p)
    return dict(
        Xtrain=d["Xtrain"],
        ytrain_clean=d["ytrain"],
        Xval=d["Xval"],
        yval_clean=d["yval"],
        Xtest=d["X"],
        ytest=d["y"],
        x=d["x"],
        parameters=p,
    )


def build_observations(job):
    d = clean_data(job)
    snr = job["snr_db"]
    std = 0.0
    if snr != "clean":
        std = float(np.sqrt(np.mean(d["ytrain_clean"] ** 2)) * 10 ** (-float(snr) / 20))
    train_rng = np.random.default_rng(job["noise_seed"])
    valid_rng = np.random.default_rng(job["noise_seed"] + 100000)
    d["ytrain"] = d["ytrain_clean"] + train_rng.normal(0, std, len(d["ytrain_clean"]))
    d["yval"] = d["yval_clean"] + valid_rng.normal(0, std, len(d["yval_clean"]))
    d["noise_std"] = std
    return d


def save_data(path, d):
    from pathlib import Path
    import os

    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp%d.npz" % os.getpid())
    np.savez_compressed(temporary, **{k: v for k, v in d.items() if k != "parameters"})
    temporary.replace(path)


if __name__ == "__main__":
    print(check_reference())
