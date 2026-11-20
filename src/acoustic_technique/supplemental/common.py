"""Portable settings and train-only symbolic utilities for supplemental studies."""

import os
from pathlib import Path
import sys
import json
import hashlib
import time
import threading

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

from ..paths import REPOSITORY_ROOT, ROOT as STUDY_ROOT

SETTINGS = json.loads(
    Path(os.environ.get("ACOUSTIC_CONFIG", REPOSITORY_ROOT / "config.json")).read_text()
)
CONFIG = dict(
    data_seed=SETTINGS.get("supplemental", {}).get("data_seed", 2026),
    julia_threads=SETTINGS["execution"]["julia_threads"],
    snapshot_interval_seconds=SETTINGS.get("supplemental", {}).get("snapshot_interval_seconds", 20),
)
CODE = Path(__file__).resolve().parent
ROOT = STUDY_ROOT / "supplemental_study"
os.environ.setdefault("PYTHON_JULIACALL_THREADS", str(CONFIG["julia_threads"]))
os.environ.setdefault("JULIA_NUM_THREADS", str(CONFIG["julia_threads"]))
os.environ.setdefault("JULIA_NUM_GC_THREADS", "1")
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
os.environ.setdefault("PYTHON_JULIAPKG_EXE", SETTINGS["environments"]["julia_executable"])
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", SETTINGS["environments"]["julia_project"])

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

T = sp.Symbol("t", real=True)
SIGMA = sp.Symbol("sigma", real=True)
THETA = sp.Symbol("theta", real=True)


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d.%d" % (os.getpid(), threading.get_ident()))
    temporary.write_text(
        json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_expression(s):
    return sp.sympify(str(s).replace("^", "**"), locals={"t": T, "sigma": SIGMA, "theta": THETA})


def evaluate(expr, X):
    variables = [T] if X.shape[1] == 1 else [SIGMA, THETA]
    with np.errstate(all="ignore"):
        y = np.asarray(sp.lambdify(variables, expr, "numpy")(*X.T))
        if np.iscomplexobj(y):
            if np.any(abs(y.imag) > 1e-12):
                return np.full(len(X), np.nan)
            y = y.real
        return np.broadcast_to(np.asarray(y, dtype=float), (len(X),)).copy()


def mse(y, p):
    return float(np.mean((p - y) ** 2)) if np.all(np.isfinite(p)) else float("inf")


def prediction_metrics(y, p):
    if not np.all(np.isfinite(p)):
        return dict(finite=False, nrmse=None, mae=None, max_abs=None)
    return dict(
        finite=True,
        nrmse=float(np.sqrt(mse(y, p)) / max(np.sqrt(np.mean(y * y)), 1e-15)),
        mae=float(np.mean(abs(p - y))),
        max_abs=float(np.max(abs(p - y))),
    )


def refine(expr, X, y):
    """Fit floating constants on training observations, exactly as in the study."""
    constants = sorted(expr.atoms(sp.Float), key=str)
    if not constants or len(constants) > 18:
        return expr
    coeff = sp.symbols("c:" + str(len(constants)))
    template = expr.xreplace(dict(zip(constants, coeff)))
    variables = [T] if X.shape[1] == 1 else [SIGMA, THETA]
    f = sp.lambdify([*variables, *coeff], template, "numpy")
    initial = np.array([float(c) for c in constants])

    def residual(c):
        with np.errstate(all="ignore"):
            z = np.broadcast_to(np.asarray(f(*X.T, *c), dtype=float), y.shape)
        return np.nan_to_num(z - y, nan=1e10, posinf=1e10, neginf=-1e10)

    try:
        fitted = least_squares(
            residual,
            initial,
            jac="3-point",
            max_nfev=1200,
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            x_scale="jac",
        )
        if np.mean(residual(fitted.x) ** 2) <= np.mean(residual(initial) ** 2):
            return template.subs(dict(zip(coeff, [sp.Float(v, 17) for v in fitted.x])))
    except (ValueError, OverflowError, TypeError, FloatingPointError):
        pass
    return expr


def select(candidates):
    finite = [
        c
        for c in candidates
        if c.get("validation_mse") is not None and np.isfinite(c["validation_mse"])
    ]
    if not finite:
        raise RuntimeError("No finite validation candidate")
    threshold = max(1e-16, 1.1 * min(c["validation_mse"] for c in finite))
    return min(
        (c for c in finite if c["validation_mse"] <= threshold),
        key=lambda c: (c["complexity"], c["validation_mse"]),
    )


def run_folder(job):
    return ROOT / "supplemental" / job["phase"] / job["run_id"]


def elapsed():
    return time.perf_counter()
