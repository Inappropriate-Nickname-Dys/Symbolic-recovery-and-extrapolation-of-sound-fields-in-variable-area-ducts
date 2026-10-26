"""A single PySR run with ID-only candidate selection and constant refinement."""

import os

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import time
import traceback
import json
from pathlib import Path
import numpy as np
import sympy as sp
from .acoustic_core import ROOT, T, parameters, generate, dump, refine, evaluate_expr, field_metrics


def run(a):
    job = "%s_%d_N%d_%s_seed%d_%s" % (a.kind, a.index, a.n, a.sampling, a.seed, a.stage)
    folder = ROOT / a.experiment / job
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists() and not a.force:
        print("EXISTS", job, flush=True)
        return
    p = parameters(a.kind, a.index)
    data = generate(p, a.n, a.sampling, a.seed)
    np.savez_compressed(folder / "data.npz", **data)
    # Sine/cosine arguments may have affine frequency/phase; envelopes affine too.
    # This constraint contains all four exact targets, with room to spare.
    nested = {
        f: {g: 0 for g in ["sin", "cos", "exp", "cosh"]} for f in ["sin", "cos", "exp", "cosh"]
    }
    config = dict(
        niterations=10000,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "cosh"],
        maxsize=40,
        maxdepth=14,
        populations=32,
        population_size=64,
        ncycles_per_iteration=600,
        precision=64,
        parsimony=1e-6,
        warmup_maxsize_by=0.05,
        optimizer_iterations=50,
        optimizer_nrestarts=4,
        optimize_probability=0.25,
        nested_constraints=nested,
        constraints={"sin": 7, "cos": 7, "exp": 7, "cosh": 7},
        early_stop_condition="stop_if(loss, complexity) = loss < 1e-20 && complexity <= 40",
        timeout_in_seconds=a.seconds,
        parallelism="multithreading",
        random_state=a.seed,
        progress=False,
        verbosity=0,
        model_selection="accuracy",
        output_directory=str(folder / "pysr"),
        run_id="search",
    )
    if a.stage == "focused":
        # Uniform narrow-library ablation for all geometries: remove redundant subtraction.
        config["binary_operators"] = ["+", "*", "/"]
        config["populations"] = 48
        config["optimizer_iterations"] = 100
        config["ncycles_per_iteration"] = 800
    dump(
        folder / "configuration.json",
        dict(
            parameters=p,
            pysr=config,
            selection="training constants; validation MSE; simplest within max(1e-16,1.1*best)",
            seed_determinism="multithreading is not bitwise deterministic",
            input="t=kappa*xi",
            validation_access="xi in [0,0.4] only",
            ood_access="after selection freeze",
        ),
    )
    from pysr import PySRRegressor

    start = time.monotonic()
    model = PySRRegressor(**config)
    model.fit(data["Xtrain"], data["ytrain"], variable_names=["t"])
    search_seconds = time.monotonic() - start
    candidates = []
    for i, row in model.equations_.iterrows():
        expr = sp.sympify(row["sympy_format"]).subs({sp.Symbol("t"): T})
        refined = refine(expr, data["Xtrain"], data["ytrain"])
        yv = evaluate_expr(refined, data["Xval"])
        mse = float(np.mean((yv - data["yval"]) ** 2)) if np.all(np.isfinite(yv)) else 1e100
        candidates.append(
            dict(
                index=int(i),
                complexity=int(row["complexity"]),
                raw=str(expr),
                refined=str(refined),
                training_mse=float(
                    np.mean((evaluate_expr(refined, data["Xtrain"]) - data["ytrain"]) ** 2)
                ),
                validation_mse=mse,
            )
        )
    dump(folder / "candidates.json", candidates)
    best = min(c["validation_mse"] for c in candidates)
    selected = min(
        (c for c in candidates if c["validation_mse"] <= max(1e-16, 1.1 * best)),
        key=lambda c: (c["complexity"], c["validation_mse"]),
    )
    dump(folder / "selected_before_ood.json", selected)
    expr = sp.sympify(selected["refined"], locals={"t": T})
    yp = evaluate_expr(expr, data["X"])
    met = field_metrics(data, yp, p, expr)
    np.savez_compressed(folder / "predictions.npz", x=data["x"], truth=data["y"], pysr=yp)
    out = dict(
        job=job,
        parameters=p,
        n=a.n,
        sampling=a.sampling,
        seed=a.seed,
        stage=a.stage,
        budget_seconds=a.seconds,
        search_wall_seconds=search_seconds,
        selected=selected,
        metrics=met,
        status="completed",
    )
    dump(folder / "result.json", out)
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--sampling", default="uniform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--experiment", default="experiment_01_recovery")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    try:
        run(a)
    except Exception:
        traceback.print_exc()
        raise
