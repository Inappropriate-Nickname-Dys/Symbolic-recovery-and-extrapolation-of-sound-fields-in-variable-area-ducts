"""Conical two-variable discovery; IVP labels, separate phase/geometry/joint OOD."""

import os

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import time
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from .acoustic_core import ROOT, T, dump, refine, evaluate_expr, metrics

ETA = sp.Symbol("eta", real=True)


def field(t, eta):
    sol = solve_ivp(
        lambda u, y: [y[1], -2 * y[1] / (eta + u) - y[0]],
        (0, 10 * np.pi),
        [1.0, 0.0],
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
    )
    values = sol.sol(t)[0]
    exact = (eta * np.cos(t) + np.sin(t)) / (eta + t)
    if np.max(abs(values - exact)) >= 1e-8:
        raise AssertionError("joint ODE discrepancy")
    return values


def mesh_data(t, etas):
    X = np.array([(u, e) for e in etas for u in t], dtype=float)
    y = np.concatenate([field(t, e) for e in etas])
    return X, y


def data():
    Xt, yt = mesh_data(np.linspace(0, 4 * np.pi, 160), [15, 18, 21, 24, 27, 30])
    Xv, yv = mesh_data((np.arange(129) + 0.371) / 129 * 4 * np.pi, [16.5, 19.5, 22.5, 25.5, 28.5])
    regions = {}
    regions["phase_ood"] = mesh_data(
        np.linspace(4 * np.pi, 10 * np.pi, 401)[1:], [15, 18, 21, 24, 27, 30]
    )
    extra = [8, 10, 12, 14, 31, 34, 37, 40]
    regions["geometry_ood"] = mesh_data(np.linspace(0, 4 * np.pi, 241), extra)
    regions["joint_ood"] = mesh_data(np.linspace(4 * np.pi, 10 * np.pi, 401)[1:], extra)
    return Xt, yt, Xv, yv, regions


def poly_design(X, degree):
    # Same inputs, tensor-product polynomial restricted by total degree.
    u = (X[:, 0] - 2 * np.pi) / (2 * np.pi)
    v = (X[:, 1] - 22.5) / 7.5
    tu = np.polynomial.chebyshev.chebvander(u, degree)
    tv = np.polynomial.chebyshev.chebvander(v, degree)
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    return np.column_stack([tu[:, i] * tv[:, j] for i, j in terms]), terms


def run(a):
    os.environ["PYTHON_JULIACALL_THREADS"] = str(a.threads)
    folder = ROOT / "experiment_04_joint" / ("seed%d_%s" % (a.seed, a.stage))
    folder.mkdir(parents=True, exist_ok=True)
    result_name = "baseline_result.json" if a.stage == "baseline" else "result.json"
    if (folder / result_name).exists():
        print("EXISTS", folder)
        return
    Xt, yt, Xv, yv, regions = data()
    np.savez_compressed(
        folder / "data.npz",
        Xtrain=Xt,
        ytrain=yt,
        Xval=Xv,
        yval=yv,
        **{k + "_" + n: v for k, (X, y) in regions.items() for n, v in [("X", X), ("y", y)]},
    )
    pred = {}
    result = {}
    if a.stage == "baseline":
        from .run_baselines import fit_mlp

        start = time.monotonic()
        f, config = fit_mlp(Xt, yt, Xv, yv, a.seed, folder)
        result["MLP"] = dict(
            configuration=config,
            fit_seconds=time.monotonic() - start,
            metrics={k: metrics(y, f(X)) for k, (X, y) in regions.items()},
        )
        for k, (X, y) in regions.items():
            pred["MLP_" + k] = f(X)
        start = time.monotonic()
        candidates = []
        for degree in [6, 10, 14, 18, 22]:
            A, terms = poly_design(Xt, degree)
            coef = np.linalg.lstsq(A, yt, rcond=1e-12)[0]
            Av, _ = poly_design(Xv, degree)
            mse = float(np.mean((Av @ coef - yv) ** 2))
            candidates.append((mse, degree, coef, terms))
        mse, degree, coef, terms = min(candidates, key=lambda c: c[0])
        dump(
            folder / "joint_polynomial.json",
            dict(
                degree=degree,
                coefficients=coef.tolist(),
                terms=terms,
                validation_mse=mse,
                candidates=[dict(degree=c[1], mse=c[0]) for c in candidates],
            ),
        )
        result["Chebyshev"] = dict(
            configuration=dict(degree=degree, validation_mse=mse),
            fit_seconds=time.monotonic() - start,
            metrics={},
        )
        for k, (X, y) in regions.items():
            A, _ = poly_design(X, degree)
            yp = A @ coef
            pred["Chebyshev_" + k] = yp
            result["Chebyshev"]["metrics"][k] = metrics(y, yp)

        # Known target basis as an explicitly labelled strong physical upper bound.
        def basis(X):
            t, e = X.T
            return np.column_stack([e * np.cos(t) / (e + t), np.sin(t) / (e + t)])

        coef = np.linalg.lstsq(basis(Xt), yt, rcond=None)[0]
        result["Modal informed"] = dict(
            configuration=dict(
                coefficients=coef.tolist(), prior="conical envelope and phase basis"
            ),
            metrics={},
        )
        for k, (X, y) in regions.items():
            yp = basis(X) @ coef
            pred["Modal informed_" + k] = yp
            result["Modal informed"]["metrics"][k] = metrics(y, yp)
        dump(
            folder / result_name,
            dict(seed=a.seed, stage=a.stage, methods=result, status="completed"),
        )
    else:
        from pysr import PySRRegressor

        funcs = ["sin", "cos", "exp", "cosh"]
        config = dict(
            niterations=10000,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=funcs,
            maxsize=40,
            maxdepth=14,
            populations=40,
            population_size=80,
            ncycles_per_iteration=600,
            precision=64,
            parsimony=1e-6,
            optimizer_iterations=50,
            optimizer_nrestarts=4,
            optimize_probability=0.25,
            warmup_maxsize_by=0.02,
            nested_constraints={f: {g: 0 for g in funcs} for f in funcs},
            constraints={f: 7 for f in funcs},
            timeout_in_seconds=a.seconds,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-20 && complexity <= 40",
            parallelism="multithreading",
            random_state=a.seed,
            progress=False,
            verbosity=0,
            model_selection="accuracy",
            output_directory=str(folder / "pysr"),
            run_id="search",
        )
        if a.stage.startswith("focused"):
            # Declared trigonometric/rational prior ablation, not the main search library.
            config["unary_operators"] = ["sin", "cos"]
            config["binary_operators"] = ["+", "*", "/"]
            config["nested_constraints"] = {
                f: {g: 0 for g in ["sin", "cos"]} for f in ["sin", "cos"]
            }
            config["constraints"] = {"sin": 3, "cos": 3}
            config["optimizer_iterations"] = 100
        dump(folder / "configuration.json", config)
        start = time.monotonic()
        model = PySRRegressor(**config)
        dump(
            folder / "runtime.json",
            dict(julia_threads=a.threads, python=os.sys.executable, stage=a.stage),
        )
        model.fit(Xt, yt, variable_names=["t", "eta"])
        elapsed = time.monotonic() - start
        candidates = []
        for i, row in model.equations_.iterrows():
            expr = sp.sympify(row["sympy_format"]).subs({sp.Symbol("t"): T, sp.Symbol("eta"): ETA})
            re = refine(expr, Xt, yt)
            yp = evaluate_expr(re, Xv)
            mse = float(np.mean((yp - yv) ** 2)) if np.all(np.isfinite(yp)) else 1e100
            candidates.append(
                dict(
                    index=int(i),
                    complexity=int(row["complexity"]),
                    raw=str(expr),
                    refined=str(re),
                    validation_mse=mse,
                )
            )
        best = min(c["validation_mse"] for c in candidates)
        selected = min(
            (c for c in candidates if c["validation_mse"] <= max(1e-16, 1.1 * best)),
            key=lambda c: (c["complexity"], c["validation_mse"]),
        )
        dump(folder / "candidates.json", candidates)
        dump(folder / "selected_before_ood.json", selected)
        expr = sp.sympify(selected["refined"], locals={"t": T, "eta": ETA})
        result = {k: metrics(y, evaluate_expr(expr, X)) for k, (X, y) in regions.items()}
        identity = sp.simplify(sp.trigsimp(expr - (ETA * sp.cos(T) + sp.sin(T)) / (ETA + T))) == 0
        dump(
            folder / result_name,
            dict(
                seed=a.seed,
                stage=a.stage,
                budget_seconds=a.seconds,
                search_seconds=elapsed,
                selected=selected,
                metrics=result,
                exact_identity_proved=bool(identity),
                status="completed",
            ),
        )
        for k, (X, y) in regions.items():
            pred["PySR_" + k] = evaluate_expr(expr, X)
    np.savez_compressed(folder / "predictions.npz", **pred)
    print(folder.name, result, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument(
        "--threads", type=int, default=int(os.environ.get("PYTHON_JULIACALL_THREADS", "4"))
    )
    run(ap.parse_args())
