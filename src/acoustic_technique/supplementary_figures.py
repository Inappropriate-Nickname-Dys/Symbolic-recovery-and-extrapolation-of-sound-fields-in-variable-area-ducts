"""Additional scientific figures and a uniform numerical endpoint audit."""

import os
from .paths import ROOT, REPOSITORY_ROOT, SOURCE_ROOT
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs" / "matplotlib_cache"))
import numpy as np
import sympy as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .acoustic_core import ROOT, T, KINDS, area, location_metrics, dump, evaluate_expr
from .summarize import read, csvwrite, savefig, COLORS


def endpoint_audit():
    records = []
    tol = 1e-6

    def add(p, seed, method, x, y, yp, folder):
        met = location_metrics(x, y, yp, p["kappa"], endpoint_tolerance=tol)
        dump(
            folder / ("locations_boundary_tolerance_" + method.replace(" ", "_") + ".json"),
            dict(
                endpoint_tolerance=tol,
                method=method,
                metrics=met,
                reason="exclude numerical classification differences within 1e-6 of xi=.4 or xi=1; same rule for all methods; original locations retained",
            ),
        )
        for quantity, m in met.items():
            if quantity == "finite":
                continue
            records.append(
                dict(
                    kind=p["kind"],
                    index=p["index"],
                    seed=seed,
                    method=method,
                    quantity=quantity,
                    missed=m["missed"],
                    false=m["false"],
                    matched=m["matched"],
                    mean_xi_error=m["mean_xi_error"],
                    max_xi_error=m["max_xi_error"],
                    mean_wavelength_error=m["mean_wavelength_error"],
                    mean_amplitude_error=m.get("mean_amplitude_error"),
                )
            )

    for f in (ROOT / "experiment_01_recovery").glob("*/result.json"):
        r = read(f)
        if r["stage"] != "main":
            continue
        d = np.load(f.parent / "predictions.npz")
        add(r["parameters"], r["seed"], "PySR", d["x"], d["truth"], d["pysr"], f.parent)
    for f in (ROOT / "experiment_01_recovery" / "baselines").glob("*/result.json"):
        r = read(f)
        d = np.load(f.parent / "predictions.npz")
        for method in ["MLP", "Chebyshev", "Modal informed"]:
            add(r["parameters"], r["seed"], method, d["x"], d["truth"], d[method], f.parent)
        dp = np.load(f.parent / "mlp_polished_predictions.npz")
        add(
            r["parameters"],
            r["seed"],
            "MLP polished",
            dp["x"],
            dp["truth"],
            dp["MLP_polished"],
            f.parent,
        )
    csvwrite(ROOT / "experiment_05_locations" / "locations_boundary_tolerance.csv", records)
    dump(
        ROOT / "experiment_05_locations" / "boundary_rule.json",
        dict(
            endpoint_tolerance=tol,
            region="0.4+1e-6 < xi < 1-1e-6",
            historical_motivation="uniform kappa=10pi ODE spline peak at .400000000004154 crosses OOD boundary; analytic cos peak at .4 does not",
            original_results_retained=True,
            models_unchanged=True,
        ),
    )
    return records


def main():
    rows = endpoint_audit()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for kind in KINDS:
        p = read(ROOT / "experiment_00_generator" / (kind + "_1.json"))
        d = np.load(ROOT / "experiment_00_generator" / (kind + "_1.npz"))
        axes[0].semilogy(d["x"], area(d["x"], p), label=kind)
        axes[1].plot(d["x"], d["y"], label=kind)
    axes[0].set_ylabel("Normalized area S/S0")
    axes[1].set_ylabel("ODE normalized pressure")
    for ax in axes:
        ax.set_xlabel("Axial coordinate xi")
        ax.axvspan(0, 0.4, color="gray", alpha=0.12)
        ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, ROOT / "experiment_00_generator" / "figures" / "geometry_and_ode_truth")
    # Show why small value error does not prove correct derivative physics.
    r = read(ROOT / "experiment_04_joint" / "seed2_main" / "result.json")
    eta = sp.Symbol("eta", real=True)
    e = sp.sympify(r["selected"]["refined"], locals={"t": T, "eta": eta})
    truth = (eta * sp.cos(T) + sp.sin(T)) / (eta + T)
    t = np.logspace(-18, -7, 601)
    X = np.column_stack([t, np.full_like(t, 21.0)])
    learned = evaluate_expr(sp.diff(e, T), X)
    exact = evaluate_expr(sp.diff(truth, T), X)
    residue = sp.diff(e, T, 2) + 2 * sp.diff(e, T) / (eta + T) + e
    rr = evaluate_expr(residue, X)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].semilogx(t, learned, label="Main seed 2")
    axes[0].semilogx(t, exact, "k--", label="Exact target")
    axes[0].set_ylabel("Phase derivative dP/dt")
    axes[0].legend()
    axes[0].set_title("Actual near-origin derivative")
    axes[1].loglog(t, np.maximum(abs(rr), 1e-20))
    axes[1].set_ylabel("Absolute Webster residual")
    axes[1].set_title("Actual near-origin equation residual")
    for ax in axes:
        ax.set_xlabel("Phase t (eta = 21)")
    fig.tight_layout()
    savefig(fig, ROOT / "experiment_04_joint" / "figures" / "near_origin_failure_diagnosis")
    joint = []
    for f in (ROOT / "experiment_04_joint").glob("*/independent_verification.json"):
        v = read(f)
        r = read(f.parent / "result.json")
        if v["stage"] == "pilot":
            continue
        joint.append(
            (
                v["stage"],
                v["seed"],
                r["metrics"]["joint_ood"]["nrmse"],
                v["certificate"]["max_initial_phase_derivative_error"],
            )
        )
    fig, ax = plt.subplots(figsize=(6, 4))
    for stage, color in [
        ("main", "#c43c39"),
        ("focused", "#2678ad"),
        ("focused_extended", "#278743"),
    ]:
        for s, seed, error, derivative in joint:
            if s != stage:
                continue
            ax.scatter(
                max(error, 1e-14),
                max(derivative, 1e-16),
                color=color,
                label=(
                    stage
                    if not any(l == stage for l in ax.get_legend_handles_labels()[1])
                    else None
                ),
            )
            if error > 1e-8 or derivative > 1e-6:
                ax.annotate(
                    str(seed),
                    (max(error, 1e-14), max(derivative, 1e-16)),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Joint OOD pressure NRMSE")
    ax.set_ylabel("Maximum initial derivative error")
    ax.legend()
    fig.tight_layout()
    savefig(fig, ROOT / "experiment_04_joint" / "figures" / "value_error_and_initial_condition")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    methods = ["PySR", "MLP polished", "Chebyshev", "Modal informed"]
    pos = np.arange(len(KINDS))
    width = 0.2
    for j, method in enumerate(methods):
        vals = []
        miss = []
        for kind in KINDS:
            group = [
                r
                for r in rows
                if r["kind"] == kind and r["method"] == method and r["quantity"] == "nodes"
            ]
            errors = [
                r["mean_wavelength_error"] for r in group if r["mean_wavelength_error"] is not None
            ]
            vals.append(max(float(np.median(errors)), 1e-14) if errors else np.nan)
            miss.append(sum(r["missed"] for r in group))
        axes[0].bar(pos + (j - 1.5) * width, vals, width, color=COLORS[method], label=method)
        axes[1].bar(pos + (j - 1.5) * width, miss, width, color=COLORS[method], label=method)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median matched node error / wavelength")
    axes[1].set_ylabel("Missed OOD nodes, all 15 runs")
    for ax in axes:
        ax.set_xticks(pos)
        ax.set_xticklabels(KINDS, rotation=20, fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, ROOT / "experiment_05_locations" / "figures" / "node_localization_and_misses")
    print("supplementary figures and endpoint audit completed")


if __name__ == "__main__":
    main()
