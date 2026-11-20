"""One isolated PySR run; all selection is frozen before any OOD evaluation."""

import time

PROCESS_START = time.perf_counter()
from .common import *
import argparse, traceback
from .data import build_observations, save_data
from .evaluation import evaluate_model, save_evaluation


def configuration(job, folder):
    unary = ["sin", "cos", "exp", "cosh"]
    c = dict(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=unary,
        niterations=100000000,
        ncycles_per_iteration=150,
        populations=64,
        population_size=128,
        maxsize=80,
        maxdepth=30,
        parsimony=1e-8,
        warmup_maxsize_by=0,
        optimizer_iterations=100,
        optimizer_nrestarts=8,
        optimize_probability=0.3,
        nested_constraints={},
        constraints={},
        precision=64,
        batching=False,
        batch_size=512,
        early_stop_condition=None,
        timeout_in_seconds=job["budget_seconds"],
        parallelism="multithreading",
        random_state=job["pysr_seed"],
        progress=False,
        verbosity=0,
        model_selection="accuracy",
        print_precision=17,
        output_directory=str(folder / "pysr"),
        run_id="search",
        delete_tempfiles=False,
    )
    if job["phase"] == "simple_noise":
        # Reproduce the earlier LINEAR noise experiment; nonlinear defaults above
        # are always enhanced. Independent efficiency A/B receive enhanced overrides.
        c.update(
            niterations=10000,
            ncycles_per_iteration=600,
            populations=32,
            population_size=64,
            maxsize=40,
            maxdepth=14,
            parsimony=1e-6,
            warmup_maxsize_by=0.05,
            optimizer_iterations=50,
            optimizer_nrestarts=4,
            optimize_probability=0.25,
            nested_constraints={f: {g: 0 for g in unary} for f in unary},
            constraints={f: 7 for f in unary},
        )
    c.update(job.get("overrides", {}))
    return c


def run(job):
    folder = run_folder(job)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        print("EXISTS", job["run_id"], flush=True)
        return
    dump(folder / "job.json", job)
    d = build_observations(job)
    save_data(folder / "data.npz", d)
    from pysr import PySRRegressor
    from pysr.julia_import import jl
    from .logger_spec import SnapshotLoggerSpec

    setup = time.perf_counter() - PROCESS_START
    runtime = dict(
        julia=str(jl.VERSION),
        symbolic_regression=str(jl.seval("Base.pkgversion(SymbolicRegression)")),
        julia_threads=int(jl.seval("Threads.nthreads()")),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
    )
    assert runtime["julia_threads"] == CONFIG["julia_threads"]
    c = configuration(job, folder)
    dump(
        folder / "configuration.json",
        dict(
            job=job,
            pysr=c,
            runtime=runtime,
            selection="train-only constant refit; ID validation simplest within max(1e-16,1.1 E_min)",
            validation_access="observed domain only",
            ood_access="only after selected_before_ood.json freeze",
            logging="native Julia AbstractSRLogger; HOF snapshots every approximately %g s; full float strings"
            % CONFIG["snapshot_interval_seconds"],
        ),
    )
    # Explicit, separately timed warmup makes search-call timings comparable.
    warm = dict(c)
    warm.update(
        niterations=2,
        populations=1,
        population_size=16,
        ncycles_per_iteration=5,
        timeout_in_seconds=None,
        output_directory=str(folder / "warmup"),
        run_id="compile",
        maxsize=12,
        warmup_maxsize_by=0.0,
        logger_spec=SnapshotLoggerSpec(folder / "warmup_trace.tsv", 0),
    )
    start = elapsed()
    wm = PySRRegressor(**warm)
    names = ["t"] if d["Xtrain"].shape[1] == 1 else ["sigma", "theta"]
    wm.fit(
        d["Xtrain"][: min(64, len(d["Xtrain"]))],
        d["ytrain"][: min(64, len(d["ytrain"]))],
        variable_names=names,
    )
    warm_seconds = elapsed() - start
    c["logger_spec"] = SnapshotLoggerSpec(
        folder / "search_trace.tsv", CONFIG["snapshot_interval_seconds"]
    )
    model = PySRRegressor(**c)
    dump(
        folder / "search_started.json",
        dict(unix_time=time.time(), process_import_seconds=setup, warmup_seconds=warm_seconds),
    )
    start = elapsed()
    model.fit(d["Xtrain"], d["ytrain"], variable_names=names)
    search_seconds = elapsed() - start
    refit_start = elapsed()
    candidates = []
    for i, row in model.equations_.iterrows():
        try:
            expr = parse_expression(row["sympy_format"])
            refined = refine(expr, d["Xtrain"], d["ytrain"])
            candidates.append(
                dict(
                    index=int(i),
                    complexity=int(row["complexity"]),
                    raw=str(expr),
                    refined=str(refined),
                    training_mse=mse(d["ytrain"], evaluate(refined, d["Xtrain"])),
                    validation_mse=mse(d["yval"], evaluate(refined, d["Xval"])),
                )
            )
        except Exception as e:
            candidates.append(
                dict(
                    index=int(i),
                    complexity=int(row["complexity"]),
                    raw=str(row["equation"]),
                    validation_mse=float("inf"),
                    error=str(e),
                )
            )
    selected = select(candidates)
    dump(folder / "candidates.json", candidates)
    dump(folder / "selected_before_ood.json", selected)
    refit_seconds = elapsed() - refit_start
    expr = parse_expression(selected["refined"])
    pred = evaluate(expr, d["Xtest"])
    # First use of OOD labels is below the selection freeze.
    met = save_evaluation(folder, evaluate_model(job, d, pred, expr=expr))
    np.savez_compressed(folder / "predictions.npz", prediction=pred, truth=d["ytest"], X=d["Xtest"])
    h = met.get("harmonics", {})
    out = dict(
        status="completed",
        job=job,
        selected=selected,
        metrics=met,
        noise_std=d["noise_std"],
        training_clean_rms=float(np.sqrt(np.mean(d["ytrain_clean"] ** 2))),
        actual_train_noise_rms=float(np.sqrt(np.mean((d["ytrain"] - d["ytrain_clean"]) ** 2))),
        actual_valid_noise_rms=float(np.sqrt(np.mean((d["yval"] - d["yval_clean"]) ** 2))),
        n_train=len(d["ytrain"]),
        n_valid=len(d["yval"]),
        n_test=len(d["ytest"]),
        runtime=runtime,
        process_import_seconds=setup,
        warmup_seconds=warm_seconds,
        search_call_wall_seconds=search_seconds,
        post_search_refit_seconds=refit_seconds,
        end_to_end_seconds=elapsed() - PROCESS_START,
        budget_seconds=job["budget_seconds"],
        target_ood_reached=bool(met["ood"]["nrmse"] is not None and met["ood"]["nrmse"] <= 0.01),
        target_ood_and_harmonic_reached=bool(
            met["ood"]["nrmse"] is not None
            and met["ood"]["nrmse"] <= 0.01
            and h.get("amplitude_error_1_5", 1.0) <= 0.01
        ),
    )
    dump(folder / "result.json", out)
    print(job["run_id"], met["ood"], selected["refined"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job_json")
    a = ap.parse_args()
    job = json.loads(Path(a.job_json).read_text())
    try:
        run(job)
    except Exception:
        folder = run_folder(job)
        folder.mkdir(parents=True, exist_ok=True)
        dump(folder / "error.json", dict(status="error", job=job, traceback=traceback.format_exc()))
        traceback.print_exc()
        raise
