"""Offline checkpoint analysis with ID-only expression choice and censored times."""

from .common import *
from .data import build_observations
import csv


def analyze(folder):
    folder = Path(folder)
    if (folder / "cost_summary.json").exists():
        return
    job = json.loads((folder / "job.json").read_text())
    d = build_observations(job)
    nonlinear = job["scenario"] == "nonlinear_burgers"
    if nonlinear:
        from scipy.special import jv

        st = d["sigma_test"]
        nt = len(d["theta_test"])
        theta = d["theta_test"]
        n = np.arange(1, 3)
        reference = 2 * jv(n, st[:, None] * n) / (st[:, None] * n)
        basis_s = np.sin(theta[:, None] * n)
        basis_c = np.cos(theta[:, None] * n)
    states = {}
    with (folder / "search_trace.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            states.setdefault(float(row["elapsed_s"]), []).append(row)
    cache = {}
    frozen = []
    for t, rows in sorted(states.items()):
        candidates = []
        for row in rows:
            raw = row["equation"]
            if raw not in cache:
                try:
                    ex = parse_expression(raw)
                    cache[raw] = dict(
                        raw=str(ex), validation_mse=mse(d["yval"], evaluate(ex, d["Xval"]))
                    )
                except Exception:
                    cache[raw] = dict(raw=raw, validation_mse=float("inf"))
            candidates.append(dict(cache[raw], complexity=int(row["complexity"])))
        try:
            chosen = select(candidates)
        except RuntimeError:
            continue
        frozen.append(dict(elapsed_s=t, chosen=chosen))
    dump(
        folder / "cost_selection_before_test.json",
        dict(
            rule="ID validation only; raw HOF checkpoints, no post-fit refinement",
            checkpoints=frozen,
        ),
    )
    first_basic = None
    first_one_percent = None
    previous = 0.0
    history = []
    tests = {}
    for state in frozen:
        raw = state["chosen"]["raw"]
        if raw not in tests:
            try:
                ex = parse_expression(raw)
                p = evaluate(ex, d["Xtest"])
                if nonlinear:
                    o = prediction_metrics(d["ytest"], p)["nrmse"]
                    idn = prediction_metrics(d["yid"], evaluate(ex, d["Xid"]))["nrmse"]
                    P = p.reshape(len(st), nt)
                    si = 2 / nt * P @ basis_s
                    co = 2 / nt * P @ basis_c
                    err = np.sqrt(
                        np.sum((si - reference) ** 2 + co**2, axis=0) / np.sum(reference**2, axis=0)
                    )
                    basic = bool(
                        idn is not None
                        and idn <= 0.01
                        and o is not None
                        and o <= 0.05
                        and np.all(err <= 0.05)
                    )
                    tests[raw] = (o, idn, float(err[0]), float(err[1]), basic)
                else:
                    mask = d["x"] > 0.4
                    o = prediction_metrics(d["ytest"][mask], p[mask])["nrmse"]
                    tests[raw] = (o, None, None, None, o is not None and o <= 0.01)
            except Exception:
                tests[raw] = (None, None, None, None, False)
        o, idn, h1, h2, basic = tests[raw]
        entry = dict(
            first_observed_seconds=state["elapsed_s"], previous_checkpoint_seconds=previous
        )
        if basic and first_basic is None:
            first_basic = entry
        if o is not None and o <= 0.01 and first_one_percent is None:
            first_one_percent = entry
        history.append(
            dict(
                elapsed_s=state["elapsed_s"],
                ood_nrmse=o,
                id_nrmse=idn,
                h1_error=h1,
                h2_error=h2,
                basic_reached=basic,
            )
        )
        previous = state["elapsed_s"]
    dump(
        folder / "cost_summary.json",
        dict(
            first_basic=first_basic,
            first_ood_one_percent=first_one_percent,
            checkpoint_count=len(frozen),
            not_reached_is_censored=True,
            first_basic_definition="nonlinear: ID<=1%, OOD<=5%, H1 and H2 each<=5%; simple: OOD<=1%",
            time_origin="native logger creation after separate Julia warmup; roughly 20s checkpoint interval plus population latency",
            raw_candidates_only=True,
            history=history,
        ),
    )


if __name__ == "__main__":
    import argparse, concurrent.futures

    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="efficiency")
    a = p.parse_args()
    folders = [r.parent for r in (ROOT / "supplemental" / a.phase).glob("*/result.json")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(analyze, folders))
