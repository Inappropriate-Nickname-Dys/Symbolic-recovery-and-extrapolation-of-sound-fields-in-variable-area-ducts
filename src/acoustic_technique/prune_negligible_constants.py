"""Public post-diagnostic cleanup; ID-only acceptance, no oracle parameter snapping."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import numpy as np
import sympy as sp
from .acoustic_core import ROOT, T, dump, refine, evaluate_expr, field_metrics, metrics
from .structure_certificate import certify
from .final_audit import joint_family

THRESHOLD = 1e-10


def main():
    summaries = []
    eta = sp.Symbol("eta", real=True)
    for experiment in ["experiment_01_recovery", "experiment_03_sampling", "experiment_04_joint"]:
        for f in (ROOT / experiment).glob("*/result.json"):
            r = json.loads(f.read_text())
            expr = sp.sympify(r["selected"]["refined"], locals={"t": T, "eta": eta})
            tiny = {c: sp.Integer(0) for c in expr.atoms(sp.Float) if 0 < abs(float(c)) < THRESHOLD}
            if not tiny:
                continue
            is_joint = experiment == "experiment_04_joint"
            d = np.load(f.parent / "data.npz")
            cleaned = sp.simplify(expr.xreplace(tiny))
            cleaned = refine(cleaned, d["Xtrain"], d["ytrain"])
            val = evaluate_expr(cleaned, d["Xval"])
            loss = float(np.mean((val - d["yval"]) ** 2)) if np.all(np.isfinite(val)) else 1e100
            accepted = loss <= max(1e-16, 1.1 * r["selected"]["validation_mse"]) and sp.count_ops(
                cleaned
            ) < sp.count_ops(expr)
            snap = dict(
                original=str(expr),
                cleaned=str(cleaned),
                threshold=THRESHOLD,
                removed_constants=[str(c) for c in tiny],
                validation_mse=loss,
                accepted=bool(accepted),
                selection="same ID validation tolerance; lower symbolic operator count",
                status="post-diagnostic development refinement; original results retained",
            )
            dump(f.parent / "pruned_selected_before_ood.json", snap)
            if accepted:
                if is_joint:
                    out = {
                        region: metrics(d[region + "_y"], evaluate_expr(cleaned, d[region + "_X"]))
                        for region in ["phase_ood", "geometry_ood", "joint_ood"]
                    }
                    certificate = joint_family(cleaned, eta)
                    certificate["exact_identity_proved"] = bool(
                        sp.simplify(
                            sp.trigsimp(cleaned - (eta * sp.cos(T) + sp.sin(T)) / (eta + T))
                        )
                        == 0
                    )
                else:
                    data = {k: d[k] for k in d.files}
                    yp = evaluate_expr(cleaned, d["X"])
                    out = field_metrics(data, yp, r["parameters"], cleaned)
                    certificate = certify(str(cleaned), r["parameters"], 1e-6)
                dump(
                    f.parent / "pruned_result.json",
                    dict(**snap, metrics=out, certificate=certificate),
                )
                summaries.append(
                    dict(
                        experiment=experiment,
                        job=r.get("job", f.parent.name),
                        **snap,
                        certificate=certificate,
                    )
                )
            else:
                summaries.append(
                    dict(experiment=experiment, job=r.get("job", f.parent.name), **snap)
                )
    dump(ROOT / "docs" / "negligible_constant_cleanup.json", summaries)
    print(
        "negligible-constant candidates",
        len(summaries),
        "accepted",
        sum(s["accepted"] for s in summaries),
        flush=True,
    )


if __name__ == "__main__":
    main()
