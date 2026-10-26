"""Diagnose rank-deficient total-degree fits; anisotropic tensor Chebyshev."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import time
import copy
import numpy as np
import torch
from .acoustic_core import ROOT, dump, metrics
from .physical_diagnostics import load_net, pressure


def design(X, dt, de):
    u = (X[:, 0] - 2 * np.pi) / (2 * np.pi)
    v = (X[:, 1] - 22.5) / 7.5
    tu = np.polynomial.chebyshev.chebvander(u, dt)
    tv = np.polynomial.chebyshev.chebvander(v, de)
    return np.einsum("ni,nj->nij", tu, tv).reshape(len(X), -1)


def run(seed):
    source = ROOT / "experiment_04_joint" / ("seed%d_baseline" % seed)
    folder = ROOT / "experiment_04_joint" / ("seed%d_baseline_refined" % seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "baseline_result.json").exists():
        print("EXISTS", seed)
        return
    initial = json.loads((source / "baseline_result.json").read_text())
    d = np.load(source / "data.npz")
    Xt, yt, Xv, yv = d["Xtrain"], d["ytrain"], d["Xval"], d["yval"]
    candidates = []
    start = time.monotonic()
    # eta degree <=5 matches six independent training curves; t can be high degree.
    for dt in [6, 10, 14, 18, 22, 26]:
        for de in [1, 2, 3, 4, 5]:
            A = design(Xt, dt, de)
            coef, resid, rank, s = np.linalg.lstsq(A, yt, rcond=1e-12)
            val = float(np.mean((design(Xv, dt, de) @ coef - yv) ** 2))
            candidates.append(
                dict(
                    t_degree=dt,
                    eta_degree=de,
                    coef=coef,
                    validation_mse=val,
                    rank=int(rank),
                    columns=A.shape[1],
                    condition=float(s[0] / s[-1]),
                )
            )
    selected = min(candidates, key=lambda c: c["validation_mse"])
    dump(
        folder / "tensor_chebyshev_candidates.json",
        [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
            for c in candidates
        ],
    )
    dump(
        folder / "selected_before_ood.json",
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in selected.items()},
    )
    result = {
        "Chebyshev tensor": dict(
            configuration={k: v for k, v in selected.items() if k != "coef"},
            fit_seconds=time.monotonic() - start,
            metrics={},
        )
    }
    pred = {}
    net, checkpoint = load_net(source / "mlp.pt")
    initial_state = copy.deepcopy(net.state_dict())
    train = torch.from_numpy((Xt - checkpoint["mean"]) / checkpoint["scale"])
    target = torch.from_numpy(yt[:, None])
    valx = torch.from_numpy((Xv - checkpoint["mean"]) / checkpoint["scale"])
    valy = torch.from_numpy(yv[:, None])
    with torch.no_grad():
        initial_val = float(torch.mean((net(valx) - valy) ** 2))
    opt = torch.optim.LBFGS(
        net.parameters(),
        lr=1,
        max_iter=3000,
        tolerance_grad=1e-13,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )
    start = time.monotonic()

    def closure():
        opt.zero_grad()
        loss = torch.mean((net(train) - target) ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        new_val = float(torch.mean((net(valx) - valy) ** 2))
    if new_val >= initial_val:
        net.load_state_dict(initial_state)
    checkpoint["state_dict"] = net.state_dict()
    torch.save(checkpoint, folder / "mlp_polished.pt")
    result["MLP polished"] = dict(
        configuration=dict(
            initial_validation_mse=initial_val,
            validation_mse=min(initial_val, new_val),
            accepted=new_val < initial_val,
            additional_lbfgs_max_iter=3000,
        ),
        fit_seconds=time.monotonic() - start,
        metrics={},
    )
    result["Modal informed"] = initial["methods"]["Modal informed"]
    # All selection is complete before evaluation below.
    for region in ["phase_ood", "geometry_ood", "joint_ood"]:
        X, y = d[region + "_X"], d[region + "_y"]
        pc = design(X, selected["t_degree"], selected["eta_degree"]) @ selected["coef"]
        pm = pressure(net, checkpoint, X)
        result["Chebyshev tensor"]["metrics"][region] = metrics(y, pc)
        result["MLP polished"]["metrics"][region] = metrics(y, pm)
        pred["Chebyshev tensor_" + region] = pc
        pred["MLP polished_" + region] = pm
        if region in ["geometry_ood", "joint_ood"]:
            for suffix, mask in [("low_eta", X[:, 1] <= 14), ("high_eta", X[:, 1] >= 31)]:
                result["Chebyshev tensor"]["metrics"][region + "_" + suffix] = metrics(
                    y[mask], pc[mask]
                )
                result["MLP polished"]["metrics"][region + "_" + suffix] = metrics(
                    y[mask], pm[mask]
                )
    np.savez_compressed(folder / "predictions.npz", **pred)
    dump(
        folder / "baseline_result.json",
        dict(
            seed=seed,
            stage="baseline_refined",
            methods=result,
            status="completed",
            reason="six discrete eta values limit identifiable eta degree; train/ID validation determine degrees",
        ),
    )
    print(seed, selected["t_degree"], selected["eta_degree"], min(initial_val, new_val), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.seed)
