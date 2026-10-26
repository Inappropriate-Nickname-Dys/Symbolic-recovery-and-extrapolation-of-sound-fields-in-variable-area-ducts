"""Same data/input and ID-only selection for all baselines."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import time
import copy
import numpy as np
import torch
from numpy.polynomial import Chebyshev
from .acoustic_core import ROOT, parameters, generate, dump, field_metrics


def fit_mlp(X, y, Xv, yv, seed, folder):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    xt = torch.from_numpy((X - mean) / scale)
    yt = torch.from_numpy(y[:, None])
    xv = torch.from_numpy((Xv - mean) / scale)
    yvt = torch.from_numpy(yv[:, None])
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=300, factor=0.5, min_lr=1e-5
    )
    best = float("inf")
    state = None
    wait = 0
    trace = []
    for epoch in range(12000):
        opt.zero_grad()
        loss = torch.mean((net(xt) - yt) ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = float(torch.mean((net(xv) - yvt) ** 2))
        scheduler.step(val)
        if val < best * (1 - 1e-7):
            best = val
            state = copy.deepcopy(net.state_dict())
            wait = 0
        else:
            wait += 1
        if epoch % 100 == 0:
            trace.append([epoch, float(loss), val, opt.param_groups[0]["lr"]])
        if best < 1e-12 or wait >= 1500:
            break
    net.load_state_dict(state)
    # Training-only polishing. Accept using the same ID validation criterion.
    lb = torch.optim.LBFGS(
        net.parameters(),
        lr=1,
        max_iter=500,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lb.zero_grad()
        l = torch.mean((net(xt) - yt) ** 2)
        l.backward()
        return l

    lb.step(closure)
    with torch.no_grad():
        val = float(torch.mean((net(xv) - yvt) ** 2))
    if val < best:
        best = val
        state = copy.deepcopy(net.state_dict())
    else:
        net.load_state_dict(state)
    torch.save(
        dict(state_dict=state, mean=mean, scale=scale, input_dim=X.shape[1]), folder / "mlp.pt"
    )
    np.savetxt(
        folder / "mlp_training_trace.csv",
        np.array(trace),
        delimiter=",",
        header="epoch,training_mse,validation_mse,lr",
        comments="",
    )

    def predict(z):
        with torch.no_grad():
            return net(torch.from_numpy((z - mean) / scale)).numpy().ravel()

    return predict, dict(
        validation_mse=best,
        epochs=epoch + 1,
        architecture="3 hidden layers x 64, tanh, float64",
        adam_lr=1e-3,
        lbfgs_max_iter=500,
    )


def run(a):
    label = "%s_%d_N%d_%s_seed%d" % (a.kind, a.index, a.n, a.sampling, a.seed)
    folder = ROOT / a.experiment / "baselines" / label
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        print("EXISTS", label, flush=True)
        return
    p = parameters(a.kind, a.index)
    d = generate(p, a.n, a.sampling, a.seed)
    np.savez_compressed(folder / "data.npz", **d)
    X, y, Xv, yv = d["Xtrain"], d["ytrain"], d["Xval"], d["yval"]
    records = {}
    preds = dict(x=d["x"], truth=d["y"])
    start = time.monotonic()
    predict, info = fit_mlp(X, y, Xv, yv, a.seed, folder)
    yp = predict(d["X"])
    records["MLP"] = dict(
        configuration=info, fit_seconds=time.monotonic() - start, metrics=field_metrics(d, yp, p)
    )
    preds["MLP"] = yp
    start = time.monotonic()
    candidates = []
    for degree in [6, 10, 14, 18, 22, 26]:
        model = Chebyshev.fit(X[:, 0], y, degree, domain=[0, 0.4 * p["kappa"]])
        candidates.append((float(np.mean((model(Xv[:, 0]) - yv) ** 2)), degree, model))
    val, degree, model = min(candidates, key=lambda c: (c[0], c[1]))
    yp = model(d["X"][:, 0])
    dump(
        folder / "chebyshev.json",
        dict(
            degree=degree,
            coef=model.coef.tolist(),
            domain=model.domain.tolist(),
            validation_mse=val,
            candidates=[dict(mse=c[0], degree=c[1]) for c in candidates],
        ),
    )
    records["Chebyshev"] = dict(
        configuration=dict(degree=degree, validation_mse=val),
        fit_seconds=time.monotonic() - start,
        metrics=field_metrics(d, yp, p),
    )
    preds["Chebyshev"] = yp
    # Geometry-informed baseline with exact known effective wavenumber and envelope.
    start = time.monotonic()

    def basis(x):
        k, mu, b = p["kappa"], p["mu"], p["b"]
        q = np.sqrt(k * k - mu * mu)
        if p["kind"] == "uniform":
            env = np.ones_like(x)
            w = k
        elif p["kind"] == "conical":
            env = b / (b + x)
            w = k
        elif p["kind"] == "exponential":
            env = np.exp(-mu * x)
            w = q
        else:
            env = 1 / np.cosh(mu * x)
            w = q
        return np.column_stack([env * np.cos(w * x), env * np.sin(w * x)])

    coef = np.linalg.lstsq(basis(d["xtrain"]), y, rcond=None)[0]
    yp = basis(d["x"]) @ coef
    records["Modal informed"] = dict(
        configuration=dict(
            coefficients=coef.tolist(), prior="known geometry, envelope and effective wavenumber"
        ),
        fit_seconds=time.monotonic() - start,
        metrics=field_metrics(d, yp, p),
    )
    preds["Modal informed"] = yp
    np.savez_compressed(folder / "predictions.npz", **preds)
    out = dict(
        parameters=p, n=a.n, sampling=a.sampling, seed=a.seed, methods=records, status="completed"
    )
    dump(folder / "result.json", out)
    print(label, {k: v["metrics"]["ood"]["nrmse"] for k, v in records.items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--sampling", default="uniform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experiment", default="experiment_01_recovery")
    run(ap.parse_args())
