"""Independent saved-model checks and training-only MLP polishing."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import time
import copy
import json
import numpy as np
import sympy as sp
import torch
from numpy.polynomial import Chebyshev
from .acoustic_core import (
    ROOT,
    T,
    parameters,
    generate,
    dump,
    field_metrics,
    log_slope,
    evaluate_expr,
)


def load_net(path):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = torch.nn.Sequential(
        torch.nn.Linear(checkpoint["input_dim"], 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    )
    net.load_state_dict(checkpoint["state_dict"])
    return net, checkpoint


def pressure(net, checkpoint, X):
    with torch.no_grad():
        return net(torch.from_numpy((X - checkpoint["mean"]) / checkpoint["scale"])).numpy().ravel()


def mlp_derivatives(net, checkpoint, X, kappa):
    inputs = torch.from_numpy(X.copy()).requires_grad_(True)
    outputs = net(
        (inputs - torch.from_numpy(checkpoint["mean"])) / torch.from_numpy(checkpoint["scale"])
    )
    first = torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), inputs)[0]
    return (
        outputs.detach().numpy().ravel(),
        first.detach().numpy().ravel() * kappa,
        second.detach().numpy().ravel() * kappa * kappa,
    )


def diagnostic(x, y, yp, p1, p2, p):
    residual = p2 + log_slope(x, p) * p1 + p["kappa"] ** 2 * yp
    return dict(
        webster_relative_residual_rms=float(
            np.sqrt(np.mean(residual**2)) / (p["kappa"] ** 2 * np.sqrt(np.mean(y * y)))
        ),
        initial_pressure_error=float(abs(yp[0] - 1)),
        initial_derivative_error_over_kappa=float(abs(p1[0]) / p["kappa"]),
    )


def run(a):
    label = "%s_%d_N%d_%s_seed%d" % (a.kind, a.index, a.n, a.sampling, a.seed)
    folder = ROOT / a.experiment / "baselines" / label
    if (folder / "physical_diagnostics.json").exists():
        print("EXISTS", label)
        return
    r = json.loads((folder / "result.json").read_text())
    p = r["parameters"]
    d = generate(p, a.n, a.sampling, a.seed)
    original = np.load(folder / "predictions.npz")
    diag = {}
    # Earlier baseline runs preceded explicit data saving; clearly label reconstruction.
    if not (folder / "data.npz").exists():
        np.savez_compressed(folder / "data_reconstructed_from_configuration.npz", **d)
    net, checkpoint = load_net(folder / "mlp.pt")
    yp, p1, p2 = mlp_derivatives(net, checkpoint, d["X"], p["kappa"])
    diag["MLP"] = diagnostic(d["x"], d["y"], yp, p1, p2, p)
    discrepancy = float(np.max(abs(yp - original["MLP"])))
    if discrepancy > 1e-12:
        raise AssertionError("saved MLP prediction mismatch " + str(discrepancy))
    diag["MLP"]["saved_prediction_max_difference"] = discrepancy
    # Continuous polynomial derivatives, not finite differences.
    poly = json.loads((folder / "chebyshev.json").read_text())
    cheb = Chebyshev(poly["coef"], domain=poly["domain"])
    yp = cheb(d["X"][:, 0])
    p1 = cheb.deriv(1)(d["X"][:, 0]) * p["kappa"]
    p2 = cheb.deriv(2)(d["X"][:, 0]) * p["kappa"] ** 2
    diag["Chebyshev"] = diagnostic(d["x"], d["y"], yp, p1, p2, p)
    A, B = r["methods"]["Modal informed"]["configuration"]["coefficients"]
    x = d["x"]
    mu, b, k = p["mu"], p["b"], p["kappa"]
    q = np.sqrt(k * k - mu * mu)
    if p["kind"] == "uniform":
        env = np.ones_like(x)
        e1 = np.zeros_like(x)
        e2 = e1
        w = k
    elif p["kind"] == "conical":
        env = b / (b + x)
        e1 = -env / (b + x)
        e2 = 2 * env / (b + x) ** 2
        w = k
    elif p["kind"] == "exponential":
        env = np.exp(-mu * x)
        e1 = -mu * env
        e2 = mu * mu * env
        w = q
    else:
        env = 1 / np.cosh(mu * x)
        z = np.tanh(mu * x)
        e1 = -mu * z * env
        e2 = mu * mu * (2 * z * z - 1) * env
        w = q
    f = A * np.cos(w * x) + B * np.sin(w * x)
    f1 = w * (-A * np.sin(w * x) + B * np.cos(w * x))
    f2 = -w * w * f
    diag["Modal informed"] = diagnostic(
        x, d["y"], env * f, e1 * f + env * f1, e2 * f + 2 * e1 * f1 + env * f2, p
    )
    # Same polishing rule for every seed. No OOD data in optimization/acceptance.
    train = torch.from_numpy((d["Xtrain"] - checkpoint["mean"]) / checkpoint["scale"])
    yt = torch.from_numpy(d["ytrain"][:, None])
    val = torch.from_numpy((d["Xval"] - checkpoint["mean"]) / checkpoint["scale"])
    yv = torch.from_numpy(d["yval"][:, None])
    initial_state = copy.deepcopy(net.state_dict())
    with torch.no_grad():
        initial_val = float(torch.mean((net(val) - yv) ** 2))
    opt = torch.optim.LBFGS(
        net.parameters(),
        lr=1,
        max_iter=3000,
        tolerance_grad=1e-13,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )
    trace = []
    steps = 0
    start = time.monotonic()

    def closure():
        nonlocal steps
        opt.zero_grad()
        loss = torch.mean((net(train) - yt) ** 2)
        loss.backward()
        if steps % 100 == 0:
            with torch.no_grad():
                v = float(torch.mean((net(val) - yv) ** 2))
            trace.append([steps, float(loss), v])
        steps += 1
        return loss

    opt.step(closure)
    with torch.no_grad():
        new_val = float(torch.mean((net(val) - yv) ** 2))
    accepted = new_val < initial_val
    if not accepted:
        net.load_state_dict(initial_state)
    checkpoint["state_dict"] = net.state_dict()
    torch.save(checkpoint, folder / "mlp_polished.pt")
    dump(
        folder / "mlp_polishing.json",
        dict(
            max_iter=3000,
            closure_calls=steps,
            initial_validation_mse=initial_val,
            polished_validation_mse=new_val,
            accepted=accepted,
            seconds=time.monotonic() - start,
            selection="same ID validation, OOD not accessed",
        ),
    )
    np.savetxt(
        folder / "mlp_polishing_trace.csv",
        trace,
        delimiter=",",
        header="closure,training_mse,validation_mse",
        comments="",
    )
    yp, p1, p2 = mlp_derivatives(net, checkpoint, d["X"], k)
    diag["MLP polished"] = diagnostic(x, d["y"], yp, p1, p2, p)
    met = field_metrics(d, yp, p)
    dump(
        folder / "mlp_polished_result.json",
        dict(
            parameters=p,
            n=a.n,
            sampling=a.sampling,
            seed=a.seed,
            metrics=met,
            validation_mse=min(initial_val, new_val),
        ),
    )
    np.savez_compressed(folder / "mlp_polished_predictions.npz", x=x, truth=d["y"], MLP_polished=yp)
    dump(
        folder / "physical_diagnostics.json",
        dict(
            parameters=p,
            n=a.n,
            sampling=a.sampling,
            seed=a.seed,
            methods=diag,
            derivatives="MLP autodiff; Chebyshev analytic; modal analytic",
            labels="same deterministic ODE data",
            status="completed",
        ),
    )
    print(
        label,
        "val",
        initial_val,
        "->",
        min(initial_val, new_val),
        "OOD",
        met["ood"]["nrmse"],
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--sampling", default="uniform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experiment", default="experiment_01_recovery")
    run(ap.parse_args())
