"""Same-noise MLP, Chebyshev, and explicitly informed linear-basis controls."""

import time

PROCESS_START = time.perf_counter()
from .common import *
from .data import build_observations, save_data
from .evaluation import evaluate_model, save_evaluation
import argparse, copy, traceback
import torch
from numpy.polynomial import Chebyshev


def make_net(dim):
    return torch.nn.Sequential(
        torch.nn.Linear(dim, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    )


def fit_mlp(d, seed, folder):
    from ..run_baselines import fit_mlp as original_fit

    folder.mkdir(parents=True, exist_ok=True)
    start = elapsed()
    _, info = original_fit(d["Xtrain"], d["ytrain"], d["Xval"], d["yval"], seed, folder)
    checkpoint = torch.load(folder / "mlp.pt", map_location="cpu", weights_only=False)
    net = make_net(d["Xtrain"].shape[1])
    net.load_state_dict(checkpoint["state_dict"])
    mean, scale = checkpoint["mean"], checkpoint["scale"]
    xt = torch.from_numpy((d["Xtrain"] - mean) / scale)
    yt = torch.from_numpy(d["ytrain"][:, None])
    xv = torch.from_numpy((d["Xval"] - mean) / scale)
    yv = torch.from_numpy(d["yval"][:, None])
    initial_state = copy.deepcopy(net.state_dict())
    with torch.no_grad():
        initial_val = float(torch.mean((net(xv) - yv) ** 2))
    optimizer = torch.optim.LBFGS(
        net.parameters(),
        lr=1,
        max_iter=3000,
        tolerance_grad=1e-13,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = torch.mean((net(xt) - yt) ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        new_val = float(torch.mean((net(xv) - yv) ** 2))
    if new_val >= initial_val:
        net.load_state_dict(initial_state)
    checkpoint["state_dict"] = net.state_dict()
    torch.save(checkpoint, folder / "mlp_polished.pt")
    info.update(
        initial_validation_mse=initial_val,
        validation_mse=min(initial_val, new_val),
        additional_lbfgs_max_iter=3000,
        additional_accepted=new_val < initial_val,
        labels="same noisy training/validation samples as paired PySR run",
        precision="float64",
    )

    def predict(X):
        parts = []
        with torch.no_grad():
            for xx in np.array_split(X, max(1, int(np.ceil(len(X) / 4096)))):
                parts.append(net(torch.from_numpy((xx - mean) / scale)).numpy().ravel())
        return np.concatenate(parts)

    def derivatives(X, nonlinear):
        first = []
        second = []
        for xx in np.array_split(X, max(1, int(np.ceil(len(X) / 2048)))):
            x = torch.tensor(xx, dtype=torch.float64, requires_grad=True)
            y = net((x - torch.from_numpy(mean)) / torch.from_numpy(scale))
            g = torch.autograd.grad(y.sum(), x, create_graph=not nonlinear)[0]
            if nonlinear:
                first.append(g[:, 0].detach().numpy())
                second.append(g[:, 1].detach().numpy())
            else:
                g2 = torch.autograd.grad(g[:, 0].sum(), x)[0][:, 0]
                first.append(g[:, 0].detach().numpy())
                second.append(g2.detach().numpy())
        return np.concatenate(first), np.concatenate(second)

    return predict, derivatives, info, elapsed() - start


def run(job):
    folder = run_folder(job) / "baselines"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        print("EXISTS BASELINE", job["run_id"])
        return
    d = build_observations(job)
    save_data(folder / "data.npz", d)
    paired = run_folder(job) / "data.npz"
    if paired.exists():
        with np.load(paired) as a:
            for k in ["Xtrain", "Xval", "ytrain", "yval", "ytest"]:
                assert np.array_equal(a[k], d[k]), k
    nonlinear = job["scenario"] == "nonlinear_burgers"
    methods = {}
    predictions = {}
    predict, derivatives, info, seconds = fit_mlp(d, job["pysr_seed"], folder / "MLP")
    p = predict(d["Xtest"])
    predictions["MLP"] = p
    m = save_evaluation(
        folder / "MLP",
        evaluate_model(job, d, p, predict=predict, derivatives=derivatives(d["Xtest"], nonlinear)),
    )
    methods["MLP"] = dict(metrics=m, configuration=info, fit_seconds=seconds)
    if not nonlinear:
        start = elapsed()
        candidates = []
        for degree in [6, 10, 14, 18, 22, 26]:
            model = Chebyshev.fit(
                d["Xtrain"][:, 0], d["ytrain"], degree, domain=[0, 0.4 * d["parameters"]["kappa"]]
            )
            candidates.append((mse(d["yval"], model(d["Xval"][:, 0])), degree, model))
        val, degree, model = min(candidates, key=lambda c: (c[0], c[1]))
        configuration = dict(
            degree=degree,
            coef=model.coef,
            domain=model.domain,
            validation_mse=val,
            validation_candidates=[dict(degree=c[1], mse=c[0]) for c in candidates],
        )
        dump(folder / "chebyshev.json", configuration)
        p = model(d["Xtest"][:, 0])
        predictions["Chebyshev"] = p
        derivative = (model.deriv()(d["Xtest"][:, 0]), model.deriv(2)(d["Xtest"][:, 0]))
        methods["Chebyshev"] = dict(
            configuration=configuration,
            fit_seconds=elapsed() - start,
            metrics=evaluate_model(job, d, p, derivatives=derivative),
        )
        start = elapsed()
        par = d["parameters"]
        k, mu, b = par["kappa"], par["mu"], par["b"]
        w = np.sqrt(k * k - mu * mu) / k
        if par["kind"] == "uniform":
            env = sp.Integer(1)
            w = 1.0
        elif par["kind"] == "conical":
            env = b * k / (b * k + T)
            w = 1.0
        elif par["kind"] == "exponential":
            env = sp.exp(-mu / k * T)
        else:
            env = 1 / sp.cosh(mu / k * T)
        f1, f2 = env * sp.cos(w * T), env * sp.sin(w * T)
        A = np.column_stack([evaluate(f1, d["Xtrain"]), evaluate(f2, d["Xtrain"])])
        coef = np.linalg.lstsq(A, d["ytrain"], rcond=None)[0]
        expr = coef[0] * f1 + coef[1] * f2
        dump(
            folder / "informed_basis.json",
            dict(
                expression=str(expr),
                coefficients=coef,
                prior="known exact geometry, envelope and effective wavenumber; two linear coefficients fitted",
            ),
        )
        p = evaluate(expr, d["Xtest"])
        predictions["Informed basis"] = p
        derivative = (
            evaluate(sp.diff(expr, T), d["Xtest"]),
            evaluate(sp.diff(expr, T, 2), d["Xtest"]),
        )
        methods["Informed basis"] = dict(
            fit_seconds=elapsed() - start,
            configuration=dict(coefficients=coef, prior="correct envelope and wavenumber"),
            metrics=evaluate_model(job, d, p, derivatives=derivative),
        )
    np.savez_compressed(folder / "predictions.npz", X=d["Xtest"], truth=d["ytest"], **predictions)
    dump(
        folder / "result.json",
        dict(
            status="completed",
            job=job,
            methods=methods,
            comparison="Identical observations and ID validation; method training costs not matched to PySR search budget",
            end_to_end_seconds=elapsed() - PROCESS_START,
        ),
    )
    print(
        "BASELINES",
        job["run_id"],
        {k: v["metrics"]["ood"]["nrmse"] for k, v in methods.items()},
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job_json")
    a = ap.parse_args()
    job = json.loads(Path(a.job_json).read_text())
    try:
        run(job)
    except Exception:
        dump(
            run_folder(job) / "baselines/error.json",
            dict(traceback=traceback.format_exc(), job=job),
        )
        raise
