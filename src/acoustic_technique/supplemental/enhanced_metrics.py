"""Additional harmonic checks, used only after ID-selected model freezing."""

from .common import *
from .data import grid, implicit
from .evaluation import nonlinear_metrics


def assess(folder):
    folder = Path(folder)
    result = json.loads((folder / "result.json").read_text())
    m = result["metrics"]
    h = np.load(folder / "harmonics.npz")
    ref = h["reference_amplitude"]
    si = h["sine"]
    co = h["cosine"]
    amp = h["amplitude"]
    den = np.sum(ref * ref, axis=0)
    coefficient = np.sqrt(np.sum((si - ref) ** 2 + co**2, axis=0) / den)
    amplitude = np.sqrt(np.sum((amp - ref) ** 2, axis=0) / den)
    # Independent, offset evaluation grid; never used for expression choice.
    selected = json.loads((folder / "selected_before_ood.json").read_text())
    expr = parse_expression(selected["refined"])
    sx = 0.32 + (np.arange(97) + 0.371) / 97 * 0.48
    tx = 2 * np.pi * (np.arange(257) + 0.219) / 257
    xx, yy = grid(sx, tx)
    independent = prediction_metrics(yy, evaluate(expr, xx))
    idn = m["id_test"]["nrmse"]
    ood = m["ood"]["nrmse"]
    finite = lambda x: x is not None and np.isfinite(x)
    basic = bool(
        finite(idn)
        and idn <= 0.01
        and finite(ood)
        and ood <= 0.05
        and np.all(coefficient[:2] <= 0.05)
    )
    out = dict(
        basic_reached=basic,
        id_nrmse=idn,
        ood_nrmse=ood,
        independent_offset_ood=independent,
        coefficient_error_by_order=coefficient,
        amplitude_error_by_order=amplitude,
        advanced_orders_at_10_percent=[
            int(i + 1) for i, v in enumerate(coefficient) if i >= 2 and v <= 0.10
        ],
        criteria="ID pressure <=1%; OOD pressure <=5%; each of fundamental and second harmonic coefficient errors <=5%",
        selection="ID-validation only, no harmonic decomposition, constraints, labels or losses used during search",
        exploratory_note="Adaptive configuration search is exploratory; all failed attempts retained; offset grid checks numerical reproducibility, not a new physical distribution",
    )
    dump(folder / "enhanced_metrics.json", out)
    return out


if __name__ == "__main__":
    import sys

    print(json.dumps(clean_json(assess(sys.argv[1])), ensure_ascii=False))
