"""Scientific figures from saved results; no fitting or model selection."""

from pathlib import Path
import json, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT

OUT = ROOT / "summary"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    rows = summary["rows"]
    winner = summary["winner"]
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "figure.dpi": 150})
    phases = sorted(
        {r["batch"] for r in rows if r["scenario"] == "nonlinear_burgers" and r["snr"] == "clean"}
    )
    if not phases:
        return
    phase = winner["confirmation_phase"] if winner else phases[-1]
    new = [r for r in rows if r["method"] == "PySR" and r["batch"] == phase]
    old = [r for r in rows if r["method"] == "MLP" and r["batch"] == phase and r["snr"] == "clean"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, key, title, target in zip(
        axes,
        ["ood_nrmse", "harmonic_1_coefficient_error", "harmonic_2_coefficient_error"],
        ["OOD pressure", "Fundamental", "Second harmonic"],
        [0.05, 0.05, 0.05],
    ):
        for i, (rs, label) in enumerate([(old, "Paired MLP"), (new, "Enhanced confirmation")]):
            v = [r[key] for r in rs if r[key] is not None]
            ax.scatter(np.full(len(v), i) + np.linspace(-0.06, 0.06, len(v)), v, s=28)
            if v:
                ax.hlines(np.median(v), i - 0.15, i + 0.15, color="black")
        ax.axhline(target, ls=":", color="gray")
        ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Paired MLP", "Enhanced PySR"])
        ax.set_title(title)
        ax.set_ylabel("Relative error")
    fig.tight_layout()
    fig.savefig(OUT / "01_clean_comparison.png", dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for snr, rs in [("clean", new)] + [
        (
            snr,
            [
                r
                for r in rows
                if r["method"] == "PySR"
                and r["batch"] == "noise_performance"
                and str(r["snr"]) == str(snr)
            ],
        )
        for snr in [50, 40, 30]
    ]:
        if not rs:
            continue
        values = np.array(
            [[r["harmonic_%d_coefficient_error" % n] for n in range(1, 11)] for r in rs],
            dtype=float,
        )
        median = np.nanmedian(values, axis=0)
        q1 = np.nanquantile(values, 0.25, axis=0)
        q3 = np.nanquantile(values, 0.75, axis=0)
        axes[0].plot(range(1, 11), median, "o-", label=str(snr))
        axes[0].fill_between(range(1, 11), q1, q3, alpha=0.12)
        i = ["clean", "50", "40", "30"].index(str(snr))
        v = [r["ood_nrmse"] for r in rs if r["ood_nrmse"] is not None]
        axes[1].scatter(np.full(len(v), i) + np.linspace(-0.06, 0.06, len(v)), v)
        axes[1].hlines(np.median(v), i - 0.15, i + 0.15, color="black")
    axes[0].axhline(0.05, ls=":", color="gray")
    axes[0].axhline(0.1, ls="--", color="gray")
    axes[0].set_xlabel("Harmonic order")
    axes[0].set_ylabel("Phase-sensitive relative coefficient error")
    axes[0].set_yscale("log")
    axes[0].legend(title="SNR (dB)")
    axes[1].axhline(0.05, ls=":", color="gray")
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(["Clean", "50", "40", "30"])
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("OOD pressure NRMSE")
    fig.tight_layout()
    fig.savefig(OUT / "02_noise_and_harmonics.png", dpi=240)
    plt.close(fig)
    rs = [r for r in new if r["ood_nrmse"] is not None]
    if rs:
        representative = sorted(rs, key=lambda r: r["ood_nrmse"])[len(rs) // 2]
        with np.load(ROOT / representative["path"] / "waveforms.npz") as z:
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
            for ax, s in zip(axes, ["0.4", "0.6", "0.8"]):
                ax.plot(z[s + "_theta"], z[s + "_truth"], "k", label="Reference")
                ax.plot(z[s + "_theta"], z[s + "_prediction"], "--", label="Enhanced PySR")
                ax.set_title("sigma=" + s)
                ax.set_xlabel("Phase theta")
                ax.set_ylabel("Normalized pressure")
            axes[0].legend()
            fig.tight_layout()
            fig.savefig(OUT / "03_clean_representative_waveforms.png", dpi=240)
            plt.close(fig)
        (OUT / "figure_provenance.json").write_text(
            json.dumps(
                {
                    "waveform_representative": representative,
                    "selection": "median OOD error of independent clean confirmation; post-hoc illustration only",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    costs = [r for r in rows if r["batch"] == "efficiency"]
    if costs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        for i, label in enumerate(["A", "B", "C", "D"]):
            rs = [r for r in costs if r["run_id"].startswith(label + "_")]
            axes[0].bar(i, sum(r["target_reached"] for r in rs))
            axes[0].text(
                i,
                sum(r["target_reached"] for r in rs) + 0.1,
                "%d/%d" % (sum(r["target_reached"] for r in rs), len(rs)),
                ha="center",
            )
            axes[1].scatter(
                np.full(len(rs), i),
                [r["search_seconds"] for r in rs],
                color="tab:blue",
                label="Search" if i == 0 else None,
            )
            axes[1].scatter(
                np.full(len(rs), i) + 0.12,
                [r["total_seconds"] for r in rs],
                color="tab:orange",
                marker="x",
                label="End-to-end" if i == 0 else None,
            )
        for ax in axes:
            ax.set_xticks(range(4))
            ax.set_xticklabels(
                ["A: simple clean", "B: simple 40", "C: nonlinear clean", "D: nonlinear 40"],
                rotation=15,
            )
        axes[0].set_ylim(0, 6)
        axes[0].set_ylabel("Runs meeting scenario-specific target")
        axes[1].set_ylabel("Wall time (s)")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(OUT / "04_independent_cost.png", dpi=240)
        plt.close(fig)


if __name__ == "__main__":
    main()
