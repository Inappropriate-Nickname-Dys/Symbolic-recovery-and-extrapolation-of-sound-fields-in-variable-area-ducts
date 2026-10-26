"""Article-level audit, tables and figures; run with the existing sr matplotlib env."""

import os
from .paths import ROOT, REPOSITORY_ROOT, SOURCE_ROOT
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs" / "matplotlib_cache"))
import json
import csv
import hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .acoustic_core import ROOT, KINDS, area, dump

plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300}
)
COLORS = {
    "PySR": "#b8332d",
    "MLP": "#2274a5",
    "MLP polished": "#0b526d",
    "Chebyshev": "#8c6bb1",
    "Modal informed": "#238b45",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csvwrite(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + ".png", bbox_inches="tight")
    fig.savefig(str(path) + ".svg", bbox_inches="tight")
    plt.close(fig)


def wilson(s, n):
    if not n:
        return (0.0, 1.0)
    z = 1.959963984540054
    center = (s / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt((s / n) * (1 - s / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def stats(values):
    v = np.array([float(x) for x in values if x is not None and np.isfinite(x)])
    if len(v) == 0:
        return dict(n=0)
    return dict(
        n=len(v),
        median=float(np.median(v)),
        q1=float(np.quantile(v, 0.25)),
        q3=float(np.quantile(v, 0.75)),
        mean=float(np.mean(v)),
        std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        min=float(np.min(v)),
        max=float(np.max(v)),
    )


def number(v):
    return "NA" if v is None else ("%.3g" % v)


def mdtable(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def collect(experiment, stage):
    searches = []
    baselines = []
    for f in (ROOT / experiment).glob("*/result.json"):
        r = read(f)
        if r.get("stage") == stage:
            r["_path"] = str(f.parent)
            searches.append(r)
    for f in (ROOT / experiment / "baselines").glob("*/result.json"):
        r = read(f)
        r["_path"] = str(f.parent)
        baselines.append(r)
    searches.sort(
        key=lambda r: (
            r["parameters"]["kind"],
            r["parameters"]["index"],
            r["n"],
            r["sampling"],
            r["seed"],
        )
    )
    return searches, baselines


def flatten(searches, baselines):
    rows = []
    for r in searches:
        p = r["parameters"]
        m = r["metrics"]
        certpath = Path(r["_path"]) / "structure_certificate.json"
        cert = read(certpath) if certpath.exists() else {}
        rows.append(
            dict(
                kind=p["kind"],
                index=p["index"],
                kappa=p["kappa"],
                mu=p["mu"],
                b=p["b"],
                n=r["n"],
                sampling=r["sampling"],
                seed=r["seed"],
                method="PySR",
                stage=r["stage"],
                validation_mse=r["selected"]["validation_mse"],
                ood_nrmse=m["ood"]["nrmse"],
                ood_mae=m["ood"]["mae"],
                ood_max_abs=m["ood"]["max_abs"],
                near_nrmse=m["(0.4, 0.6)"]["nrmse"],
                middle_nrmse=m["(0.6, 0.8)"]["nrmse"],
                far_nrmse=m["(0.8, 1.0)"]["nrmse"],
                usable_ood_xi=m["usable_ood_length_xi"],
                fit_seconds=r["search_wall_seconds"],
                complexity=r["selected"]["complexity"],
                structure_numeric=m["structure_numerically_consistent"],
                expression=r["selected"]["refined"],
            )
        )
    for r in baselines:
        p = r["parameters"]
        for method, result in r["methods"].items():
            m = result["metrics"]
            rows.append(
                dict(
                    kind=p["kind"],
                    index=p["index"],
                    kappa=p["kappa"],
                    mu=p["mu"],
                    b=p["b"],
                    n=r["n"],
                    sampling=r["sampling"],
                    seed=r["seed"],
                    method=method,
                    stage="baseline",
                    validation_mse=result["configuration"].get("validation_mse"),
                    ood_nrmse=m["ood"]["nrmse"],
                    ood_mae=m["ood"]["mae"],
                    ood_max_abs=m["ood"]["max_abs"],
                    near_nrmse=m["(0.4, 0.6)"]["nrmse"],
                    middle_nrmse=m["(0.6, 0.8)"]["nrmse"],
                    far_nrmse=m["(0.8, 1.0)"]["nrmse"],
                    usable_ood_xi=m["usable_ood_length_xi"],
                    fit_seconds=result["fit_seconds"],
                    complexity=None,
                    structure_numeric=None,
                    expression=None,
                )
            )
        polished = Path(r["_path"]) / "mlp_polished_result.json"
        if polished.exists():
            v = read(polished)
            m = v["metrics"]
            info = read(Path(r["_path"]) / "mlp_polishing.json")
            rows.append(
                dict(
                    kind=p["kind"],
                    index=p["index"],
                    kappa=p["kappa"],
                    mu=p["mu"],
                    b=p["b"],
                    n=r["n"],
                    sampling=r["sampling"],
                    seed=r["seed"],
                    method="MLP polished",
                    stage="baseline_polished",
                    validation_mse=v["validation_mse"],
                    ood_nrmse=m["ood"]["nrmse"],
                    ood_mae=m["ood"]["mae"],
                    ood_max_abs=m["ood"]["max_abs"],
                    near_nrmse=m["(0.4, 0.6)"]["nrmse"],
                    middle_nrmse=m["(0.6, 0.8)"]["nrmse"],
                    far_nrmse=m["(0.8, 1.0)"]["nrmse"],
                    usable_ood_xi=m["usable_ood_length_xi"],
                    fit_seconds=r["methods"]["MLP"]["fit_seconds"] + info["seconds"],
                    complexity=None,
                    structure_numeric=None,
                    expression=None,
                )
            )
    return rows


def audit(searches, baselines):
    failures = []
    checked = 0
    bykey = {
        (r["parameters"]["kind"], r["parameters"]["index"], r["n"], r["sampling"], r["seed"]): r
        for r in baselines
    }
    for r in searches:
        folder = Path(r["_path"])
        d = np.load(folder / "data.npz")
        key = (r["parameters"]["kind"], r["parameters"]["index"], r["n"], r["sampling"], r["seed"])
        if d["xtrain"].max() > 0.4 + 1e-12 or d["xval"].max() > 0.4 + 1e-12:
            failures.append([str(folder), "split"])
        if min(abs(d["xtrain"][:, None] - d["xval"][None, :]).ravel()) < 1e-12:
            failures.append([str(folder), "overlap"])
        if (folder / "selected_before_ood.json").stat().st_mtime > (
            folder / "result.json"
        ).stat().st_mtime:
            failures.append([str(folder), "selection chronology"])
        if key not in bykey:
            failures.append([str(folder), "missing baseline"])
        else:
            bfolder = Path(bykey[key]["_path"])
            snapshot = bfolder / "data.npz"
            if not snapshot.exists():
                snapshot = bfolder / "data_reconstructed_from_configuration.npz"
            if snapshot.exists():
                bd = np.load(snapshot)
                for column in ["Xtrain", "ytrain", "Xval", "yval"]:
                    if not np.array_equal(d[column], bd[column]):
                        failures.append([str(folder), "baseline array differs: " + column])
        checked += 1
    report = dict(
        checked=checked,
        failures=failures,
        passed=not failures,
        note="data generation uses identical deterministic arrays for baseline/search; validation inside training domain; result follows selection snapshot",
    )
    return report


def main():
    searches, baselines = collect("experiment_01_recovery", "main")
    rows = flatten(searches, baselines)
    csvwrite(ROOT / "experiment_01_recovery" / "all_methods.csv", rows)
    summary = []
    for kind in KINDS:
        for index in range(3):
            rr = [
                r
                for r in searches
                if r["parameters"]["kind"] == kind and r["parameters"]["index"] == index
            ]
            success = sum(r["metrics"]["structure_numerically_consistent"] for r in rr)
            certs = [
                read(Path(r["_path"]) / "structure_certificate.json")
                for r in rr
                if (Path(r["_path"]) / "structure_certificate.json").exists()
            ]
            lo, hi = wilson(success, len(rr))
            summary.append(
                dict(
                    kind=kind,
                    index=index,
                    n=len(rr),
                    structure_consistent=success,
                    wilson_low=lo,
                    wilson_high=hi,
                    certified_family=sum(
                        c.get("family_structure_matched", False)
                        and c.get("coefficients_match_theory_within_tolerance", False)
                        for c in certs
                    ),
                    exact_identity_proved=sum(c.get("exact_identity_proved", False) for c in certs),
                    ood=stats([r["metrics"]["ood"]["nrmse"] for r in rr]),
                    search_seconds=stats([r["search_wall_seconds"] for r in rr]),
                )
            )
    dump(ROOT / "experiment_01_recovery" / "summary.json", summary)
    table = mdtable(
        [
            "构型",
            "参数组",
            "结构数值一致",
            "独立族系数认证",
            "精确恒等",
            "95% Wilson区间",
            "OOD NRMSE中位数",
            "搜索秒中位数",
        ],
        [
            [
                r["kind"],
                r["index"],
                str(r["structure_consistent"]) + "/" + str(r["n"]),
                r["certified_family"],
                r["exact_identity_proved"],
                number(r["wilson_low"]) + "–" + number(r["wilson_high"]),
                number(r["ood"].get("median")),
                number(r["search_seconds"].get("median")),
            ]
            for r in summary
        ],
    )
    intro = "# 四类声管公式恢复结果\n\n正式搜索每个场景180秒上限，每个构型三组配对参数，每场景5次。预算相同，但找到训练MSE低于10⁻²⁰的式子可提前停止。多线程的随机种子不保证逐位复现；保存完整环境、配置和模型。预实验不计入下表。\n\n"
    formulas = mdtable(
        ["构型", "参数组", "种子", "复杂度", "冻结后公式"],
        [
            [
                r["parameters"]["kind"],
                r["parameters"]["index"],
                r["seed"],
                r["selected"]["complexity"],
                "`" + r["selected"]["refined"] + "`",
            ]
            for r in searches
        ],
    )
    (ROOT / "experiment_01_recovery" / "实验说明与结果.md").write_text(
        intro
        + table
        + "\n\n## 全部公式\n\n"
        + formulas
        + "\n\n数值结构一致不是严格符号恒等证明。每个运行的structure_certificate.json提供独立结构审查（如已生成）。候选误差、常数重新拟合、选择快照、配置、数据、检查点和日志全部保留。\n",
        encoding="utf-8",
    )
    ood_summary = []
    for kind in KINDS:
        for method in COLORS:
            group = [r for r in rows if r["kind"] == kind and r["method"] == method]
            ood_summary.append(
                dict(kind=kind, method=method, **stats([r["ood_nrmse"] for r in group]))
            )
    dump(ROOT / "experiment_02_extrapolation" / "ood_summary.json", ood_summary)
    csvwrite(ROOT / "experiment_02_extrapolation" / "segmented_errors.csv", rows)
    table2 = mdtable(
        ["构型", "方法", "运行数", "OOD中位数", "25%分位", "75%分位", "最差"],
        [
            [
                r["kind"],
                r["method"],
                r.get("n", 0),
                number(r.get("median")),
                number(r.get("q1")),
                number(r.get("q3")),
                number(r.get("max")),
            ]
            for r in ood_summary
        ],
    )
    (ROOT / "experiment_02_extrapolation" / "实验说明与结果.md").write_text(
        "# 未观测下游区域空间外推比较\n\n复用实验1按ID验证选定的全部模型，不进行OOD调参。区域为(0.4,0.6]、(0.6,0.8]、(0.8,1]；归一化分母为评价区间真实p的RMS。1%累计OOD NRMSE门槛定义可用距离。\n\n"
        + table2
        + "\n\nMLP为三层64单元tanh双精度网络，Adam最多12000步，验证集早停1500步，ReduceLROnPlateau，再用训练点LBFGS500步精修，按原验证集决定是否接受。MLP polished为统一追加最多3000步LBFGS且只按原ID验证接受的加强基线，原MLP结果保留。Chebyshev阶数6/10/14/18/22/26按ID验证选取。模态基线使用已知几何包络和有效波数，仅拟合两个线性系数，含额外物理先验。物理残差初始JSON的非符号基线采用样条导数；论文应优先使用physical_diagnostics.json中的MLP自动微分及Chebyshev/模态解析导数结果。\n\n全部分段误差、训练时间和可用距离在segmented_errors.csv。公式、训练模型和逐点预测分别位于实验1各运行子目录。图示预先固定中间参数、seed=0，不能把该曲线当作5次统计结果。压力图纵轴限制±1.2，超出范围的多项式曲线见右侧完整误差图。\n",
        encoding="utf-8",
    )
    if len(searches) >= 60 and len(baselines) >= 60:
        figures(searches, baselines, rows)
    sampling, sampling_base = collect("experiment_03_sampling", "sampling")
    sample_rows = flatten(sampling, sampling_base)
    csvwrite(ROOT / "experiment_03_sampling" / "all_methods.csv", sample_rows)
    sample_summary = []
    for kind in ["conical", "catenoidal"]:
        for n in [40, 80, 160, 320]:
            for scheme in ["uniform", "random"]:
                group = [
                    r
                    for r in sampling
                    if r["parameters"]["kind"] == kind and r["n"] == n and r["sampling"] == scheme
                ]
                s = sum(r["metrics"]["structure_numerically_consistent"] for r in group)
                lo, hi = wilson(s, len(group))
                certificates = [
                    read(Path(r["_path"]) / "structure_certificate.json")
                    for r in group
                    if (Path(r["_path"]) / "structure_certificate.json").exists()
                ]
                certified = sum(
                    c.get("family_structure_matched", False)
                    and c.get("coefficients_match_theory_within_tolerance", False)
                    for c in certificates
                )
                maxgaps = [
                    float(np.diff(np.load(Path(r["_path"]) / "data.npz")["xtrain"]).max())
                    for r in group
                ]
                sample_summary.append(
                    dict(
                        kind=kind,
                        n=n,
                        sampling=scheme,
                        runs=len(group),
                        success=s,
                        certified_family=certified,
                        wilson_low=lo,
                        wilson_high=hi,
                        ood=stats([r["metrics"]["ood"]["nrmse"] for r in group]),
                        max_sampling_gap_xi=max(maxgaps) if maxgaps else None,
                    )
                )
    dump(ROOT / "experiment_03_sampling" / "summary.json", sample_summary)
    tab3 = mdtable(
        [
            "构型",
            "N",
            "布点",
            "结构数值一致",
            "独立严格族认证",
            "95%数值一致率区间",
            "OOD中位数",
            "最大间隔ξ",
        ],
        [
            [
                r["kind"],
                r["n"],
                r["sampling"],
                str(r["success"]) + "/" + str(r["runs"]),
                str(r["certified_family"]) + "/" + str(r["runs"]),
                number(r["wilson_low"]) + "–" + number(r["wilson_high"]),
                number(r["ood"].get("median")),
                number(r["max_sampling_gap_xi"]),
            ]
            for r in sample_summary
        ],
    )
    (ROOT / "experiment_03_sampling" / "实验说明与结果.md").write_text(
        "# 采样数量与不规则测点实验\n\n选择圆锥和悬链线两类非平凡包络，中间参数κ=10π，b=0.65，μ=1.4。覆盖有理和cosh包络。点数40/80/160/320，两种采样，各5次，合计80次PySR；相同样本和输入分别训练全部基线。预算和算子与主实验相同。随机采样固定端点，内部N−2点为均匀随机；报告最大间隔而非强行筛除不利布点。\n\n"
        + tab3
        + "\n\n在本无噪声且含两个周期的设计中观察到的最低有效N，只是所测试40/80/160/320中的最小值；不能声称工程实测所需的普遍最少测点。5次统计的不确定性见Wilson区间。\n",
        encoding="utf-8",
    )
    if len(sampling) >= 80 and len(sampling_base) >= 80:
        sampling_figures(sample_rows, sample_summary)
    locations = []
    for r in searches:
        for name, loc in r["metrics"]["locations"].items():
            if name == "finite":
                continue
            locations.append(
                dict(
                    kind=r["parameters"]["kind"],
                    index=r["parameters"]["index"],
                    seed=r["seed"],
                    method="PySR",
                    quantity=name,
                    missed=loc["missed"],
                    false=loc["false"],
                    matched=loc["matched"],
                    mean_xi_error=loc["mean_xi_error"],
                    max_xi_error=loc["max_xi_error"],
                    mean_wavelength_error=loc["mean_wavelength_error"],
                    mean_amplitude_error=loc.get("mean_amplitude_error"),
                )
            )
    for r in baselines:
        methods = r["methods"].copy()
        polished = Path(r["_path"]) / "mlp_polished_result.json"
        if polished.exists():
            methods["MLP polished"] = read(polished)
        for method, result in methods.items():
            for name, loc in result["metrics"]["locations"].items():
                if name == "finite":
                    continue
                locations.append(
                    dict(
                        kind=r["parameters"]["kind"],
                        index=r["parameters"]["index"],
                        seed=r["seed"],
                        method=method,
                        quantity=name,
                        missed=loc["missed"],
                        false=loc["false"],
                        matched=loc["matched"],
                        mean_xi_error=loc["mean_xi_error"],
                        max_xi_error=loc["max_xi_error"],
                        mean_wavelength_error=loc["mean_wavelength_error"],
                        mean_amplitude_error=loc.get("mean_amplitude_error"),
                    )
                )
    csvwrite(ROOT / "experiment_05_locations" / "locations_all_methods.csv", locations)
    (ROOT / "experiment_05_locations" / "实验说明与结果.md").write_text(
        "# 下游声压节点与极值位置\n\n复用实验1冻结模型，只评价0.4<ξ<1。所有模型使用同一个2001点网格的三次样条，样条求根检测声压节点，样条一阶导数求根检测極值。真值也按同样规则处理。位置用匈牙利算法最小距离匹配，并删除超过四分之一波长的匹配；同时报告漏检和误检。位置误差除以2π/κ得到波长归一化误差。峰值幅度比较真实极值与匹配预测极值。\n\n相对于该统一网格插值参考的极小差异不能当作真实物理定位精度：2001点对应Δξ=5×10⁻⁴，需考虑ODE误差、插值误差和一维模型近似。此处未加入测量噪声。\n\n详见locations_all_methods.csv；每个运行JSON还存有全部真实与预测位置，便于重新检验匹配和漏检。\n",
        encoding="utf-8",
    )
    joint_summary = joint_report()
    audit_report = dict(
        main=audit(searches, baselines),
        sampling=audit(sampling, sampling_base),
        counts=dict(
            main_search=len(searches),
            main_baseline=len(baselines),
            sampling_search=len(sampling),
            sampling_baseline=len(sampling_base),
        ),
    )
    dump(ROOT / "docs" / "audit.json", audit_report)
    hashes = {
        str(p.relative_to(REPOSITORY_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in SOURCE_ROOT.glob("*.py")
    }
    dump(ROOT / "docs" / "code_sha256.json", hashes)
    dump(
        ROOT / "docs" / "summary.json",
        dict(
            main=summary,
            ood=ood_summary,
            sampling=sample_summary,
            joint=joint_summary,
            audit=audit_report,
        ),
    )
    print(json.dumps(audit_report, ensure_ascii=False))


def figures(searches, baselines, rows):
    fig, axes = plt.subplots(4, 2, figsize=(10, 11))
    for i, kind in enumerate(KINDS):
        r = next(
            r
            for r in searches
            if r["parameters"]["kind"] == kind and r["parameters"]["index"] == 1 and r["seed"] == 0
        )
        b = next(
            b
            for b in baselines
            if b["parameters"]["kind"] == kind and b["parameters"]["index"] == 1 and b["seed"] == 0
        )
        d = np.load(Path(r["_path"]) / "predictions.npz")
        db = np.load(Path(b["_path"]) / "predictions.npz")
        x = d["x"]
        truth = d["truth"]
        ax = axes[i, 0]
        ax.plot(x, truth, "k-", lw=2, label="ODE truth")
        ax.plot(x, d["pysr"], color=COLORS["PySR"], ls="--", label="PySR")
        mp = Path(b["_path"]) / "mlp_polished_predictions.npz"
        if mp.exists():
            polished = np.load(mp)["MLP_polished"]
            ax.plot(x, polished, color=COLORS["MLP polished"], lw=1, label="MLP polished")
        else:
            polished = db["MLP"]
            ax.plot(x, polished, color=COLORS["MLP"], lw=1, label="MLP")
        for method in ["Chebyshev", "Modal informed"]:
            ax.plot(x, db[method], color=COLORS[method], lw=1, alpha=0.8, label=method)
        ax.axvspan(0, 0.4, color="#eeeeee")
        ax.axvline(0.4, color="gray", ls=":")
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(kind)
        ax.set_ylabel("Normalized pressure")
        ax = axes[i, 1]
        for method, yp in [
            ("PySR", d["pysr"]),
            ("MLP polished", polished),
            *[(m, db[m]) for m in ["Chebyshev", "Modal informed"]],
        ]:
            mask = x > 0.4
            xx = x[mask]
            err = np.sqrt(np.cumsum((yp[mask] - truth[mask]) ** 2) / np.arange(1, len(xx) + 1))
            den = np.sqrt(np.cumsum(truth[mask] ** 2) / np.arange(1, len(xx) + 1))
            ax.semilogy(xx, np.maximum(err / den, 1e-14), label=method, color=COLORS[method])
        ax.axhline(0.01, color="gray", ls=":", label="1% threshold" if i == 0 else None)
        ax.set_ylabel("Cumulative OOD NRMSE")
        ax.set_title(kind + " error growth")
    for ax in axes[-1]:
        ax.set_xlabel("Axial coordinate xi")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, ROOT / "experiment_02_extrapolation" / "figures" / "pressure_and_error_growth")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, kind in zip(axes, KINDS):
        values = [
            [
                max(r["ood_nrmse"], 1e-14)
                for r in rows
                if r["kind"] == kind and r["method"] == method
            ]
            for method in COLORS
        ]
        ax.boxplot(values, labels=["PySR", "MLP", "MLP+", "Cheb", "Modal"], showfliers=True)
        ax.set_yscale("log")
        ax.set_title(kind)
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("OOD NRMSE, all 15 runs")
    fig.tight_layout()
    savefig(fig, ROOT / "experiment_02_extrapolation" / "figures" / "all_run_error_distribution")


def sampling_figures(rows, summary):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.7))
    for i, kind in enumerate(["conical", "catenoidal"]):
        for scheme, style, color, offset in [
            ("uniform", "-", "#2678ad", -5),
            ("random", "--", "#e28224", 5),
        ]:
            rr = sorted(
                [r for r in summary if r["kind"] == kind and r["sampling"] == scheme],
                key=lambda r: r["n"],
            )
            fractions = np.array([r["certified_family"] / r["runs"] for r in rr])
            bounds = np.array([wilson(r["certified_family"], r["runs"]) for r in rr])
            axes[i, 0].errorbar(
                [r["n"] + offset for r in rr],
                fractions,
                yerr=np.maximum(np.vstack([fractions - bounds[:, 0], bounds[:, 1] - fractions]), 0),
                fmt=style + "o",
                color=color,
                capsize=3,
                label=scheme,
            )
        axes[i, 0].set_ylim(-0.05, 1.05)
        axes[i, 0].set_title(kind + " certified recovery")
        axes[i, 0].set_ylabel("Target-family fraction (95% Wilson CI)")
        axes[i, 0].legend(loc="lower left", fontsize=8)
        for method in COLORS:
            for scheme, style in [("uniform", "-"), ("random", "--")]:
                meds = [
                    np.median(
                        [
                            r["ood_nrmse"]
                            for r in rows
                            if r["kind"] == kind
                            and r["n"] == n
                            and r["sampling"] == scheme
                            and r["method"] == method
                        ]
                    )
                    for n in [40, 80, 160, 320]
                ]
                axes[i, 1].semilogy(
                    [40, 80, 160, 320],
                    meds,
                    style + "o",
                    color=COLORS[method],
                    label=method + " " + scheme,
                )
        axes[i, 1].set_title(kind + " OOD")
        axes[i, 1].set_ylabel("Median OOD NRMSE")
    for ax in axes[-1]:
        ax.set_xlabel("Number of training points")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 1))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, ROOT / "experiment_03_sampling" / "figures" / "sampling_sensitivity")


def joint_report():
    folder = ROOT / "experiment_04_joint"
    results = []
    baseline = []
    for f in folder.glob("*/result.json"):
        r = read(f)
        r["_path"] = str(f.parent)
        results.append(r)
    for f in folder.glob("*/baseline_result.json"):
        r = read(f)
        baseline.append(r)
    rows = []
    for r in results:
        for region, m in r["metrics"].items():
            rows.append(
                dict(
                    stage=r["stage"],
                    seed=r["seed"],
                    method="PySR",
                    region=region,
                    nrmse=m["nrmse"],
                    max_abs=m["max_abs"],
                    expression=r["selected"]["refined"],
                    exact_identity_proved=r["exact_identity_proved"],
                )
            )
    for r in baseline:
        for method, v in r["methods"].items():
            for region, m in v["metrics"].items():
                rows.append(
                    dict(
                        stage=r["stage"],
                        seed=r["seed"],
                        method=method,
                        region=region,
                        nrmse=m["nrmse"],
                        max_abs=m["max_abs"],
                        expression=None,
                        exact_identity_proved=None,
                    )
                )
    csvwrite(folder / "all_results.csv", rows)
    table = mdtable(
        ["阶段", "种子", "方法", "测试区域", "NRMSE", "最大绝对误差"],
        [
            [
                r["stage"],
                r["seed"],
                r["method"],
                r["region"],
                number(r["nrmse"]),
                number(r["max_abs"]),
            ]
            for r in rows
        ],
    )
    (folder / "实验说明与结果.md").write_text(
        "# 圆锥声管双变量联合公式恢复\n\n变量t=kx、η=kx₀。ODE为pₜₜ+2pₜ/(η+t)+p=0，初值p(0)=1、pₜ(0)=0。每条η曲线由DOP853数值积分生成，解析式仅核验。训练η=15/18/21/24/27/30，每曲线160点t∈[0,4π]，总960点。验证η=16.5/19.5/22.5/25.5/28.5、129个独立t点，仍位于ID矩形。\n\n相位OOD为t∈(4π,10π]、训练η；几何OOD为t∈[0,4π]、η=8/10/12/14/31/34/37/40；联合OOD同时改变这两个范围。全部域的具体网格和数值解保存在data.npz。PySR只接收t和η，不输入理论解或复合特征。\n\n初始main搜索为40个种群、80个个体和600秒上限；预实验300秒单独记录。focused为后诊断复合加强配置：+*/sin/cos、非代数参数最多3节点、常数优化100轮；保留目标树但并非仅允许仿射相位。其成功率不能与main合并，也不能单独归因于算子消融。focused_extended如存在为失败种子的独立重启，900秒与16线程，单独报告。\n\n联合MLP和二维总阶Chebyshev原结果保留；baseline_refined为原ID验证下的MLP精修与张量Chebyshev修正，η阶最多5以避免仅6条曲线的秩不足。模态基线包含额外圆锥公式先验。\n\n"
        + table
        + "\n\n所有种子和失败候选保留。independent_verification.json提供仅看式子/理论参数的族系数认证、符号ODE/初值验证和低/高η分段指标。压力误差小不能代表结构正确，各候选是否存在非线性相位或入口导数异常，以本次独立核验文件为准。展示与发布使用canonical_expression.json中消除可去奇点的规范式，连续延拓与原树定义域差异明确记录。联合式只验证理想同一家族指定初值下的相似性参数迁移。\n",
        encoding="utf-8",
    )
    selected = [r for r in results if r["stage"] == "main" and r["seed"] == 0]
    if selected:
        r = selected[0]
        d = np.load(Path(r["_path"]) / "data.npz")
        pred = np.load(Path(r["_path"]) / "predictions.npz")
        X = d["joint_ood_X"]
        y = d["joint_ood_y"]
        yp = pred["PySR_joint_ood"]
        ts = np.unique(X[:, 0])
        etas = np.unique(X[:, 1])
        error = abs(yp - y).reshape(len(etas), len(ts))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
        for ax, mask in zip(axes, [etas <= 14, etas >= 31]):
            im = ax.pcolormesh(
                ts,
                etas[mask],
                np.log10(np.maximum(error[mask], 1e-14)),
                shading="nearest",
                cmap="magma",
                vmin=-14,
                vmax=np.ceil(np.log10(max(float(np.max(error)), 1e-14))),
            )
            fig.colorbar(im, ax=ax, label="log10 absolute error")
            ax.set_ylabel("Unseen eta")
            ax.set_xlabel("Unseen phase t")
            ax.set_title("Main seed 0: low eta" if mask[0] else "Main seed 0: high eta")
        fig.tight_layout()
        savefig(fig, folder / "figures" / "joint_ood_error_heatmaps")
    return dict(search_runs=len(results), baseline_runs=len(baseline), results=rows)


if __name__ == "__main__":
    main()
