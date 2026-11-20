"""Summarize only this run's saved models; never import archived results as trials."""

import collections
import csv
from .common import *


def statistics(values):
    a = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(a):
        return dict(
            n_finite=0, mean=None, sd=None, median=None, q1=None, q3=None, min=None, max=None
        )
    return dict(
        n_finite=len(a),
        mean=float(np.mean(a)),
        sd=float(np.std(a, ddof=1)) if len(a) > 1 else None,
        median=float(np.median(a)),
        q1=float(np.quantile(a, 0.25)),
        q3=float(np.quantile(a, 0.75)),
        min=float(a.min()),
        max=float(a.max()),
    )


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(clean_json(rows))


def extract(folder, method="PySR"):
    r = json.loads((folder / "result.json").read_text())
    j = r["job"]
    model = r if method == "PySR" else r["methods"][method]
    m = model["metrics"]
    nonlinear = j["scenario"] == "nonlinear_burgers"
    error = [None] * 10
    hfile = folder / "harmonics.npz" if method == "PySR" else folder / method / "harmonics.npz"
    if nonlinear:
        with np.load(hfile) as h:
            ref = h["reference_amplitude"]
            error = np.sqrt(
                np.sum((h["sine"] - ref) ** 2 + h["cosine"] ** 2, axis=0) / np.sum(ref**2, axis=0)
            ).tolist()
    idn = m.get("id_test", {}).get("nrmse")
    ood = m["ood"]["nrmse"]
    basic = bool(
        nonlinear
        and idn is not None
        and idn <= 0.01
        and ood is not None
        and ood <= 0.05
        and error[0] <= 0.05
        and error[1] <= 0.05
    )
    row = dict(
        batch=j["phase"],
        method=method,
        run_id=j["run_id"],
        scenario=j["scenario"],
        kind=j.get("kind"),
        parameter_group=j.get("parameter_group"),
        snr=j["snr_db"],
        seed=j["pysr_seed"],
        budget_seconds=j["budget_seconds"],
        configuration=j.get("configuration_label", j["run_id"].split("_snr")[0]),
        condition=j.get("efficiency_group"),
        id_nrmse=idn,
        ood_nrmse=ood,
        basic_reached=basic if nonlinear else None,
        target_reached=basic if nonlinear else bool(ood is not None and ood <= 0.01),
        prediction_one_percent=bool(ood is not None and ood <= 0.01),
        residual=m.get("burgers_residual_normalized_rms", m.get("webster_relative_residual_rms")),
        search_seconds=r.get("search_call_wall_seconds") if method == "PySR" else None,
        total_seconds=r.get("end_to_end_seconds"),
        fit_seconds=model.get("fit_seconds"),
        path=str(folder.relative_to(ROOT)),
        source=str((folder / "result.json").relative_to(ROOT)),
    )
    for n, e in enumerate(error, 1):
        row["harmonic_%d_coefficient_error" % n] = e
    if method == "PySR":
        assert json.loads((folder / "selected_before_ood.json").read_text()) == r["selected"]
        row["expression"] = r["selected"]["refined"]
    else:
        row["expression"] = ""  # Parameters/weights stay in the method's own files.
    cost = folder / "cost_summary.json"
    if cost.exists():
        c = json.loads(cost.read_text())
        row["first_basic_seconds"] = (c.get("first_basic") or {}).get("first_observed_seconds")
        row["first_ood_one_percent_seconds"] = (c.get("first_ood_one_percent") or {}).get(
            "first_observed_seconds"
        )
    return row


def generate():
    dest = ROOT / "summary"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted((ROOT / "supplemental").glob("*/*/result.json")):
        rows.append(extract(p.parent))
        bp = p.parent / "baselines/result.json"
        if bp.exists():
            for method in json.loads(bp.read_text())["methods"]:
                rows.append(extract(bp.parent, method))
    buckets = collections.defaultdict(list)
    for row in rows:
        key = tuple(
            str(row.get(k))
            for k in [
                "batch",
                "scenario",
                "kind",
                "parameter_group",
                "snr",
                "method",
                "configuration",
                "budget_seconds",
                "condition",
            ]
        )
        buckets[key].append(row)
    stats = []
    harmonics = []
    metrics = [
        "id_nrmse",
        "ood_nrmse",
        "residual",
        "search_seconds",
        "total_seconds",
        "fit_seconds",
        "first_basic_seconds",
        "first_ood_one_percent_seconds",
    ] + ["harmonic_%d_coefficient_error" % n for n in range(1, 11)]
    for key, rs in sorted(buckets.items()):
        meta = dict(
            zip(
                [
                    "batch",
                    "scenario",
                    "kind",
                    "parameter_group",
                    "snr",
                    "method",
                    "configuration",
                    "budget_seconds",
                    "condition",
                ],
                key,
            )
        )
        for name in metrics:
            s = statistics([r.get(name) for r in rs])
            stats.append(
                dict(
                    meta,
                    metric=name,
                    n_total=len(rs),
                    n_missing=len(rs) - s["n_finite"],
                    target_count=sum(r["target_reached"] for r in rs),
                    one_percent_count=sum(r["prediction_one_percent"] for r in rs),
                    **s,
                )
            )
        if rs[0]["scenario"] == "nonlinear_burgers":
            for n in range(1, 11):
                vals = [r["harmonic_%d_coefficient_error" % n] for r in rs]
                harmonics.append(
                    dict(
                        meta,
                        order=n,
                        n_total=len(rs),
                        at_5_percent=sum(v is not None and v <= 0.05 for v in vals),
                        at_10_percent=sum(v is not None and v <= 0.1 for v in vals),
                    )
                )
    winner_path = ROOT / "selection/confirmed_configuration.json"
    winner = json.loads(winner_path.read_text()) if winner_path.exists() else None
    jobs = [json.loads(p.read_text()) for p in sorted((ROOT / "jobs").glob("*.json"))]
    incomplete = [
        dict(phase=j["phase"], run_id=j["run_id"], path=str(run_folder(j).relative_to(ROOT)))
        for j in jobs
        if not (run_folder(j) / "result.json").exists()
    ]
    write_csv(dest / "all_runs.csv", rows)
    write_csv(dest / "grouped_statistics.csv", stats)
    write_csv(dest / "harmonic_successes.csv", harmonics)
    dump(
        dest / "summary.json",
        dict(rows=rows, statistics=stats, winner=winner, incomplete_jobs=incomplete),
    )
    lines = [
        "# 本次补充实验结果",
        "",
        "所有统计均来自本次工作区。reference_results中的历史摘要不参与训练、选模或本次成功率计数。",
        "",
        "非线性基础标准为独立域内声压NRMSE≤1%、域外≤5%、基频与二次谐波相位敏感系数误差各≤5%；五个确认种子至少三次同时通过。线性声场的场景目标为域外声压NRMSE≤1%，不与非线性目标等同。",
        "",
        (
            "最终配置已通过本次无噪声确认。"
            if winner
            else "本次尚无通过五种子基础确认的配置；预检结果不代表科学目标达成。"
        ),
        "",
        "| 批次 | 场景 | 构型 | 参数组 | SNR | 方法 | 配置 | 预算秒 | 效率组 | 完成数 | 场景通过 | 域外NRMSE中位数 | 基频误差中位数 | 二次误差中位数 |",
        "|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(x):
        return "不适用" if x is None else "%.6g" % x

    for key, rs in sorted(buckets.items()):
        lines.append(
            "|"
            + "|".join(
                [
                    *key,
                    str(len(rs)),
                    "%d/%d" % (sum(r["target_reached"] for r in rs), len(rs)),
                    *[
                        fmt(statistics([r.get(m) for r in rs])["median"])
                        for m in [
                            "ood_nrmse",
                            "harmonic_1_coefficient_error",
                            "harmonic_2_coefficient_error",
                        ]
                    ],
                ]
            )
            + "|"
        )
    lines += [
        "",
        "## 解释与文件",
        "",
        "四分位数、范围、均值、样本SD和缺失数见grouped_statistics.csv；第1—10阶通过次数见harmonic_successes.csv。未完成任务数：%d，原始日志全部保留。"
        % len(incomplete),
        "规模预热为0，非线性默认的函数嵌套与子树约束为空；仍有表达式最大规模和深度。搜索只使用原始声压；谐波与物理残差在表达式冻结后评价。有限表达式逼近不等于隐式解的严格恒等恢复。",
        "独立效率A/B为线性无噪声/40 dB，C/D为非线性无噪声/40 dB。首次通过时间来自未做事后精修的原始快照，是离线回溯统计；未通过者按预算截尾留空，不能填成预算秒数的成功。最终模型与曾通过的快照可能不同。",
        "同数据MLP并未与PySR匹配计算预算，不作等算力比较。多线程时间预算搜索不保证逐位复现或固定成功数。",
        "supplemental/按阶段保存模型和数组；jobs/、manifests/保存任务；provenance/保存协议；status.json和controller_finished.json记录状态。",
    ]
    (dest / "实验结果总结.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "SUPPLEMENTAL_SUMMARY",
        len(rows),
        "model-method rows;",
        len(incomplete),
        "incomplete jobs",
        flush=True,
    )
    return rows


if __name__ == "__main__":
    generate()
