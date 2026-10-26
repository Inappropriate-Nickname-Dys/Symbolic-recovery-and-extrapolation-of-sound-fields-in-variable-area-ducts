"""Read actual artifacts; report completeness separately from scientific success."""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def artifact(root, job):
    script = job.get("script", "run_search.py")
    seed = job.get("seed", 0)
    if script == "run_joint.py":
        stage = job.get("stage", "pilot")
        return (
            root
            / "experiment_04_joint"
            / ("seed%d_%s" % (seed, stage))
            / ("baseline_result.json" if stage == "baseline" else "result.json")
        )
    if script == "joint_baseline_refinement.py":
        return (
            root
            / "experiment_04_joint"
            / ("seed%d_baseline_refined" % seed)
            / "baseline_result.json"
        )
    label = "%s_%d_N%d_%s_seed%d" % (
        job["kind"],
        job.get("index", 1),
        job.get("n", 160),
        job.get("sampling", "uniform"),
        seed,
    )
    experiment = root / job.get("experiment", "experiment_01_recovery")
    if script == "run_search.py":
        return experiment / (label + "_" + job.get("stage", "pilot")) / "result.json"
    filename = "physical_diagnostics.json" if script == "physical_diagnostics.py" else "result.json"
    return experiment / "baselines" / label / filename


def family_ok(certificate):
    return bool(
        certificate.get("family_structure_matched")
        and certificate.get("coefficients_match_theory_within_tolerance")
    )


def statistics(values):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(
        total=len(values),
        finite=len(finite),
        median=float(np.median(finite)) if finite else None,
        minimum=min(finite) if finite else None,
        maximum=max(finite) if finite else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory inside root; absent means read-only",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors, manifest_rows, groups = [], [], defaultdict(list)
    manifests = sorted(root.glob("*manifest.json"))
    if len(manifests) != 14:
        errors.append("Expected the 14 archived manifests; found %d" % len(manifests))
    for manifest in manifests:
        jobs = read(manifest)
        missing = [
            str(artifact(root, j).relative_to(root))
            for j in jobs
            if not artifact(root, j).is_file()
        ]
        errors.extend("Missing artifact: " + name for name in missing)
        status_path = root / "logs" / (manifest.stem + "_status.json")
        failed, absent_labels = [], []
        if status_path.exists():
            records = read(status_path)
            failed = [v["label"] for v in records if v["returncode"] != 0]
            absent_labels = sorted(set(j["label"] for j in jobs) - set(v["label"] for v in records))
            errors.extend("Failed process: " + name for name in failed)
            errors.extend("Missing process record: " + name for name in absent_labels)
        else:
            errors.append("Missing dispatch status: " + str(status_path.relative_to(root)))
        manifest_rows.append(
            dict(
                manifest=manifest.name,
                expected_jobs=len(jobs),
                present_jobs=len(jobs) - len(missing),
                failed=failed,
                missing_process_records=absent_labels,
            )
        )
    for name in ["verification.json", "solver_diagnostics.json"]:
        if not (root / "experiment_00_generator" / name).is_file():
            errors.append("Missing generator output: " + name)
    for experiment in ["experiment_01_recovery", "experiment_03_sampling", "experiment_04_joint"]:
        for path in sorted((root / experiment).glob("*/result.json")):
            result = read(path)
            selected = result["selected"]
            for name in [
                "data.npz",
                "predictions.npz",
                "configuration.json",
                "candidates.json",
                "selected_before_ood.json",
                "canonical_expression.json",
            ]:
                if not (path.parent / name).is_file():
                    errors.append(
                        "Missing search artifact: " + str((path.parent / name).relative_to(root))
                    )
            snapshot = path.parent / "selected_before_ood.json"
            if snapshot.exists() and read(snapshot).get("refined") != selected["refined"]:
                errors.append("Frozen expression differs: " + str(path.relative_to(root)))
            joint = experiment == "experiment_04_joint"
            certpath = path.parent / (
                "independent_verification.json" if joint else "structure_certificate.json"
            )
            if not certpath.exists():
                errors.append("Missing certificate: " + str(certpath.relative_to(root)))
                certificate = {}
            else:
                certificate = read(certpath)
                if joint:
                    certificate = certificate["certificate"]
            metrics = result["metrics"]
            regions = ["phase_ood", "geometry_ood", "joint_ood"] if joint else ["ood"]
            entry = dict(
                directory=str(path.parent.relative_to(root)),
                family_recovered=family_ok(certificate),
                exact_identity_proved=bool(certificate.get("exact_identity_proved")),
                numerical_consistency=metrics.get("structure_numerically_consistent"),
                errors={r: metrics[r]["nrmse"] for r in regions},
            )
            group = experiment + "/" + result["stage"]
            groups[group].append(entry)
    summaries = {}
    for key, entries in sorted(groups.items()):
        regions = entries[0]["errors"]
        summaries[key] = dict(
            runs=len(entries),
            family_recovered=sum(e["family_recovered"] for e in entries),
            exact_identity_proved=sum(e["exact_identity_proved"] for e in entries),
            numerical_consistency=sum(bool(e["numerical_consistency"]) for e in entries),
            error_statistics={r: statistics([e["errors"][r] for e in entries]) for r in regions},
            cases=entries,
        )
    copied_changed = []
    provenance = root / "docs" / "reproduction_provenance.json"
    if provenance.exists():
        for f in read(provenance)["copied_files"]:
            # Original experiment code and task definitions must remain frozen.
            if f["path"].startswith("src/") or f["path"].endswith("manifest.json"):
                path = root / f["path"]
                if (
                    not path.exists()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != f["sha256"]
                ):
                    copied_changed.append(f["path"])
        errors.extend("Changed experiment source or manifest: " + n for n in copied_changed)
    for name in ["audit.json", "archive_audit.json"]:
        path = root / "docs" / name
        if not path.exists():
            errors.append("Missing final audit: docs/" + name)
        elif name == "archive_audit.json" and not read(path).get("passed"):
            errors.append("Native archive audit failed; inspect docs/archive_audit.json")
        elif name == "audit.json" and not all(
            read(path)[k]["passed"] for k in ["main", "sampling"]
        ):
            errors.append("Native train/validation/baseline consistency audit failed")
    cleanup_path = root / "docs" / "negligible_constant_cleanup.json"
    cleanup = read(cleanup_path) if cleanup_path.exists() else []
    if not cleanup_path.exists():
        errors.append("Missing independent pruning record")
    cleanup_summary = dict(
        candidates=len(cleanup),
        accepted=sum(bool(c["accepted"]) for c in cleanup),
        accepted_family=sum(
            bool(c["accepted"]) and family_ok(c.get("certificate", {})) for c in cleanup
        ),
    )
    output = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        root=str(root),
        complete_and_consistent=not errors,
        errors=errors,
        expected_manifest_jobs=sum(m["expected_jobs"] for m in manifest_rows),
        manifests=manifest_rows,
        stages=summaries,
        independent_pruning=cleanup_summary,
        definition="Completeness is separate from recovery. A classified scientific failure is still a valid completed run.",
    )
    lines = [
        "# 本次复现实验实际结果",
        "",
        "结果根目录：`" + str(root) + "`。",
        "",
        "以下统计直接读取本次运行文件，原始搜索与后处理分开。",
        "",
        "文件和流程完整性检查：" + ("通过" if not errors else "未通过，请先处理下列缺项") + "。",
        "",
        "| 阶段 | 次数 | 函数族与系数通过 | 严格恒等 | OOD指标中位数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, row in summaries.items():
        score = "; ".join(
            r
            + "="
            + ("%.6g" % v["median"] if v["median"] is not None else "NA")
            + " (finite %d/%d)" % (v["finite"], v["total"])
            for r, v in row["error_statistics"].items()
        )
        lines.append(
            "| %s | %d | %d | %d | %s |"
            % (key, row["runs"], row["family_recovered"], row["exact_identity_proved"], score)
        )
    lines.extend(
        [
            "",
            "独立稀疏化：候选%d，接受%d，其中目标族及系数通过%d。它们没有计入上表原始恢复率。"
            % (
                cleanup_summary["candidates"],
                cleanup_summary["accepted"],
                cleanup_summary["accepted_family"],
            ),
            "",
            "完整性检查通过只表示运行与文件完整，不保证恢复率或误差等同旧论文。数值结构一致、函数族认证和严格恒等分别存于JSON。",
            "",
            "## 运行缺项",
            "",
        ]
    )
    lines.extend(["- " + e for e in errors] if errors else ["无。"])
    lines.extend(
        ["", "## 每批任务完整性", "", "| 清单 | 应有任务 | 已有结果 |", "| --- | --- | --- |"]
    )
    lines.extend(
        "| %s | %d | %d |" % (r["manifest"], r["expected_jobs"], r["present_jobs"])
        for r in manifest_rows
    )
    if args.output_dir:
        target = args.output_dir.resolve()
        if root != target and root not in target.parents:
            parser.error("--output-dir must be inside the inspected project root")
        target.mkdir(parents=True, exist_ok=True)
        (target / "reproduction_check.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (target / "本次复现结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
