"""Final artifact completeness and joint data consistency; never rerun training."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .acoustic_core import ROOT, dump
from .paths import REPOSITORY_ROOT, SOURCE_ROOT


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    errors = []
    rows = []
    counts = {}
    joint_reference = None
    for experiment, expected in [
        ("experiment_01_recovery", 64),
        ("experiment_03_sampling", 81),
        ("experiment_04_joint", 12),
    ]:
        files = sorted((ROOT / experiment).glob("*/result.json"))
        counts[experiment] = len(files)
        if len(files) != expected:
            errors.append(f"{experiment}: {len(files)} searches; expected {expected}")
        for f in files:
            r = read(f)
            folder = f.parent
            is_joint = experiment == "experiment_04_joint"
            required = [
                "configuration.json",
                "candidates.json",
                "data.npz",
                "predictions.npz",
                "selected_before_ood.json",
                "canonical_expression.json",
                "independent_verification.json" if is_joint else "structure_certificate.json",
            ]
            for name in required:
                if not (folder / name).is_file() or not (folder / name).stat().st_size:
                    errors.append(str(folder.relative_to(ROOT)) + "/" + name + ": missing or empty")
            artifacts = {}
            for name in ["checkpoint.pkl", "hall_of_fame.csv"]:
                matches = [
                    p for p in (folder / "pysr").rglob(name) if p.is_file() and p.stat().st_size
                ]
                if not matches:
                    errors.append(
                        str(folder.relative_to(ROOT)) + "/pysr/**/" + name + ": missing or empty"
                    )
                artifacts[name] = str(matches[0].relative_to(ROOT)) if matches else ""
            snapshot = read(folder / "selected_before_ood.json")
            if snapshot.get("refined") != r["selected"]["refined"]:
                errors.append(str(folder.relative_to(ROOT)) + ": selected expression changed")
            if (folder / "selected_before_ood.json").stat().st_mtime > f.stat().st_mtime:
                errors.append(str(folder.relative_to(ROOT)) + ": snapshot timestamp after result")
            with np.load(folder / "data.npz") as d:
                Xtrain, ytrain, Xval, yval = d["Xtrain"], d["ytrain"], d["Xval"], d["yval"]
                if not all(np.isfinite(a).all() for a in [Xtrain, ytrain, Xval, yval]):
                    errors.append(
                        str(folder.relative_to(ROOT)) + ": nonfinite training/validation data"
                    )
                if len(Xtrain) != len(ytrain) or len(Xval) != len(yval):
                    errors.append(str(folder.relative_to(ROOT)) + ": data shape mismatch")
                if is_joint:
                    arrays = {k: d[k].copy() for k in d.files}
                    if joint_reference is None:
                        joint_reference = arrays
                    elif any(
                        not np.array_equal(arrays[k], joint_reference[k]) for k in joint_reference
                    ):
                        errors.append(
                            str(folder.relative_to(ROOT))
                            + ": joint arrays differ from other search stages"
                        )
            rows.append(
                dict(
                    experiment=experiment,
                    stage=r["stage"],
                    seed=r["seed"],
                    directory=str(folder.relative_to(ROOT)),
                    training_points=len(Xtrain),
                    validation_points=len(Xval),
                    validation_mse=r["selected"]["validation_mse"],
                    expression=r["selected"]["refined"],
                    checkpoint=artifacts["checkpoint.pkl"],
                    pareto_csv=artifacts["hall_of_fame.csv"],
                    result_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                    data_sha256=hashlib.sha256((folder / "data.npz").read_bytes()).hexdigest(),
                )
            )
    baseline_counts = {}
    for experiment, expected in [("experiment_01_recovery", 60), ("experiment_03_sampling", 80)]:
        files = list((ROOT / experiment / "baselines").glob("*/result.json"))
        baseline_counts[experiment] = len(files)
        if len(files) != expected:
            errors.append(experiment + ": baseline count mismatch")
        for f in files:
            for name in [
                "mlp.pt",
                "predictions.npz",
                "physical_diagnostics.json",
                "mlp_polished.pt",
                "mlp_polished_predictions.npz",
                "mlp_polished_result.json",
            ]:
                if not (f.parent / name).is_file():
                    errors.append(str(f.parent.relative_to(ROOT)) + "/" + name + ": missing")
            if not any(
                (f.parent / name).is_file()
                for name in ["data.npz", "data_reconstructed_from_configuration.npz"]
            ):
                errors.append(str(f.parent.relative_to(ROOT)) + ": no baseline data archive")
    joint_files = list((ROOT / "experiment_04_joint").glob("*/baseline_result.json"))
    baseline_counts["experiment_04_joint"] = len(joint_files)
    if len(joint_files) != 10:
        errors.append("joint: expected five original and five refined baselines")
    for f in joint_files:
        if f.parent.name.endswith("_baseline"):
            with np.load(f.parent / "data.npz") as d:
                for key in joint_reference:
                    if not np.array_equal(d[key], joint_reference[key]):
                        errors.append(
                            str(f.parent.relative_to(ROOT))
                            + ": joint baseline data mismatch "
                            + key
                        )
        if not (f.parent / "predictions.npz").is_file():
            errors.append(str(f.parent.relative_to(ROOT)) + ": baseline predictions missing")
    figures = sorted(p for p in ROOT.glob("experiment_*/figures/*.png"))
    if len(figures) != 8:
        errors.append("expected eight scientific PNG figures")
    for f in figures:
        if not f.with_suffix(".svg").is_file():
            errors.append(str(f.relative_to(ROOT)) + ": SVG counterpart missing")
    audit = read(ROOT / "docs" / "audit.json")
    if not all(audit[k]["passed"] for k in ["main", "sampling"]):
        errors.append("existing main/sampling data audit failed")
    source_hash = read(ROOT / "source" / "provenance.json")["original_document_sha256"]
    if source_hash.upper() != "DC0990400719AC861942DD8AC04213A69E3C2C20B71143EE8B743C46C093C63A":
        errors.append("source DOCX no longer matches original desktop hash")
    hashes = {
        str(p.relative_to(REPOSITORY_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in SOURCE_ROOT.glob("*.py")
    }
    dump(ROOT / "docs" / "code_sha256.json", hashes)
    with (ROOT / "docs" / "search_run_inventory.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        searches=sum(counts.values()),
        search_counts=counts,
        baseline_counts=baseline_counts,
        figures=[str(p.relative_to(ROOT)) for p in figures],
        original_source_docx_sha256=source_hash,
        source_document_distributed=False,
        passed=not errors,
        errors=errors,
        note="artifact presence, frozen expression, joint-array consistency and source integrity audit; not an additional scientific recovery test",
    )
    dump(ROOT / "docs" / "archive_audit.json", out)
    print(json.dumps(out, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
