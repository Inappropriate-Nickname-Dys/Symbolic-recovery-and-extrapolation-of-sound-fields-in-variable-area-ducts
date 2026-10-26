#!/usr/bin/env python3
"""Run the archived experiment design using paths and environments in config.json."""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

REPO = Path(__file__).resolve().parent
EXPERIMENTS = [
    "experiment_00_generator",
    "experiment_01_recovery",
    "experiment_02_extrapolation",
    "experiment_03_sampling",
    "experiment_04_joint",
    "experiment_05_locations",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_config(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("config.schema_version must be 1")
    root = Path(value["paths"]["repository_root"]).expanduser().resolve()
    if root != REPO:
        raise ValueError(
            "Configured repository_root differs from this checkout. "
            "Run python3 reproduce.py --init-config and edit config.local.json."
        )
    runs = Path(value["paths"]["runs_root"]).expanduser().resolve()
    if REPO not in runs.parents:
        raise ValueError("runs_root must be a subdirectory of this repository")
    for key in ["workers", "joint_workers", "julia_threads", "blas_threads"]:
        v = value["execution"][key]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError("execution." + key + " must be a positive integer")
    return value


def build_plan(config, stage, config_path=None, run_root=None):
    """Return commands without starting programs or creating output files."""
    envs = config["environments"]
    config_path = config_path or REPO / "config.json"
    run_root = run_root or Path(config["paths"]["runs_root"]) / config["execution"]["run_name"]
    reg, plot = envs["regression_python"], envs["analysis_python"]
    workers, joint_workers = config["execution"]["workers"], config["execution"]["joint_workers"]
    steps = []

    def module(label, name, args=(), analysis=False, allow_failure=False):
        steps.append(
            dict(
                label=label,
                command=[
                    plot if analysis else reg,
                    "-m",
                    "acoustic_technique." + name,
                    *map(str, args),
                ],
                allow_failure=allow_failure,
            )
        )

    def dispatch(manifest, count=workers):
        module(manifest[:-5], "dispatch", [manifest, "--workers", count])

    def environment():
        for role in ["regression", "analysis"]:
            module(
                "environment_" + role,
                "environment_check",
                ["--config", str(config_path), "--role", role],
                analysis=role == "analysis",
            )
        steps.append(
            dict(
                label="julia_version",
                command=[envs["julia_executable"], "--startup-file=no", "--version"],
                allow_failure=False,
            )
        )

    def generator():
        module("generator", "acoustic_core")

    def pilot():
        dispatch("pilot_manifest.json")
        dispatch("joint_pilot_manifest.json", 1)

    def main_runs():
        for manifest in ["main_manifest.json", "baseline_manifest.json", "polish_manifest.json"]:
            dispatch(manifest)

    def sampling():
        for manifest in [
            "sampling_manifest.json",
            "sampling_baseline_manifest.json",
            "sampling_polish_manifest.json",
        ]:
            dispatch(manifest)
        dispatch("sampling_extended_manifest.json", 1)

    def joint():
        for manifest in [
            "joint_main_manifest.json",
            "joint_baseline_manifest.json",
            "joint_refined_manifest.json",
            "joint_focused_manifest.json",
        ]:
            dispatch(manifest, joint_workers)
        dispatch("joint_extended_manifest.json", 1)

    def post():
        module(
            "main_certificates",
            "structure_certificate",
            ["--experiment", "experiment_01_recovery", "--self-check"],
        )
        module(
            "sampling_certificates",
            "structure_certificate",
            ["--experiment", "experiment_03_sampling"],
        )
        for name in [
            "final_audit",
            "prune_negligible_constants",
            "summarize",
            "supplementary_figures",
            "archive_audit",
        ]:
            module(name, name, analysis=True, allow_failure=name == "archive_audit")
        module(
            "actual_result_check",
            "result_check",
            ["--root", str(run_root), "--output-dir", str(run_root / "docs/reproduction_check")],
            analysis=True,
        )

    actions = dict(
        env=environment,
        generator=generator,
        pilot=pilot,
        main=main_runs,
        sampling=sampling,
        joint=joint,
        post=post,
    )
    if stage == "all":
        for name in ["env", "generator", "pilot", "main", "sampling", "joint", "post"]:
            actions[name]()
    elif stage == "smoke":
        environment()
        generator()
        module(
            "smoke_search",
            "run_search",
            ["--kind", "uniform", "--index", 1, "--seed", 0, "--seconds", 15, "--stage", "smoke"],
        )
        module("smoke_baseline", "run_baselines", ["--kind", "uniform", "--index", 1, "--seed", 0])
        module(
            "smoke_polishing",
            "physical_diagnostics",
            ["--kind", "uniform", "--index", 1, "--seed", 0],
        )
        module("smoke_certification", "structure_certificate", ["--self-check"])
        module("smoke_final_audit", "final_audit", analysis=True)
        module("smoke_result", "smoke_report", analysis=True)
    else:
        actions[stage]()
    return steps


def prepare_workspace(config, run_root, profile):
    manifests = sorted((REPO / "experiments/manifests").glob("*manifest.json"))
    if len(manifests) != 14:
        raise ValueError("Exactly 14 archived manifests are required")
    source_hashes = {
        str(p.relative_to(REPO)): digest(p)
        for p in sorted((REPO / "src/acoustic_technique").glob("*.py"))
    }
    frozen = dict(
        config=config,
        profile=profile,
        source_hashes=source_hashes,
        entrypoint_hashes={name: digest(REPO / name) for name in ["reproduce.py", "run.sh"]},
        julia_project_hashes={
            name: digest(Path(config["environments"]["julia_project"]) / name)
            for name in ["Project.toml", "Manifest.toml"]
        },
        manifest_hashes={p.name: digest(p) for p in manifests},
    )
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    path = run_root / "docs/reproduction_provenance.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise ValueError("Configuration, code or manifests changed. Choose a fresh --run-name.")
        for f in old["copied_files"]:
            if digest(run_root / f["path"]) != f["sha256"]:
                raise ValueError("Workspace manifest changed: " + f["path"])
        return
    existing = [p for p in run_root.iterdir() if p.name != ".run.lock"]
    if existing:
        raise ValueError("Unrecognized nonempty run directory; choose a fresh --run-name")
    for name in ["docs", "logs", "source", *EXPERIMENTS]:
        (run_root / name).mkdir(parents=True, exist_ok=True)
    copied = []
    for p in manifests:
        target = run_root / p.name
        shutil.copy2(p, target)
        copied.append(dict(path=p.name, sha256=digest(target)))
    shutil.copy2(REPO / "reference_results/provenance.json", run_root / "source/provenance.json")
    save(
        path,
        dict(
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            root=str(run_root),
            fingerprint=fingerprint,
            copied_results=False,
            copied_files=copied,
            **frozen,
        ),
    )
    save(run_root / "docs/config_used.json", config)


@contextlib.contextmanager
def exclusive_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This run directory is already active; do not start a second driver")
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def execute(steps, config, root, stage):
    env = os.environ.copy()
    env.update(
        ACOUSTIC_REPOSITORY_ROOT=str(REPO),
        ACOUSTIC_OUTPUT_ROOT=str(root),
        PYTHONPATH=str(REPO / "src"),
        PYTHONUNBUFFERED="1",
        PYTHON_JULIACALL_THREADS=str(config["execution"]["julia_threads"]),
        PYTHON_JULIAPKG_EXE=config["environments"]["julia_executable"],
        PYTHON_JULIAPKG_PROJECT=config["environments"]["julia_project"],
        # Dependencies must be prepared explicitly, never resolved during a run.
        PYTHON_JULIAPKG_OFFLINE="yes",
        MPLCONFIGDIR=str(root / "logs/matplotlib_cache"),
    )
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        env[key] = str(config["execution"]["blas_threads"])
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    records = []
    failed = []
    for step in steps:
        log = root / "logs" / ("driver_" + stamp + "_" + step["label"] + ".log")
        print("\nRUN", step["label"], flush=True)
        print("COMMAND", json.dumps(step["command"]), flush=True)
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                step["command"],
                cwd=str(root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                raise
        records.append(
            dict(
                label=step["label"],
                returncode=code,
                started_utc=started,
                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                command=step["command"],
                log=str(log.relative_to(root)),
            )
        )
        save(root / "logs" / ("driver_status_" + stamp + ".json"), dict(stage=stage, steps=records))
        if code:
            failed.append(step["label"])
            if not step["allow_failure"]:
                raise RuntimeError("Step failed: " + step["label"] + "; see " + str(log))
    if failed:
        raise RuntimeError("Archive checks failed: " + ", ".join(failed))
    save(
        root / "docs" / ("completed_" + stage + ".json"),
        dict(
            stage=stage,
            completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            steps=len(steps),
        ),
    )
    print("\nREPRODUCTION_COMPLETE", stage, str(root), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "config.json")
    parser.add_argument(
        "--stage",
        choices=["all", "env", "generator", "pilot", "main", "sampling", "joint", "post", "smoke"],
        default="all",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Create ignored config.local.json for this checkout",
    )
    args = parser.parse_args()
    if args.init_config:
        local = REPO / "config.local.json"
        if local.exists():
            parser.error("config.local.json already exists; edit it in place")
        config = json.loads((REPO / "config.json").read_text())
        config["paths"] = dict(repository_root=str(REPO), runs_root=str(REPO / "runs"))
        config["environments"]["julia_project"] = str(REPO / "environment/julia")
        save(local, config)
        print("Created", local, "; edit environment paths, then use --config config.local.json")
        return
    config_path = args.config.resolve()
    config = load_config(config_path)
    name = args.run_name or (
        "smoke_v1" if args.stage == "smoke" else config["execution"]["run_name"]
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        parser.error("Invalid run name; use ASCII letters, numbers, underscore or hyphen")
    run_root = Path(config["paths"]["runs_root"]).resolve() / name
    steps = build_plan(config, args.stage, config_path, run_root)
    if args.dry_run:
        print(json.dumps(dict(output_root=str(run_root), stage=args.stage, steps=steps), indent=2))
        return
    for key in ["regression_python", "analysis_python", "julia_executable"]:
        path = Path(config["environments"][key]).expanduser()
        if not path.is_file() or not os.access(str(path), os.X_OK):
            parser.error("Configured executable is unavailable: " + str(path))
    if not (Path(config["environments"]["julia_project"]) / "Manifest.toml").is_file():
        parser.error("Configured Julia project is missing Manifest.toml")
    with exclusive_lock(run_root):
        prepare_workspace(config, run_root, "smoke" if args.stage == "smoke" else "full")
        execute(steps, config, run_root, args.stage)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(1)
