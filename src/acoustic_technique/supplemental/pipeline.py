"""Run linear-noise and latest enhanced nonlinear studies inside the run workspace."""

import argparse
import concurrent.futures
import os
import signal
import subprocess
import sys
import time

from .common import CONFIG, ROOT, REPOSITORY_ROOT, SETTINGS, dump, json, np, run_folder
from .data import check_reference
from .enhanced_metrics import assess

MODULE = "acoustic_technique.supplemental."
PLAN = json.loads((REPOSITORY_ROOT / "experiments/nonlinear_plan.json").read_text())
OPTIONS = SETTINGS.get("supplemental", {})
CPUS = sorted(os.sched_getaffinity(0))
THREADS = CONFIG["julia_threads"]
WORKERS = min(OPTIONS.get("workers", 8), max(1, len(CPUS) // THREADS))
ACTIVE = {}
STOPPING = False


def status(stage, **extra):
    dump(ROOT / "status.json", dict(stage=stage, updated_unix=time.time(), **extra))


def make_job(phase, label, seed, budget, overrides, snr="clean", kind=None, group=None):
    simple = kind is not None
    base = 71010 if simple else 900000
    if phase == "simple_noise":
        base = (
            70000
            + ["uniform", "conical", "exponential", "catenoidal"].index(kind) * 1000
            + group * 10
        )
    noise_seed = base + (0 if snr == "clean" else int(snr) * 10000) + seed
    return dict(
        run_id="%s_snr%s_seed%d" % (label, snr, seed),
        scenario="simple_" + kind if simple else "nonlinear_burgers",
        kind=kind,
        parameter_group=group,
        phase=phase,
        pysr_seed=seed,
        noise_seed=noise_seed,
        data_seed=CONFIG["data_seed"],
        snr_db=snr,
        budget_seconds=budget,
        overrides=overrides,
        configuration_label=label,
    )


def job_file(job):
    p = ROOT / "jobs" / (job["phase"] + "__" + job["run_id"] + ".json")
    dump(p, job)
    return p


def execute(job, slot, deadline=None, evaluate_harmonics=True):
    folder = run_folder(job)
    folder.mkdir(parents=True, exist_ok=True)
    if STOPPING:
        raise RuntimeError("Pipeline interrupted before job launch")
    p = job_file(job)
    if (folder / "result.json").exists():
        if json.loads((folder / "result.json").read_text())["job"] != job:
            raise ValueError("Existing job differs; choose a new --run-name: " + str(folder))
        if (
            evaluate_harmonics
            and job["scenario"] == "nonlinear_burgers"
            and not (folder / "enhanced_metrics.json").exists()
        ):
            assess(folder)
        return folder
    affinity = CPUS[slot * THREADS : (slot + 1) * THREADS]
    if len(affinity) != THREADS:
        raise ValueError("Not enough CPUs for configured Julia threads")
    command = [
        "taskset",
        "-c",
        ",".join(map(str, affinity)),
        sys.executable,
        "-m",
        MODULE + "search",
        str(p),
    ]
    started = time.time()
    timeout = job["budget_seconds"] + 1200
    if deadline is not None:
        timeout = min(timeout, max(1, deadline - started))
    with (folder / "process.log").open("wb") as log:
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        ACTIVE[process.pid] = process
        if STOPPING:
            os.killpg(process.pid, signal.SIGTERM)
        dump(
            folder / "process.json",
            dict(pid=process.pid, cpus=affinity, start_unix=started, command=command),
        )
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = -9
        finally:
            ACTIVE.pop(process.pid, None)
    dump(
        folder / "process_exit.json",
        dict(returncode=code, wall_seconds=time.time() - started, deadline_stopped=code == -9),
    )
    print(json.dumps(dict(phase=job["phase"], run_id=job["run_id"], returncode=code)), flush=True)
    if code:
        if deadline is not None and time.time() >= deadline:
            return folder
        raise RuntimeError("Search failed; see " + str(folder / "process.log"))
    if evaluate_harmonics and job["scenario"] == "nonlinear_burgers":
        assess(folder)
    return folder


def wave(jobs, deadline=None, evaluate_harmonics=True, workers=None):
    workers = min(workers or WORKERS, WORKERS)
    for offset in range(0, len(jobs), workers):
        if deadline is not None and time.time() >= deadline:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            fs = [
                pool.submit(execute, job, slot, deadline, evaluate_harmonics)
                for slot, job in enumerate(jobs[offset : offset + workers])
            ]
            for f in concurrent.futures.as_completed(fs):
                f.result()


def records(phase):
    rows = []
    for p in sorted((ROOT / "supplemental" / phase).glob("*/result.json")):
        r = json.loads(p.read_text())
        e = p.with_name("enhanced_metrics.json")
        rows.append((r, json.loads(e.read_text()) if e.exists() else {}))
    return rows


def confirm(label, overrides, budget, phase, deadline):
    jobs = [make_job(phase, label, s, budget, overrides) for s in range(100, 105)]
    dump(ROOT / "manifests" / (phase + ".json"), jobs)
    status(phase)
    wave(jobs, deadline)
    passed = sum(e.get("basic_reached", False) for _, e in records(phase))
    return passed >= 3 and len(records(phase)) == 5


def discover(profile, deadline):
    if profile == "final":
        c = PLAN["final_configuration"]
        if confirm(c["label"], c["overrides"], c["budget"], "confirmation_final", deadline):
            return dict(c, confirmation_phase="confirmation_final")
        return None
    for number, rnd in enumerate(PLAN["rounds"]):
        if time.time() >= deadline:
            break
        phase = "exploration_r%d" % number
        jobs = [
            make_job(phase, c["label"], s, rnd["budget"], c["overrides"])
            for c in rnd["configs"]
            for s in rnd["seeds"]
        ]
        dump(ROOT / "manifests" / (phase + ".json"), jobs)
        status(phase)
        wave(jobs, deadline)
        groups = {}
        for r, e in records(phase):
            groups.setdefault(r["job"]["run_id"].split("_snr")[0], []).append((r, e))
        if not groups:
            continue
        label = min(
            groups, key=lambda k: np.median([r["selected"]["validation_mse"] for r, _ in groups[k]])
        )
        overrides = groups[label][0][0]["job"]["overrides"]
        dump(
            ROOT / "selection" / (phase + ".json"),
            dict(label=label, rule="lowest median ID validation MSE", overrides=overrides),
        )
        if any(e.get("basic_reached", False) for _, e in groups[label]):
            phase = "confirmation_r%d" % number
            if confirm(label, overrides, rnd["budget"], phase, deadline):
                return dict(
                    label=label, overrides=overrides, budget=rnd["budget"], confirmation_phase=phase
                )
    return None


def simple_noise():
    jobs = []
    for kind in ["uniform", "conical", "exponential", "catenoidal"]:
        for group in range(3):
            for s in range(5):
                jobs.append(
                    make_job("simple_noise", "%s_g%d" % (kind, group), s, 180, {}, 40, kind, group)
                )
    for kind in ["conical", "catenoidal"]:
        for snr in [50, 30]:
            for s in range(5):
                jobs.append(make_job("simple_noise", kind + "_g1", s, 180, {}, snr, kind, 1))
    dump(ROOT / "manifests/simple_noise.json", jobs)
    status("simple_noise")
    wave(jobs)


def nonlinear(profile):
    start_file = ROOT / "provenance/nonlinear_start.json"
    if start_file.exists():
        run_start = json.loads(start_file.read_text())
        if run_start["profile"] != profile:
            raise ValueError("Choose a new run name for a different nonlinear profile")
    else:
        started = time.time()
        limit = float(OPTIONS.get("basic_time_limit_seconds", PLAN["basic_time_limit_seconds"]))
        run_start = dict(
            start_unix=started,
            deadline_unix=started + limit,
            profile=profile,
            workers=WORKERS,
            threads=THREADS,
            available_cpus=CPUS,
        )
        dump(start_file, run_start)
    winner_path = ROOT / "selection/confirmed_configuration.json"
    winner = (
        json.loads(winner_path.read_text())
        if winner_path.exists()
        else discover(profile, run_start["deadline_unix"])
    )
    if winner is None:
        return False
    dump(winner_path, winner)
    jobs = [
        make_job(
            "noise_performance", winner["label"], s, winner["budget"], winner["overrides"], snr
        )
        for snr in [50, 40, 30]
        for s in range(5)
    ]
    dump(ROOT / "manifests/noise_performance.json", jobs)
    status("noise_performance")
    wave(jobs)
    status("independent_efficiency")
    for s in range(200, 205):
        jobs = []
        for label, snr, simple in [
            ("A", "clean", True),
            ("B", 40, True),
            ("C", "clean", False),
            ("D", 40, False),
        ]:
            job = make_job(
                "efficiency",
                label,
                s,
                winner["budget"],
                winner["overrides"],
                snr,
                "catenoidal" if simple else None,
                1 if simple else None,
            )
            job.update(efficiency_group=label, replicate=s - 200)
            jobs.append(job)
            job["configuration_label"] = winner["label"]
        dump(ROOT / "manifests" / ("efficiency_seed%d.json" % s), jobs)
        wave(jobs, evaluate_harmonics=False, workers=4)
        for job in jobs:
            if job["scenario"] == "nonlinear_burgers":
                assess(run_folder(job))
    return True


def baseline(job, slot):
    folder = run_folder(job) / "baselines"
    if (folder / "result.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    affinity = [CPUS[min(32 + slot, len(CPUS) - 1)]]
    command = [
        "taskset",
        "-c",
        str(affinity[0]),
        sys.executable,
        "-m",
        MODULE + "baselines",
        str(job_file(job)),
    ]
    with (folder / "process.log").open("wb") as f:
        subprocess.run(
            command, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, check=True, timeout=3600
        )


def finish_baselines():
    # Outside the isolated cost waves. Every method reads identical saved observations.
    jobs = []
    for p in sorted((ROOT / "jobs").glob("*.json")):
        j = json.loads(p.read_text())
        if j["phase"] in ["simple_noise", "noise_performance"] or j["phase"].startswith(
            "confirmation"
        ):
            if (run_folder(j) / "result.json").exists():
                jobs.append(j)
    workers = min(4, len(CPUS))
    for offset in range(0, len(jobs), workers):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            fs = [
                pool.submit(baseline, j, i) for i, j in enumerate(jobs[offset : offset + workers])
            ]
            for f in concurrent.futures.as_completed(fs):
                f.result()


def finalize():
    if records("efficiency"):
        subprocess.run([sys.executable, "-m", MODULE + "analyze_cost"], check=True)
    from .summarize import generate

    generate()
    subprocess.run([SETTINGS["environments"]["analysis_python"], "-m", MODULE + "plot"], check=True)


def main():
    global STOPPING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["all", "simple", "nonlinear", "adaptive", "smoke", "post"], default="all"
    )
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)

    def interrupt(signum, frame):
        global STOPPING
        STOPPING = True
        for process in list(ACTIVE.values()):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise KeyboardInterrupt("Supplemental pipeline interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    error = None
    success = False
    try:
        if args.mode == "post":
            finalize()
            success = True
            return
        dump(
            ROOT / "provenance/protocol.json",
            dict(
                plan=PLAN,
                settings=OPTIONS,
                profile=args.mode,
                note="Final profile repeats the frozen 3600s configuration. Adaptive profile repeats capacity exploration. Harmonics are post-selection checks only.",
            ),
        )
        check_reference()
        if args.mode == "smoke":
            c = PLAN["final_configuration"]
            wave(
                [
                    make_job(
                        "compatibility",
                        "enhanced_smoke",
                        9,
                        OPTIONS.get("smoke_budget_seconds", 30),
                        c["overrides"],
                    )
                ]
            )
            success = True
        else:
            if args.mode in ["all", "simple"]:
                simple_noise()
            success = (
                nonlinear("adaptive" if args.mode == "adaptive" else "final")
                if args.mode in ["all", "nonlinear", "adaptive"]
                else True
            )
            finish_baselines()
        dump(
            ROOT / "controller_finished.json",
            dict(
                success=success,
                finished_unix=time.time(),
                note="Scientific basic-target failure is distinct from process errors. Noise and cost are gated on clean confirmation.",
            ),
        )
        finalize()
    except (Exception, KeyboardInterrupt):
        import traceback

        error = traceback.format_exc()
        dump(ROOT / "controller_error.json", dict(error=error))
        try:
            from .summarize import generate

            generate()
        except Exception:
            pass  # Preserve the original failure and raw logs.
        raise
    finally:
        status("success" if success and error is None else "failed", error=error)
    if not success:
        raise SystemExit(
            "Clean confirmation did not reach 3/5; dependent noise/cost tasks were not started. See saved results."
        )


if __name__ == "__main__":
    main()
