"""Bounded parallel Linux processes; all log/checkpoint files stay in the project."""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .acoustic_core import ROOT, dump


def worker(job):
    label = job["label"]
    log = ROOT / "logs" / (label + ".log")
    cmd = [
        sys.executable,
        "-m",
        "acoustic_technique." + Path(job.get("script", "run_search.py")).stem,
    ]
    for key, value in job.items():
        if key in ["label", "script"]:
            continue
        cmd.extend(["--" + key.replace("_", "-"), str(value)])
    env = os.environ.copy()
    env.update(
        PYTHON_JULIACALL_THREADS=os.environ.get("PYTHON_JULIACALL_THREADS", "4"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLCONFIGDIR=str(ROOT / "logs" / "matplotlib_cache"),
        PYTHONUNBUFFERED="1",
    )
    started = time.time()
    with log.open("w") as stream:
        result = subprocess.run(
            cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)
        )
    record = dict(
        label=label, returncode=result.returncode, elapsed=time.time() - started, command=cmd
    )
    print(json.dumps(record), flush=True)
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = json.loads((ROOT / a.manifest).read_text())
    if not isinstance(jobs, list) or not all(isinstance(j, dict) and "label" in j for j in jobs):
        raise ValueError("Manifest must be a JSON array of job dictionaries with labels")
    print("DISPATCH", len(jobs), "jobs", a.workers, "workers", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(worker, j) for j in jobs]):
            records.append(f.result())
            dump(ROOT / "logs" / (Path(a.manifest).stem + "_status.json"), records)
    if any(r["returncode"] != 0 for r in records):
        sys.exit(1)
