"""Record and check versions before starting scientific tasks."""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import sys
import subprocess
from .paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=["regression", "analysis"], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    expected = config["expected_versions"][args.role]
    actual = {name: metadata.version(name) for name in expected}
    mismatch = {
        name: dict(expected=expected[name], actual=actual[name])
        for name in expected
        if actual[name] != expected[name]
    }
    julia_manifest = Path(config["environments"]["julia_project"]) / "Manifest.toml"
    content = julia_manifest.read_text()
    julia = re.search(r'^julia_version\s*=\s*"([^"]+)"', content, re.M)
    backend = re.search(r"\[\[deps.SymbolicRegression\]\](.*?)(?=\n\[\[|\Z)", content, re.S)
    backend_version = (
        re.search(r'^version\s*=\s*"([^"]+)"', backend.group(1), re.M) if backend else None
    )
    if not julia or julia.group(1) != config["expected_versions"]["julia"]:
        mismatch["julia_manifest"] = "Julia version differs from configured expectation"
    if (
        not backend_version
        or backend_version.group(1) != config["expected_versions"]["symbolic_regression"]
    ):
        mismatch["symbolic_regression"] = "Backend version differs from configured expectation"
    julia_runtime = subprocess.run(
        [config["environments"]["julia_executable"], "--startup-file=no", "--version"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip()
    if julia_runtime != "julia version " + config["expected_versions"]["julia"]:
        mismatch["julia_runtime"] = julia_runtime
    report = dict(
        role=args.role,
        python=sys.version,
        executable=sys.executable,
        versions=actual,
        julia_project=str(julia_manifest.parent),
        julia_runtime=julia_runtime,
        mismatches=mismatch,
        passed=not mismatch,
    )
    (ROOT / "logs" / ("environment_" + args.role + ".json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
