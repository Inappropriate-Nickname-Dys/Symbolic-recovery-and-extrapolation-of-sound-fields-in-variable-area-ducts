#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Default: legacy study + linear-noise study + latest unrestricted nonlinear
# configuration (see experiments/nonlinear_plan.json), including noise/cost.
# System Python only selects the configured Python 3.8+ interpreter.
exec "${ACOUSTIC_BOOTSTRAP_PYTHON:-python3}" -c '
import json, os, sys
args = sys.argv[1:]
config = "config.local.json" if os.path.isfile("config.local.json") else "config.json"
for i, arg in enumerate(args):
    if arg == "--config" and i + 1 < len(args):
        config = args[i + 1]
    elif arg.startswith("--config="):
        config = arg.split("=", 1)[1]
with open(config, encoding="utf-8") as stream:
    interpreter = os.path.expanduser(json.load(stream)["environments"]["regression_python"])
if not os.path.isfile(interpreter) or not os.access(interpreter, os.X_OK):
    raise SystemExit("Configured regression_python is unavailable. Run python3 reproduce.py --init-config, then edit config.local.json.")
os.execv(interpreter, [interpreter, "reproduce.py"] + args)
' "$@"
