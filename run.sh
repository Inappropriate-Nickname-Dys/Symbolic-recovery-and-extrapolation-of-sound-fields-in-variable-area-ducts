#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# The host's system Python may be old. Use it only to read the configured
# interpreter, then replace this process with Python 3.8+ from that environment.
exec "${ACOUSTIC_BOOTSTRAP_PYTHON:-python3}" -c '
import json, os, sys
args = sys.argv[1:]
config = "config.json"
for i, arg in enumerate(args):
    if arg == "--config" and i + 1 < len(args):
        config = args[i + 1]
    elif arg.startswith("--config="):
        config = arg.split("=", 1)[1]
with open(config, encoding="utf-8") as stream:
    interpreter = json.load(stream)["environments"]["regression_python"]
os.execv(interpreter, [interpreter, "reproduce.py"] + args)
' "$@"
