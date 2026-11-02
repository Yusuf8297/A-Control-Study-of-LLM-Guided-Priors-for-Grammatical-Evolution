#!/usr/bin/env bash
# full campaign (resumable); pass a process count to override the default
PY=./.venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" run_fewshot_parallel.py "$@"
