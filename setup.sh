#!/usr/bin/env bash
# one-off setup: virtual environment + dependencies
set -e
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
echo
echo "done. quick check:  ./.venv/bin/python run_fewshot_parallel.py 2 --smoke"
