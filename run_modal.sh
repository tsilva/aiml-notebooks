#!/usr/bin/env bash
set -euo pipefail

exec scripts/run_modal_notebook.py "$@" --gpu
