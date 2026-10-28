#!/usr/bin/env bash
# 10 bases, 2w/10s/10q, 10 extras/classe, seed 7. Use --dry-run para conferir.
# Treino/val/teste usam bancos separados, nunca dados de outro split.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
python_bin="${FEX_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        python_bin="$CONDA_PREFIX/bin/python"
    elif [[ -x "$script_dir/venv/bin/python" ]]; then
        python_bin="$script_dir/venv/bin/python"
    else
        python_bin=python3
    fi
fi
exec "$python_bin" experimentos_extras.py "$@"
