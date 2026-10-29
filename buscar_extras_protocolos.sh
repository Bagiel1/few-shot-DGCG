#!/usr/bin/env bash
# Grade default: 2/5 ways, 1/5/10 shots, 10 queries; 5/10/20 extras/classe.
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
exec "$python_bin" experimentos_extras_protocolos.py "$@"
