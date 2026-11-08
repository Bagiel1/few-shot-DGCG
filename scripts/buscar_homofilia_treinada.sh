#!/usr/bin/env bash
# Mesmo protocolo do PDF, agora usando a projecao treinada.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$script_dir"
python_bin="${FSC_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        python_bin="$CONDA_PREFIX/bin/python"
    elif [[ -x "$script_dir/venv/bin/python" ]]; then
        python_bin="$script_dir/venv/bin/python"
    else
        python_bin=python3
    fi
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$python_bin" experimento_homofilia_treinado.py "$@"
