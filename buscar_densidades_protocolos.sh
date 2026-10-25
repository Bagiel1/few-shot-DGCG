#!/usr/bin/env bash
# Busca DGCG por densidade x way/shot/query, separada da bateria de graus.
# Uso: bash buscar_densidades_protocolos.sh --dry-run
#      bash buscar_densidades_protocolos.sh --device cuda
#      bash buscar_densidades_protocolos.sh --datasets flowers_micro --quick
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
python_bin="${FDP_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        python_bin="$CONDA_PREFIX/bin/python"
    elif [[ -x "$script_dir/venv/bin/python" ]]; then
        python_bin="$script_dir/venv/bin/python"
    else
        python_bin=python3
    fi
fi
exec "$python_bin" experimentos_densidades_protocolos.py "$@"
