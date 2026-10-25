#!/usr/bin/env bash
# Bateria separada: metricas, no-graph, kNNs e intervalos DGCG x protocolos.
# Uso: bash buscar_metricas_graus.sh --dry-run
#      bash buscar_metricas_graus.sh --datasets flowers_micro --quick
#      bash buscar_metricas_graus.sh --seeds 7 17 27
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
python_bin="${FMG_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        python_bin="$CONDA_PREFIX/bin/python"
    elif [[ -x "$script_dir/venv/bin/python" ]]; then
        python_bin="$script_dir/venv/bin/python"
    else
        python_bin=python3
    fi
fi
exec "$python_bin" experimentos_metricas_graus.py "$@"
