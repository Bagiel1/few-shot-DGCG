#!/usr/bin/env bash
# Bateria pedida pelo professor para 5-way/1-shot/1-query.
#
# O episodio mestre sempre sorteia 10 queries por classe:
#   1 avaliada + pool fixo de 9 queries de contexto.
#
# Condicoes executadas por base e seed:
#   controle: 0 extra, executado uma unica vez;
#   25 e 45 extras em treino, treino+validacao, somente teste e todas as fases.
# Isso totaliza 9 execucoes por base/seed, 54 no plano padrao.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$script_dir"

python_bin="${FCQ_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        python_bin="$CONDA_PREFIX/bin/python"
    elif [[ -x "$script_dir/venv/bin/python" ]]; then
        python_bin="$script_dir/venv/bin/python"
    else
        python_bin=python3
    fi
fi

exec "$python_bin" experimentos_extras.py \
    --datasets flowers_micro cars_micro \
    --methods knn-union \
    --n-way 5 \
    --n-shot 1 \
    --n-query 1 \
    --extra-source queries \
    --extras-per-class 5 9 \
    --bank-per-class 9 \
    --phases train train-val test all \
    --modes propagation \
    --seeds 7 17 27 \
    --output-root runs/contexto_queries_bateria \
    "$@"
