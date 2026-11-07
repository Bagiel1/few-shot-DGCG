#!/usr/bin/env bash

# Executa, para uma base do Meta-Album:
#   1. Proto-SGC em todas as topologias, sem GRaNDe;
#   2. Proto-SGC em todas as topologias, com GRaNDe;
#   3. ProtoNet sem grafo.
#
# Ao final, extrai as acuracias e os intervalos de confianca dos logs e gera
# automaticamente uma tabela LaTeX com os resultados.
#
# Uso:
#   source rodar_base.sh
#   rodar_base 44289 98 49 49 cars_mini
#
# Tambem pode ser executado diretamente:
#   bash rodar_base.sh 44289 98 49 49 cars_mini
#
# Configuracoes opcionais por variaveis de ambiente:
#   RB_PROTO_SCRIPT=protosgc_fewshot.py  Caminho do programa principal.
#   RB_PYTHON=python                    Interpretador Python.
#   RB_N_WAY=5                         Numero de classes por episodio.
#   RB_N_SHOT=1                        Exemplos de suporte por classe.
#   RB_N_QUERY=5                       Consultas por classe.
#   RB_SEED=7                          Semente aleatoria.
#   RB_GRANDE_METRIC=cosine            Metrica do GRaNDe.
#   RB_QUICK=1                         Adiciona --quick às tres execucoes.

rodar_base() {
    if [[ "$#" -ne 5 ]]; then
        echo "Uso: rodar_base OPENML_ID TRAIN VAL TEST NOME" >&2
        echo "Exemplo: rodar_base 44289 98 49 49 cars_mini" >&2
        return 2
    fi

    local rb_openml_id="$1"
    local rb_train="$2"
    local rb_val="$3"
    local rb_test="$4"
    local rb_name="$5"

    local rb_script="${RB_PROTO_SCRIPT:-protosgc_fewshot.py}"
    local rb_python="${RB_PYTHON:-python}"
    #local rb_n_way="${RB_N_WAY:-5}"
    #local rb_n_shot="${RB_N_SHOT:-5}"
    #local rb_n_query="${RB_N_QUERY:-1}"
    local rb_n_way="5"
    local rb_n_shot="10"
    local rb_n_query="10"
    local rb_seed="${RB_SEED:-7}"
    local rb_grande_metric="${RB_GRANDE_METRIC:-cosine}"

    local rb_value
    for rb_value in \
        "$rb_openml_id" "$rb_train" "$rb_val" "$rb_test" \
        "$rb_n_way" "$rb_n_shot" "$rb_n_query" "$rb_seed"; do
        if [[ ! "$rb_value" =~ ^[0-9]+$ ]]; then
            echo "Erro: IDs, divisoes, protocolo e semente devem ser inteiros nao negativos." >&2
            return 2
        fi
    done

    if (( rb_openml_id < 1 || rb_train < 1 || rb_val < 1 || rb_test < 1 )); then
        echo "Erro: OPENML_ID e os tres tamanhos da divisao devem ser maiores que zero." >&2
        return 2
    fi

    if (( rb_n_way < 1 || rb_n_shot < 1 || rb_n_query < 1 )); then
        echo "Erro: RB_N_WAY, RB_N_SHOT e RB_N_QUERY devem ser maiores que zero." >&2
        return 2
    fi

    if [[ ! "$rb_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Erro: NOME pode conter apenas letras, numeros, ponto, hifen e sublinhado." >&2
        return 2
    fi

    if [[ "$rb_grande_metric" != "cosine" && \
          "$rb_grande_metric" != "euclidean" && \
          "$rb_grande_metric" != "rbo" ]]; then
        echo "Erro: RB_GRANDE_METRIC deve ser cosine, euclidean ou rbo." >&2
        return 2
    fi

    if ! command -v "$rb_python" >/dev/null 2>&1; then
        echo "Erro: interpretador nao encontrado: $rb_python" >&2
        return 127
    fi

    if [[ ! -f "$rb_script" ]]; then
        echo "Erro: programa principal nao encontrado: $rb_script" >&2
        echo "Defina o caminho com RB_PROTO_SCRIPT=/caminho/protosgc_fewshot.py" >&2
        return 2
    fi

    local rb_work_dir="runs/${rb_name}"
    local rb_timestamp
    rb_timestamp="$(date +%Y%m%d_%H%M%S)_$$"
    local rb_result_dir="${rb_work_dir}/resultados_${rb_timestamp}"
    local rb_no_grande_log="${rb_result_dir}/sem_grande.log"
    local rb_grande_log="${rb_result_dir}/com_grande_${rb_grande_metric}.log"
    local rb_protonet_log="${rb_result_dir}/protonet_sem_grafo.log"
    local rb_table="${rb_result_dir}/tabela_resultados_${rb_name}.tex"
    local rb_latest_table="${rb_work_dir}/tabela_resultados_${rb_name}.tex"

    if ! mkdir -p "$rb_result_dir"; then
        echo "Erro: nao foi possivel criar $rb_result_dir" >&2
        return 1
    fi

    local -a rb_common_args=(
        "$rb_script"
        --openml-id "$rb_openml_id"
        --class-split "$rb_train" "$rb_val" "$rb_test"
        --work-dir "$rb_work_dir"
        --n-way "$rb_n_way"
        --n-shot "$rb_n_shot"
        --n-query "$rb_n_query"
        --seed "$rb_seed"
    )

    local -a rb_optional_args=()
    if [[ "${RB_QUICK:-0}" == "1" ]]; then
        rb_optional_args+=(--quick)
    fi

    echo
    echo "============================================================"
    echo "Base: $rb_name | OpenML: $rb_openml_id"
    echo "Classes: $rb_train/$rb_val/$rb_test"
    echo "Protocolo: ${rb_n_way}-way/${rb_n_shot}-shot/${rb_n_query}-query"
    echo "Resultados: $rb_result_dir"
    echo "============================================================"

    echo
    echo "[1/3] Proto-SGC em todas as topologias, sem GRaNDe"
    if ! (
        set -o pipefail
        "$rb_python" "${rb_common_args[@]}" \
            --model proto-sgc \
            --no-grande \
            --graph-type all \
            "${rb_optional_args[@]}" 2>&1 | tee "$rb_no_grande_log"
    ); then
        echo "Erro na execucao sem GRaNDe. Log: $rb_no_grande_log" >&2
        return 1
    fi

    echo
    echo "[2/3] Proto-SGC em todas as topologias, com GRaNDe"
    if ! (
        set -o pipefail
        "$rb_python" "${rb_common_args[@]}" \
            --model proto-sgc \
            --grande \
            --graph-type all \
            --grande-metric "$rb_grande_metric" \
            "${rb_optional_args[@]}" 2>&1 | tee "$rb_grande_log"
    ); then
        echo "Erro na execucao com GRaNDe. Log: $rb_grande_log" >&2
        return 1
    fi

    echo
    echo "[3/3] ProtoNet sem grafo"
    if ! (
        set -o pipefail
        "$rb_python" "${rb_common_args[@]}" \
            --model protonet \
            "${rb_optional_args[@]}" 2>&1 | tee "$rb_protonet_log"
    ); then
        echo "Erro na execucao do ProtoNet. Log: $rb_protonet_log" >&2
        return 1
    fi

    echo
    echo "Gerando tabela LaTeX..."

    if ! "$rb_python" - \
        "$rb_no_grande_log" \
        "$rb_grande_log" \
        "$rb_protonet_log" \
        "$rb_table" \
        "$rb_name" \
        "$rb_openml_id" \
        "$rb_n_way" \
        "$rb_n_shot" \
        "$rb_n_query" \
        "$rb_grande_metric" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path


(
    no_grande_path,
    grande_path,
    protonet_path,
    output_path,
    dataset_name,
    openml_id,
    n_way,
    n_shot,
    n_query,
    grande_metric,
) = sys.argv[1:]


RESULT_PATTERN = re.compile(
    r"^\s*(?P<model>proto-sgc|protonet)/(?P<method>[^:]+):\s*"
    r"(?P<accuracy>\d+(?:\.\d+)?)%\s*\+/-\s*"
    r"(?P<ci95>\d+(?:\.\d+)?)%\s*\(IC 95%\)\s*\|\s*"
    r"loss=(?P<loss>\d+(?:\.\d+)?)\s*$"
)


def parse_results(path: str, expected_model: str) -> dict[str, tuple[float, float, float]]:
    results: dict[str, tuple[float, float, float]] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = RESULT_PATTERN.match(line)
        if match is None or match.group("model") != expected_model:
            continue
        results[match.group("method")] = (
            float(match.group("accuracy")),
            float(match.group("ci95")),
            float(match.group("loss")),
        )
    return results


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def format_result(
    result: tuple[float, float, float],
    *,
    underline: bool = False,
    bold: bool = False,
) -> str:
    accuracy, ci95, _ = result
    value = rf"{accuracy:.2f} \pm {ci95:.2f}"
    if underline:
        value = rf"\underline{{{value}}}"
    if bold:
        value = rf"\mathbf{{{value}}}"
    return f"${value}$"


graph_order = (
    "knn-union",
    "knn-out",
    "knn-in",
    "knn-reciprocal",
    "dgcg",
    "dgcg-plus",
)

graph_labels = {
    "knn-union": "kNN-union",
    "knn-out": "kNN-out",
    "knn-in": "kNN-in",
    "knn-reciprocal": "kNN-reciprocal",
    "dgcg": "DGCG",
    "dgcg-plus": "DGCG+",
}

metric_labels = {
    "cosine": "cosseno",
    "euclidean": "Euclidiana",
    "rbo": "RBO",
}

no_grande_raw = parse_results(no_grande_path, "proto-sgc")
grande_raw = parse_results(grande_path, "proto-sgc")
protonet_raw = parse_results(protonet_path, "protonet")

no_grande = {graph: no_grande_raw.get(graph) for graph in graph_order}
grande = {
    graph: grande_raw.get(f"{graph}+grande-{grande_metric}")
    for graph in graph_order
}
protonet = protonet_raw.get("sem-grafo")

missing = []
missing.extend(
    f"sem GRaNDe: {graph}" for graph, result in no_grande.items() if result is None
)
missing.extend(
    f"com GRaNDe: {graph}" for graph, result in grande.items() if result is None
)
if protonet is None:
    missing.append("ProtoNet: sem-grafo")

if missing:
    details = "\n  - ".join(missing)
    raise SystemExit(
        "Nao foi possivel localizar todos os resultados nos logs:\n  - " + details
    )

# Os testes acima garantem que nenhum dos valores abaixo e None.
no_grande = {key: value for key, value in no_grande.items() if value is not None}
grande = {key: value for key, value in grande.items() if value is not None}
assert protonet is not None

best_no_grande = max(result[0] for result in no_grande.values())
best_grande = max(result[0] for result in grande.values())
overall_best = max(
    [result[0] for result in no_grande.values()]
    + [result[0] for result in grande.values()]
    + [protonet[0]]
)

dataset_tex = latex_escape(dataset_name)
safe_label = re.sub(r"[^A-Za-z0-9]+", "_", dataset_name).strip("_").lower()
metric_tex = latex_escape(metric_labels.get(grande_metric, grande_metric))

lines = [
    r"% Tabela gerada automaticamente por rodar_base.sh.",
    r"% Requer \usepackage{booktabs} no preambulo.",
    r"\begin{table}[t]",
    r"    \centering",
    (
        rf"    \caption{{Resultados obtidos em \texttt{{{dataset_tex}}} "
        rf"(OpenML {openml_id}) para classificacao {n_way}-way/{n_shot}-shot. "
        r"Os valores apresentam a acuracia media e o intervalo de confianca "
        r"de 95\% nas classes nao observadas durante o meta-treino. O melhor "
        r"resultado geral e destacado em negrito, enquanto o melhor resultado "
        r"entre os metodos baseados em grafos e sublinhado.}"
    ),
    rf"    \label{{tab:resultados_{safe_label}}}",
    r"    \begin{tabular}{lcc}",
    r"        \toprule",
    (
        r"        \textbf{Metodo/Topologia} & \textbf{Sem GRaNDe} "
        rf"& \textbf{{Com GRaNDe ({metric_tex})}} \\"
    ),
    r"        \midrule",
]

for graph in graph_order:
    no_result = no_grande[graph]
    grande_result = grande[graph]
    no_cell = format_result(
        no_result,
        underline=abs(no_result[0] - best_no_grande) < 1e-12,
        bold=abs(no_result[0] - overall_best) < 1e-12,
    )
    grande_cell = format_result(
        grande_result,
        underline=abs(grande_result[0] - best_grande) < 1e-12,
        bold=abs(grande_result[0] - overall_best) < 1e-12,
    )
    lines.append(
        rf"        {graph_labels[graph]} & {no_cell} & {grande_cell} \\"
    )

protonet_cell = format_result(
    protonet,
    bold=abs(protonet[0] - overall_best) < 1e-12,
)

lines.extend(
    [
        r"        \midrule",
        (
            r"        ProtoNet (sem grafo) & \multicolumn{2}{c}{"
            + protonet_cell
            + r"} \\"
        ),
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
)

latex = "\n".join(lines) + "\n"
Path(output_path).write_text(latex, encoding="utf-8")

print()
print(latex, end="")
print(f"Tabela salva em: {output_path}")
PY
    then
        echo "Erro ao gerar a tabela LaTeX." >&2
        return 1
    fi

    if ! cp -- "$rb_table" "$rb_latest_table"; then
        echo "Erro ao atualizar a copia mais recente da tabela." >&2
        return 1
    fi

    echo
    echo "Execucao concluida."
    echo "Logs preservados em: $rb_result_dir"
    echo "Tabela desta execucao: $rb_table"
    echo "Tabela mais recente: $rb_latest_table"
}


if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    rodar_base "$@"
fi
