#!/usr/bin/env bash

# Busca protocolos episodicos para Proto-SGC mantendo os hiperparametros da
# branch main. A grade inteira de knn-in e executada antes da grade de dgcg.
#
# Por padrao usa os dez datasets Micro do conjunto 0 do Meta-Album. Todos tem
# 20 classes e 40 imagens por classe; a divisao 10/5/5 e mantida fixa para que
# variar n-way nao altere tambem as classes disponiveis em treino/val/teste.
#
# Uso:
#   bash buscar_protocolos_default.sh
#   bash buscar_protocolos_default.sh --quick
#   bash buscar_protocolos_default.sh --dry-run
#
# Personalizacao por variaveis de ambiente:
#   BP_DATASETS="flowers_micro cars_micro resisc_micro"
#   BP_N_WAYS="2 3 5"
#   BP_N_SHOTS="1 5 10"
#   BP_N_QUERIES="1 5 10"
#   BP_PYTHON=./venv/bin/python
#   BP_OUTPUT_ROOT=runs/minha_busca
#   BP_RESUME=0
#   BP_FAIL_FAST=1

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir" || exit 1

if [[ -n "${BP_PYTHON:-}" ]]; then
    python_bin="$BP_PYTHON"
elif [[ -x "$script_dir/venv/bin/python" ]]; then
    python_bin="$script_dir/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
else
    python_bin="python"
fi
program="${BP_PROTO_SCRIPT:-protosgc_fewshot.py}"
output_root="${BP_OUTPUT_ROOT:-runs/busca_protocolos_default}"
resume="${BP_RESUME:-1}"
fail_fast="${BP_FAIL_FAST:-0}"
quick=0
dry_run=0

usage() {
    sed -n '3,23p' "${BASH_SOURCE[0]}"
}

while (( $# > 0 )); do
    case "$1" in
        --quick)
            quick=1
            ;;
        --dry-run)
            dry_run=1
            ;;
        --no-resume)
            resume=0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Erro: opcao desconhecida: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "$resume" != "0" && "$resume" != "1" ]]; then
    echo "Erro: BP_RESUME deve ser 0 ou 1." >&2
    exit 2
fi
if [[ "$fail_fast" != "0" && "$fail_fast" != "1" ]]; then
    echo "Erro: BP_FAIL_FAST deve ser 0 ou 1." >&2
    exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Erro: interpretador nao encontrado: $python_bin" >&2
    exit 127
fi
if [[ ! -f "$program" ]]; then
    echo "Erro: programa principal nao encontrado: $program" >&2
    exit 2
fi

# Meta-Album Micro, set 0. Cada entrada possui 20 classes x 40 imagens.
declare -A dataset_id=(
    [bacteria_micro]=44237
    [plankton_micro]=44238
    [flowers_micro]=44239
    [sports_micro]=44240
    [birds_micro]=44241
    [plant_village_micro]=44242
    [textures_micro]=44243
    [manufacturing_micro]=44244
    [cars_micro]=44245
    [resisc_micro]=44246
)

default_datasets=(
    bacteria_micro
    plankton_micro
    flowers_micro
    sports_micro
    birds_micro
    plant_village_micro
    textures_micro
    manufacturing_micro
    cars_micro
    resisc_micro
)

if [[ -n "${BP_DATASETS:-}" ]]; then
    read -r -a datasets <<< "$BP_DATASETS"
else
    datasets=("${default_datasets[@]}")
fi
read -r -a n_ways <<< "${BP_N_WAYS:-2 3 5}"
read -r -a n_shots <<< "${BP_N_SHOTS:-1 5 10}"
read -r -a n_queries <<< "${BP_N_QUERIES:-1 5 10}"

if (( ${#datasets[@]} == 0 || ${#n_ways[@]} == 0 || ${#n_shots[@]} == 0 || ${#n_queries[@]} == 0 )); then
    echo "Erro: datasets e grades de n-way/n-shot/n-query nao podem ser vazios." >&2
    exit 2
fi

for dataset in "${datasets[@]}"; do
    if [[ -z "${dataset_id[$dataset]+x}" ]]; then
        echo "Erro: dataset desconhecido: $dataset" >&2
        echo "Disponiveis: ${default_datasets[*]}" >&2
        exit 2
    fi
done

for value in "${n_ways[@]}" "${n_shots[@]}" "${n_queries[@]}"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Erro: valores da grade devem ser inteiros positivos; recebido: $value" >&2
        exit 2
    fi
done
for way in "${n_ways[@]}"; do
    if (( way > 5 )); then
        echo "Erro: n-way=$way excede as 5 classes fixas de validacao/teste." >&2
        exit 2
    fi
done
for shot in "${n_shots[@]}"; do
    for query in "${n_queries[@]}"; do
        if (( shot + query > 40 )); then
            echo "Erro: n-shot+n-query deve ser <= 40 nos datasets Micro; recebido: $shot+$query." >&2
            exit 2
        fi
    done
done

methods=(knn-in dgcg)
total_runs=$(( ${#methods[@]} * ${#datasets[@]} * ${#n_ways[@]} * ${#n_shots[@]} * ${#n_queries[@]} ))
run_number=0
failed_runs=0
skipped_runs=0

mkdir -p "$output_root"

echo "Busca de protocolos default"
echo "  datasets : ${datasets[*]}"
echo "  n-way    : ${n_ways[*]}"
echo "  n-shot   : ${n_shots[*]}"
echo "  n-query  : ${n_queries[*]}"
echo "  ordem    : knn-in, depois dgcg"
echo "  execucoes: $total_runs"
echo "  saida    : $output_root"
if (( quick )); then
    echo "  modo     : quick (200/50/200 episodios)"
else
    echo "  modo     : completo (defaults: 2000/200/1000 episodios)"
fi
echo

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        openml_id="${dataset_id[$dataset]}"
        shared_work_dir="$output_root/_work/$dataset"

        for way in "${n_ways[@]}"; do
            for shot in "${n_shots[@]}"; do
                for query in "${n_queries[@]}"; do
                    ((run_number += 1))
                    protocol="${way}way_${shot}shot_${query}query"
                    run_dir="$output_root/$method/$dataset/$protocol"
                    log_path="$run_dir/run.log"
                    ok_path="$run_dir/status.ok"
                    checkpoint_copy="$run_dir/checkpoint.pt"
                    mkdir -p "$run_dir" "$shared_work_dir"

                    printf '[%d/%d] %s | %s | %s\n' \
                        "$run_number" "$total_runs" "$method" "$dataset" "$protocol"

                    if (( resume )) && [[ -f "$ok_path" ]]; then
                        echo "  ja concluido; pulando (BP_RESUME=1)."
                        ((skipped_runs += 1))
                        continue
                    fi

                    command=(
                        "$python_bin" "$program"
                        --openml-id "$openml_id"
                        --class-split 10 5 5
                        --work-dir "$shared_work_dir"
                        --n-way "$way"
                        --n-shot "$shot"
                        --n-query "$query"
                        --graph-type "$method"
                    )
                    if (( quick )); then
                        command+=(--quick)
                    fi

                    if (( dry_run )); then
                        printf ' '
                        printf '%q ' "${command[@]}"
                        printf '\n'
                        continue
                    fi

                    rm -f "$ok_path"
                    {
                        printf 'Comando:'
                        printf ' %q' "${command[@]}"
                        printf '\n'
                    } > "$run_dir/command.txt"

                    "${command[@]}" 2>&1 | tee "$log_path"
                    run_status=${PIPESTATUS[0]}
                    if (( run_status != 0 )); then
                        echo "  FALHOU (codigo $run_status): $log_path" >&2
                        ((failed_runs += 1))
                        if (( fail_fast )); then
                            break 5
                        fi
                        continue
                    fi

                    : > "$ok_path"
                    checkpoint_path="$({ sed -n 's/^[[:space:]]*checkpoint: //p' "$log_path" || true; } | tail -n 1)"
                    if [[ -n "$checkpoint_path" && -f "$checkpoint_path" ]]; then
                        cp -f -- "$checkpoint_path" "$checkpoint_copy"
                    else
                        echo "  Aviso: checkpoint nao localizado no log." >&2
                    fi
                done
            done
        done
    done
done

if (( dry_run )); then
    echo
    echo "Dry-run concluido; nenhuma execucao foi iniciada."
    exit 0
fi

"$python_bin" - "$output_root" <<'PY'
from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


root = Path(sys.argv[1])
validation_pattern = re.compile(
    r"Ep\.\s+(?P<episode>\d+)\s+\|\s+validacao:\s+"
    r"(?P<accuracy>[\d.]+)%\s+\+/-\s+(?P<ci95>[\d.]+)%\s+\|\s+"
    r"loss=(?P<loss>[\d.]+)"
)
test_pattern = re.compile(
    r"^\s*proto-sgc/(?P<label>[^:]+):\s+"
    r"(?P<accuracy>[\d.]+)%\s+\+/-\s+(?P<ci95>[\d.]+)%\s+"
    r"\(IC 95%\)\s+\|\s+loss=(?P<loss>[\d.]+)\s*$",
    re.MULTILINE,
)
protocol_pattern = re.compile(r"(?P<way>\d+)way_(?P<shot>\d+)shot_(?P<query>\d+)query")

fieldnames = [
    "metodo", "dataset", "openml_id", "n_way", "n_shot", "n_query",
    "val_accuracy", "val_ci95", "val_loss", "val_acima_acaso",
    "test_accuracy", "test_ci95", "test_loss", "status", "log", "checkpoint",
]

ids = {
    "bacteria_micro": 44237, "plankton_micro": 44238, "flowers_micro": 44239,
    "sports_micro": 44240, "birds_micro": 44241, "plant_village_micro": 44242,
    "textures_micro": 44243, "manufacturing_micro": 44244, "cars_micro": 44245,
    "resisc_micro": 44246,
}

rows: list[dict[str, object]] = []
for method in ("knn-in", "dgcg"):
    method_root = root / method
    if not method_root.exists():
        continue
    for log_path in sorted(method_root.glob("*/*/run.log")):
        dataset = log_path.parents[1].name
        protocol = protocol_pattern.fullmatch(log_path.parent.name)
        if protocol is None:
            continue
        way = int(protocol.group("way"))
        text = log_path.read_text(encoding="utf-8", errors="replace")
        validations = list(validation_pattern.finditer(text))
        # O treino conserva o primeiro checkpoint em caso de empate estrito.
        best_val = max(validations, key=lambda item: float(item.group("accuracy"))) if validations else None
        tests = list(test_pattern.finditer(text))
        test = tests[-1] if tests else None
        successful = (log_path.parent / "status.ok").is_file() and best_val is not None and test is not None
        val_accuracy = float(best_val.group("accuracy")) if best_val else None
        chance = 100.0 / way
        normalized = ((val_accuracy - chance) / (100.0 - chance)) if val_accuracy is not None else None
        rows.append({
            "metodo": method,
            "dataset": dataset,
            "openml_id": ids.get(dataset, ""),
            "n_way": way,
            "n_shot": int(protocol.group("shot")),
            "n_query": int(protocol.group("query")),
            "val_accuracy": val_accuracy,
            "val_ci95": float(best_val.group("ci95")) if best_val else None,
            "val_loss": float(best_val.group("loss")) if best_val else None,
            "val_acima_acaso": normalized,
            "test_accuracy": float(test.group("accuracy")) if test else None,
            "test_ci95": float(test.group("ci95")) if test else None,
            "test_loss": float(test.group("loss")) if test else None,
            "status": "ok" if successful else "incompleto",
            "log": str(log_path),
            "checkpoint": str(log_path.parent / "checkpoint.pt") if (log_path.parent / "checkpoint.pt").is_file() else "",
        })

def write_csv(path: Path, data: list[dict[str, object]], fields: list[str] = fieldnames) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


rows.sort(key=lambda row: (
    0 if row["metodo"] == "knn-in" else 1,
    str(row["dataset"]), int(row["n_way"]), int(row["n_shot"]), int(row["n_query"]),
))
write_csv(root / "resultados.csv", rows)

valid = [row for row in rows if row["status"] == "ok"]

# Dentro de um mesmo n-way, acuracias sao diretamente comparaveis.
by_way: dict[tuple[object, object, object], list[dict[str, object]]] = defaultdict(list)
for row in valid:
    by_way[(row["metodo"], row["dataset"], row["n_way"])].append(row)
best_by_way = [
    max(group, key=lambda row: (float(row["val_accuracy"]), -float(row["val_loss"])))
    for group in by_way.values()
]
best_by_way.sort(key=lambda row: (str(row["metodo"]), str(row["dataset"]), int(row["n_way"])))
write_csv(root / "melhores_por_nway.csv", best_by_way)

# Para comparar n-way diferentes, remove a vantagem mecanica do nivel de acaso.
by_dataset: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
for row in valid:
    by_dataset[(row["metodo"], row["dataset"])].append(row)
best_overall = [
    max(group, key=lambda row: (float(row["val_acima_acaso"]), float(row["val_accuracy"])))
    for group in by_dataset.values()
]
best_overall.sort(key=lambda row: (str(row["metodo"]), str(row["dataset"])))
write_csv(root / "melhores_por_dataset.csv", best_overall)

# Ranking de protocolos pela media de validacao nos datasets concluidos.
by_protocol: dict[tuple[object, object, object, object], list[dict[str, object]]] = defaultdict(list)
for row in valid:
    by_protocol[(row["metodo"], row["n_way"], row["n_shot"], row["n_query"])].append(row)
ranking_fields = [
    "metodo", "n_way", "n_shot", "n_query", "datasets_concluidos",
    "media_val_acima_acaso", "media_val_accuracy", "media_test_accuracy",
]
ranking: list[dict[str, object]] = []
for (method, way, shot, query), group in by_protocol.items():
    ranking.append({
        "metodo": method,
        "n_way": way,
        "n_shot": shot,
        "n_query": query,
        "datasets_concluidos": len(group),
        "media_val_acima_acaso": statistics.fmean(float(row["val_acima_acaso"]) for row in group),
        "media_val_accuracy": statistics.fmean(float(row["val_accuracy"]) for row in group),
        "media_test_accuracy": statistics.fmean(float(row["test_accuracy"]) for row in group),
    })
ranking.sort(key=lambda row: (str(row["metodo"]), -float(row["media_val_acima_acaso"])))
write_csv(root / "ranking_medio.csv", ranking, ranking_fields)

best_mean: list[dict[str, object]] = []
for method in ("knn-in", "dgcg"):
    candidates = [row for row in ranking if row["metodo"] == method]
    if candidates:
        # Nao deixa um protocolo avaliado em poucos datasets vencer outro com
        # cobertura completa apenas porque os demais runs ainda nao terminaram.
        max_coverage = max(int(row["datasets_concluidos"]) for row in candidates)
        comparable = [row for row in candidates if int(row["datasets_concluidos"]) == max_coverage]
        best_mean.append(max(comparable, key=lambda row: float(row["media_val_acima_acaso"])))
write_csv(root / "melhores_medios.csv", best_mean, ranking_fields)

print(f"Resultados agregados: {root / 'resultados.csv'}")
print(f"Melhores por dataset: {root / 'melhores_por_dataset.csv'}")
print(f"Melhores medios: {root / 'melhores_medios.csv'}")
PY
aggregate_status=$?

echo
echo "Busca encerrada: $((run_number - skipped_runs - failed_runs)) novas execucoes concluidas, $skipped_runs puladas, $failed_runs falhas."
echo "Resultados: $output_root/resultados.csv"
echo "Melhores por n-way: $output_root/melhores_por_nway.csv"
echo "Melhores por dataset: $output_root/melhores_por_dataset.csv"
echo "Ranking medio: $output_root/ranking_medio.csv"

if (( aggregate_status != 0 || failed_runs != 0 )); then
    exit 1
fi
