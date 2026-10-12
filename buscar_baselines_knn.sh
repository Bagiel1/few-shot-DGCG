#!/usr/bin/env bash

# Compara as baselines kNN-in, kNN-out e kNN-reciprocal em uma grade de
# n-way/n-shot/n-query. Mantem os demais hiperparametros nos defaults da main:
# k=5, distancia Euclidiana, arestas binarias, 2 saltos SGC e GRaNDe desligado.
# Nao usa nem altera qualquer limiar do DGCG.
#
# Uso:
#   ./buscar_baselines_knn.sh
#   ./buscar_baselines_knn.sh --quick
#   ./buscar_baselines_knn.sh --dry-run
#
# Personalizacao:
#   BKN_DATASETS="flowers_micro cars_micro resisc_micro"
#   BKN_N_WAYS="2 3 5"
#   BKN_N_SHOTS="1 5 10"
#   BKN_N_QUERIES="1 5 10"
#   BKN_PYTHON=./venv/bin/python
#   BKN_OUTPUT_ROOT=runs/minhas_baselines_knn
#   BKN_CACHE_DIR=data/openml
#   BKN_RESUME=0
#   BKN_FAIL_FAST=1

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir" || exit 1

if [[ -n "${BKN_PYTHON:-}" ]]; then
    python_bin="$BKN_PYTHON"
elif [[ -x "$script_dir/venv/bin/python" ]]; then
    python_bin="$script_dir/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
else
    python_bin="python"
fi

program="${BKN_PROTO_SCRIPT:-protosgc_fewshot.py}"
output_root="${BKN_OUTPUT_ROOT:-runs/busca_baselines_knn}"
cache_dir="${BKN_CACHE_DIR:-data/openml}"
resume="${BKN_RESUME:-1}"
fail_fast="${BKN_FAIL_FAST:-0}"
quick=0
dry_run=0

usage() {
    sed -n '3,24p' "${BASH_SOURCE[0]}"
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
    echo "Erro: BKN_RESUME deve ser 0 ou 1." >&2
    exit 2
fi
if [[ "$fail_fast" != "0" && "$fail_fast" != "1" ]]; then
    echo "Erro: BKN_FAIL_FAST deve ser 0 ou 1." >&2
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

if [[ -n "${BKN_DATASETS:-}" ]]; then
    read -r -a datasets <<< "$BKN_DATASETS"
else
    datasets=("${default_datasets[@]}")
fi
read -r -a n_ways <<< "${BKN_N_WAYS:-2 3 5}"
read -r -a n_shots <<< "${BKN_N_SHOTS:-1 5 10}"
read -r -a n_queries <<< "${BKN_N_QUERIES:-1 5 10}"

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

methods=(knn-in knn-out knn-reciprocal)
total_runs=$(( ${#methods[@]} * ${#datasets[@]} * ${#n_ways[@]} * ${#n_shots[@]} * ${#n_queries[@]} ))
run_number=0
failed_runs=0
skipped_runs=0

mkdir -p "$output_root" "$cache_dir"

echo "Busca das baselines kNN"
echo "  metodos   : ${methods[*]}"
echo "  datasets  : ${datasets[*]}"
echo "  n-way     : ${n_ways[*]}"
echo "  n-shot    : ${n_shots[*]}"
echo "  n-query   : ${n_queries[*]}"
echo "  execucoes : $total_runs"
echo "  saida     : $output_root"
echo "  kNN       : k=5, Euclidiano, binario (defaults)"
if (( quick )); then
    echo "  modo      : quick (200/50/200 episodios)"
else
    echo "  modo      : completo (defaults: 2000/200/1000 episodios)"
fi
echo

# Depois do primeiro download, usa diretamente o labels.csv e as imagens do
# cache. Isso evita chamar a API OpenML de novo em cada configuracao.
data_arguments() {
    local openml_id="$1"
    local dataset_cache="$cache_dir/org/openml/www/datasets/$openml_id"
    local -a labels_files=()
    if [[ -d "$dataset_cache" ]]; then
        mapfile -t labels_files < <(find "$dataset_cache" -mindepth 1 -maxdepth 2 -type f -name labels.csv -print)
    fi
    if (( ${#labels_files[@]} == 1 )); then
        printf '%s\0%s\0' --data-root "$(dirname -- "${labels_files[0]}")"
    else
        printf '%s\0%s\0%s\0%s\0' --openml-id "$openml_id" --cache-dir "$cache_dir"
    fi
}

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
                        echo "  ja concluido; pulando (BKN_RESUME=1)."
                        ((skipped_runs += 1))
                        continue
                    fi

                    data_args=()
                    while IFS= read -r -d '' argument; do
                        data_args+=("$argument")
                    done < <(data_arguments "$openml_id")

                    command=(
                        "$python_bin" "$program"
                        "${data_args[@]}"
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
methods = ("knn-in", "knn-out", "knn-reciprocal")
method_order = {method: index for index, method in enumerate(methods)}
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

fields = [
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
for method in methods:
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


def write_csv(path: Path, data: list[dict[str, object]], fieldnames: list[str] = fields) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


rows.sort(key=lambda row: (
    method_order[str(row["metodo"])], str(row["dataset"]),
    int(row["n_way"]), int(row["n_shot"]), int(row["n_query"]),
))
write_csv(root / "resultados.csv", rows)
valid = [row for row in rows if row["status"] == "ok"]

# Melhor shot/query dentro de cada n-way, onde a acuracia e comparavel.
by_way: dict[tuple[object, object, object], list[dict[str, object]]] = defaultdict(list)
for row in valid:
    by_way[(row["metodo"], row["dataset"], row["n_way"])].append(row)
best_by_way = [
    max(group, key=lambda row: (float(row["val_accuracy"]), -float(row["val_loss"])))
    for group in by_way.values()
]
best_by_way.sort(key=lambda row: (method_order[str(row["metodo"])], str(row["dataset"]), int(row["n_way"])))
write_csv(root / "melhores_por_nway.csv", best_by_way)

# Melhor protocolo por metodo/base, normalizando a vantagem do n-way menor.
by_dataset_method: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
for row in valid:
    by_dataset_method[(row["dataset"], row["metodo"])].append(row)
best_by_dataset_method = [
    max(group, key=lambda row: (float(row["val_acima_acaso"]), float(row["val_accuracy"])))
    for group in by_dataset_method.values()
]
best_by_dataset_method.sort(key=lambda row: (str(row["dataset"]), method_order[str(row["metodo"])]))
write_csv(root / "melhores_por_dataset_e_metodo.csv", best_by_dataset_method)

# Melhor baseline e protocolo de cada dataset.
by_dataset: dict[object, list[dict[str, object]]] = defaultdict(list)
for row in valid:
    by_dataset[row["dataset"]].append(row)
best_baseline_by_dataset = [
    max(group, key=lambda row: (float(row["val_acima_acaso"]), float(row["val_accuracy"])))
    for group in by_dataset.values()
]
best_baseline_by_dataset.sort(key=lambda row: str(row["dataset"]))
write_csv(root / "melhor_baseline_por_dataset.csv", best_baseline_by_dataset)

# Ranking medio de cada metodo/protocolo entre as bases concluidas.
by_protocol: dict[tuple[object, object, object, object], list[dict[str, object]]] = defaultdict(list)
for row in valid:
    by_protocol[(row["metodo"], row["n_way"], row["n_shot"], row["n_query"])].append(row)
ranking_fields = [
    "metodo", "n_way", "n_shot", "n_query", "datasets_concluidos",
    "media_val_acima_acaso", "media_val_accuracy", "desvio_val_accuracy",
    "media_test_accuracy",
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
        "desvio_val_accuracy": statistics.pstdev(float(row["val_accuracy"]) for row in group),
        "media_test_accuracy": statistics.fmean(float(row["test_accuracy"]) for row in group),
    })
ranking.sort(key=lambda row: (method_order[str(row["metodo"])], -float(row["media_val_acima_acaso"])))
write_csv(root / "ranking_medio.csv", ranking, ranking_fields)

best_mean_by_method: list[dict[str, object]] = []
for method in methods:
    candidates = [row for row in ranking if row["metodo"] == method]
    if not candidates:
        continue
    coverage = max(int(row["datasets_concluidos"]) for row in candidates)
    comparable = [row for row in candidates if int(row["datasets_concluidos"]) == coverage]
    best_mean_by_method.append(max(comparable, key=lambda row: float(row["media_val_acima_acaso"])))
write_csv(root / "melhores_medios_por_metodo.csv", best_mean_by_method, ranking_fields)

best_baseline_mean: list[dict[str, object]] = []
if ranking:
    coverage = max(int(row["datasets_concluidos"]) for row in ranking)
    comparable = [row for row in ranking if int(row["datasets_concluidos"]) == coverage]
    best_baseline_mean.append(max(comparable, key=lambda row: float(row["media_val_acima_acaso"])))
write_csv(root / "melhor_baseline_medio.csv", best_baseline_mean, ranking_fields)

print(f"Resultados agregados: {root / 'resultados.csv'}")
print(f"Melhor baseline por dataset: {root / 'melhor_baseline_por_dataset.csv'}")
print(f"Melhor baseline medio: {root / 'melhor_baseline_medio.csv'}")
PY
aggregate_status=$?

echo
echo "Busca encerrada: $((run_number - skipped_runs - failed_runs)) novas execucoes concluidas, $skipped_runs puladas, $failed_runs falhas."
echo "Resultados: $output_root/resultados.csv"
echo "Melhores por n-way: $output_root/melhores_por_nway.csv"
echo "Melhores por dataset e metodo: $output_root/melhores_por_dataset_e_metodo.csv"
echo "Melhor baseline por dataset: $output_root/melhor_baseline_por_dataset.csv"
echo "Ranking medio: $output_root/ranking_medio.csv"
echo "Melhor baseline medio: $output_root/melhor_baseline_medio.csv"

if (( aggregate_status != 0 || failed_runs != 0 )); then
    exit 1
fi
