#!/usr/bin/env bash

# Busca o melhor alvo de grau medio e de densidade do DGCG, sem alterar os
# demais defaults do projeto. Executa primeiro a grade de grau e depois a de
# densidade, sempre selecionando configuracoes pela validacao.
#
# Protocolo default: 5-way/1-shot/5-query, 30 nos e 29 candidatos por no.
# Nesse caso densidade rho corresponde ao grau medio alvo rho * 29.
#
# Uso:
#   ./buscar_dgcg_grau_densidade.sh
#   ./buscar_dgcg_grau_densidade.sh --quick
#   ./buscar_dgcg_grau_densidade.sh --dry-run
#
# Personalizacao:
#   DGD_DATASETS="flowers_micro cars_micro resisc_micro"
#   DGD_N_WAY=5 DGD_N_SHOT=1 DGD_N_QUERY=5
#   DGD_DEGREES="1 2 3 4 5 4:6 8 10 15 20"
#   DGD_DENSITIES="0.05 0.10 0.125 0.15 0.175 0.20 0.25 0.30 0.50"
#   DGD_PYTHON=./venv/bin/python
#   DGD_OUTPUT_ROOT=runs/minha_busca_dgcg
#   DGD_WORK_ROOT=runs/cache_compartilhado
#   DGD_RESUME=0
#   DGD_FAIL_FAST=1

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir" || exit 1

if [[ -n "${DGD_PYTHON:-}" ]]; then
    python_bin="$DGD_PYTHON"
elif [[ -x "$script_dir/venv/bin/python" ]]; then
    python_bin="$script_dir/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
else
    python_bin="python"
fi

program="${DGD_PROTO_SCRIPT:-protosgc_fewshot.py}"
output_root="${DGD_OUTPUT_ROOT:-runs/busca_dgcg_grau_densidade}"
work_root="${DGD_WORK_ROOT:-$output_root/_work}"
resume="${DGD_RESUME:-1}"
fail_fast="${DGD_FAIL_FAST:-0}"
n_way="${DGD_N_WAY:-5}"
n_shot="${DGD_N_SHOT:-1}"
n_query="${DGD_N_QUERY:-5}"
quick=0
dry_run=0

usage() {
    sed -n '3,25p' "${BASH_SOURCE[0]}"
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
    echo "Erro: DGD_RESUME deve ser 0 ou 1." >&2
    exit 2
fi
if [[ "$fail_fast" != "0" && "$fail_fast" != "1" ]]; then
    echo "Erro: DGD_FAIL_FAST deve ser 0 ou 1." >&2
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

for value in "$n_way" "$n_shot" "$n_query"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Erro: n-way, n-shot e n-query devem ser inteiros positivos." >&2
        exit 2
    fi
done
if (( n_way > 5 )); then
    echo "Erro: n-way=$n_way excede as 5 classes fixas de validacao/teste." >&2
    exit 2
fi
if (( n_shot + n_query > 40 )); then
    echo "Erro: n-shot+n-query deve ser <= 40 nos datasets Micro." >&2
    exit 2
fi

episode_nodes=$(( n_way * (n_shot + n_query) ))
candidate_k=$(( episode_nodes - 1 ))

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

if [[ -n "${DGD_DATASETS:-}" ]]; then
    read -r -a datasets <<< "$DGD_DATASETS"
else
    datasets=("${default_datasets[@]}")
fi

# Um valor simples representa [valor, valor]. MIN:MAX permite testar intervalos
# como o default oficial 4:6, incluido explicitamente como baseline.
read -r -a degree_targets <<< "${DGD_DEGREES:-1 2 3 4 5 4:6 6 8 10 12 15 20 25}"
read -r -a density_targets <<< "${DGD_DENSITIES:-0.025 0.05 0.075 0.10 0.125 0.15 0.175 0.20 0.25 0.30 0.40 0.50 0.60 0.75 0.90}"

if (( ${#datasets[@]} == 0 || ${#degree_targets[@]} == 0 || ${#density_targets[@]} == 0 )); then
    echo "Erro: datasets e grades de grau/densidade nao podem ser vazios." >&2
    exit 2
fi

for dataset in "${datasets[@]}"; do
    if [[ -z "${dataset_id[$dataset]+x}" ]]; then
        echo "Erro: dataset desconhecido: $dataset" >&2
        echo "Disponiveis: ${default_datasets[*]}" >&2
        exit 2
    fi
done

split_target() {
    local token="$1"
    local low high
    if [[ "$token" == *:* ]]; then
        low="${token%%:*}"
        high="${token#*:}"
    else
        low="$token"
        high="$token"
    fi
    printf '%s %s\n' "$low" "$high"
}

for token in "${degree_targets[@]}"; do
    read -r low high < <(split_target "$token")
    if ! awk -v low="$low" -v high="$high" -v maximum="$candidate_k" \
        'BEGIN { exit !(low ~ /^[0-9]+([.][0-9]+)?$/ && high ~ /^[0-9]+([.][0-9]+)?$/ && low > 0 && low <= high && high <= maximum) }'; then
        echo "Erro: alvo de grau invalido '$token'; use 0 < MIN <= MAX <= $candidate_k." >&2
        exit 2
    fi
done

for token in "${density_targets[@]}"; do
    read -r low high < <(split_target "$token")
    if ! awk -v low="$low" -v high="$high" \
        'BEGIN { exit !(low ~ /^[0-9]+([.][0-9]+)?$/ && high ~ /^[0-9]+([.][0-9]+)?$/ && low > 0 && low <= high && high < 1) }'; then
        echo "Erro: alvo de densidade invalido '$token'; use 0 < MIN <= MAX < 1." >&2
        exit 2
    fi
done

total_runs=$(( ${#datasets[@]} * (${#degree_targets[@]} + ${#density_targets[@]}) ))
run_number=0
failed_runs=0
skipped_runs=0
stop_requested=0

mkdir -p "$output_root" "$work_root"

echo "Busca de grau medio e densidade do DGCG"
echo "  datasets    : ${datasets[*]}"
echo "  protocolo   : ${n_way}-way/${n_shot}-shot/${n_query}-query"
echo "  nos         : $episode_nodes"
echo "  candidatos  : $candidate_k por no"
echo "  graus       : ${degree_targets[*]}"
echo "  densidades  : ${density_targets[*]}"
echo "  ordem       : grau, depois densidade"
echo "  execucoes   : $total_runs"
echo "  resultados  : $output_root"
echo "  cache/work  : $work_root"
if (( quick )); then
    echo "  modo        : quick (200/50/200 episodios)"
else
    echo "  modo        : completo (defaults: 2000/200/1000 episodios)"
fi
echo

run_configuration() {
    local mode="$1"
    local dataset="$2"
    local token="$3"
    local openml_id="${dataset_id[$dataset]}"
    local low high label protocol run_dir log_path ok_path checkpoint_copy
    local shared_work_dir checkpoint_path run_status
    local -a command

    read -r low high < <(split_target "$token")
    label="${low}a${high}"
    label="${label//./p}"
    protocol="${n_way}way_${n_shot}shot_${n_query}query"
    run_dir="$output_root/$mode/$dataset/${mode}_${label}"
    log_path="$run_dir/run.log"
    ok_path="$run_dir/status.ok"
    checkpoint_copy="$run_dir/checkpoint.pt"
    shared_work_dir="$work_root/$dataset"
    mkdir -p "$run_dir" "$shared_work_dir"

    ((run_number += 1))
    printf '[%d/%d] %s | %s | alvo=%s | %s\n' \
        "$run_number" "$total_runs" "$mode" "$dataset" "$token" "$protocol"

    if (( resume )) && [[ -f "$ok_path" ]]; then
        echo "  ja concluido; pulando (DGD_RESUME=1)."
        ((skipped_runs += 1))
        return 0
    fi

    command=(
        "$python_bin" "$program"
        --openml-id "$openml_id"
        --class-split 10 5 5
        --work-dir "$shared_work_dir"
        --n-way "$n_way"
        --n-shot "$n_shot"
        --n-query "$n_query"
        --graph-type dgcg
    )
    if [[ "$mode" == "grau" ]]; then
        command+=(--dgcg-target-degree "$low" "$high")
    else
        command+=(--dgcg-target-density "$low" "$high")
    fi
    if (( quick )); then
        command+=(--quick)
    fi

    if (( dry_run )); then
        printf ' '
        printf '%q ' "${command[@]}"
        printf '\n'
        return 0
    fi

    rm -f "$ok_path"
    {
        printf 'Comando:'
        printf ' %q' "${command[@]}"
        printf '\n'
        printf 'Modo: %s\nAlvo: %s %s\nCandidatos: %s\n' \
            "$mode" "$low" "$high" "$candidate_k"
    } > "$run_dir/command.txt"

    "${command[@]}" 2>&1 | tee "$log_path"
    run_status=${PIPESTATUS[0]}
    if (( run_status != 0 )); then
        echo "  FALHOU (codigo $run_status): $log_path" >&2
        ((failed_runs += 1))
        if (( fail_fast )); then
            stop_requested=1
        fi
        return "$run_status"
    fi

    : > "$ok_path"
    checkpoint_path="$({ sed -n 's/^[[:space:]]*checkpoint: //p' "$log_path" || true; } | tail -n 1)"
    if [[ -n "$checkpoint_path" && -f "$checkpoint_path" ]]; then
        cp -f -- "$checkpoint_path" "$checkpoint_copy"
    else
        echo "  Aviso: checkpoint nao localizado no log." >&2
    fi
    return 0
}

# A ordem e intencional: termina toda a busca por grau antes de iniciar a
# parametrizacao equivalente por densidade.
for mode in grau densidade; do
    if [[ "$mode" == "grau" ]]; then
        targets=("${degree_targets[@]}")
    else
        targets=("${density_targets[@]}")
    fi
    for dataset in "${datasets[@]}"; do
        for token in "${targets[@]}"; do
            run_configuration "$mode" "$dataset" "$token" || true
            if (( stop_requested )); then
                break 3
            fi
        done
    done
done

if (( dry_run )); then
    echo
    echo "Dry-run concluido; nenhuma execucao foi iniciada."
    exit 0
fi

"$python_bin" - "$output_root" "$n_way" "$n_shot" "$n_query" "$candidate_k" <<'PY'
from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


root = Path(sys.argv[1])
n_way, n_shot, n_query, candidate_k = map(int, sys.argv[2:])
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
target_pattern = re.compile(r"^(grau|densidade)_(?P<low>[0-9p]+)a(?P<high>[0-9p]+)$")

ids = {
    "bacteria_micro": 44237, "plankton_micro": 44238, "flowers_micro": 44239,
    "sports_micro": 44240, "birds_micro": 44241, "plant_village_micro": 44242,
    "textures_micro": 44243, "manufacturing_micro": 44244, "cars_micro": 44245,
    "resisc_micro": 44246,
}

fields = [
    "modo", "dataset", "openml_id", "n_way", "n_shot", "n_query",
    "candidatos", "alvo_min", "alvo_max", "grau_equivalente_min",
    "grau_equivalente_max", "densidade_equivalente_min",
    "densidade_equivalente_max", "val_accuracy", "val_ci95", "val_loss",
    "test_accuracy", "test_ci95", "test_loss", "status", "log", "checkpoint",
]

rows: list[dict[str, object]] = []
for mode in ("grau", "densidade"):
    mode_root = root / mode
    if not mode_root.exists():
        continue
    for log_path in sorted(mode_root.glob("*/*/run.log")):
        dataset = log_path.parents[1].name
        target = target_pattern.fullmatch(log_path.parent.name)
        if target is None:
            continue
        low = float(target.group("low").replace("p", "."))
        high = float(target.group("high").replace("p", "."))
        if mode == "grau":
            degree_low, degree_high = low, high
            density_low, density_high = low / candidate_k, high / candidate_k
        else:
            density_low, density_high = low, high
            degree_low, degree_high = low * candidate_k, high * candidate_k

        text = log_path.read_text(encoding="utf-8", errors="replace")
        validations = list(validation_pattern.finditer(text))
        best_val = max(validations, key=lambda item: float(item.group("accuracy"))) if validations else None
        tests = list(test_pattern.finditer(text))
        test = tests[-1] if tests else None
        successful = (log_path.parent / "status.ok").is_file() and best_val is not None and test is not None
        rows.append({
            "modo": mode,
            "dataset": dataset,
            "openml_id": ids.get(dataset, ""),
            "n_way": n_way,
            "n_shot": n_shot,
            "n_query": n_query,
            "candidatos": candidate_k,
            "alvo_min": low,
            "alvo_max": high,
            "grau_equivalente_min": degree_low,
            "grau_equivalente_max": degree_high,
            "densidade_equivalente_min": density_low,
            "densidade_equivalente_max": density_high,
            "val_accuracy": float(best_val.group("accuracy")) if best_val else None,
            "val_ci95": float(best_val.group("ci95")) if best_val else None,
            "val_loss": float(best_val.group("loss")) if best_val else None,
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
    0 if row["modo"] == "grau" else 1,
    str(row["dataset"]), float(row["alvo_min"]), float(row["alvo_max"]),
))
write_csv(root / "resultados.csv", rows)
valid = [row for row in rows if row["status"] == "ok"]

by_dataset_mode: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
by_dataset: dict[object, list[dict[str, object]]] = defaultdict(list)
for row in valid:
    by_dataset_mode[(row["dataset"], row["modo"])].append(row)
    by_dataset[row["dataset"]].append(row)

best_by_mode = [
    max(group, key=lambda row: (float(row["val_accuracy"]), -float(row["val_loss"])))
    for group in by_dataset_mode.values()
]
best_by_mode.sort(key=lambda row: (str(row["dataset"]), str(row["modo"])))
write_csv(root / "melhores_por_dataset_e_modo.csv", best_by_mode)

best_by_dataset = [
    max(group, key=lambda row: (float(row["val_accuracy"]), -float(row["val_loss"])))
    for group in by_dataset.values()
]
best_by_dataset.sort(key=lambda row: str(row["dataset"]))
write_csv(root / "melhores_por_dataset.csv", best_by_dataset)

by_target: dict[tuple[object, object, object], list[dict[str, object]]] = defaultdict(list)
for row in valid:
    by_target[(row["modo"], row["alvo_min"], row["alvo_max"])].append(row)

ranking_fields = [
    "modo", "alvo_min", "alvo_max", "grau_equivalente_min",
    "grau_equivalente_max", "densidade_equivalente_min",
    "densidade_equivalente_max", "datasets_concluidos", "media_val_accuracy",
    "desvio_val_accuracy", "media_test_accuracy",
]
ranking: list[dict[str, object]] = []
for (mode, low, high), group in by_target.items():
    ranking.append({
        "modo": mode,
        "alvo_min": low,
        "alvo_max": high,
        "grau_equivalente_min": group[0]["grau_equivalente_min"],
        "grau_equivalente_max": group[0]["grau_equivalente_max"],
        "densidade_equivalente_min": group[0]["densidade_equivalente_min"],
        "densidade_equivalente_max": group[0]["densidade_equivalente_max"],
        "datasets_concluidos": len(group),
        "media_val_accuracy": statistics.fmean(float(row["val_accuracy"]) for row in group),
        "desvio_val_accuracy": statistics.pstdev(float(row["val_accuracy"]) for row in group),
        "media_test_accuracy": statistics.fmean(float(row["test_accuracy"]) for row in group),
    })
ranking.sort(key=lambda row: (str(row["modo"]), -float(row["media_val_accuracy"])))
write_csv(root / "ranking_medio.csv", ranking, ranking_fields)

best_mean: list[dict[str, object]] = []
for mode in ("grau", "densidade"):
    candidates = [row for row in ranking if row["modo"] == mode]
    if not candidates:
        continue
    coverage = max(int(row["datasets_concluidos"]) for row in candidates)
    comparable = [row for row in candidates if int(row["datasets_concluidos"]) == coverage]
    best_mean.append(max(comparable, key=lambda row: float(row["media_val_accuracy"])))
write_csv(root / "melhores_medios.csv", best_mean, ranking_fields)

print(f"Resultados agregados: {root / 'resultados.csv'}")
print(f"Melhores por dataset: {root / 'melhores_por_dataset.csv'}")
print(f"Ranking medio: {root / 'ranking_medio.csv'}")
print(f"Melhores medios: {root / 'melhores_medios.csv'}")
PY
aggregate_status=$?

echo
echo "Busca encerrada: $((run_number - skipped_runs - failed_runs)) novas execucoes concluidas, $skipped_runs puladas, $failed_runs falhas."
echo "Resultados: $output_root/resultados.csv"
echo "Melhores por dataset e modo: $output_root/melhores_por_dataset_e_modo.csv"
echo "Melhores por dataset: $output_root/melhores_por_dataset.csv"
echo "Ranking medio: $output_root/ranking_medio.csv"
echo "Melhores medios: $output_root/melhores_medios.csv"

if (( aggregate_status != 0 || failed_runs != 0 )); then
    exit 1
fi
