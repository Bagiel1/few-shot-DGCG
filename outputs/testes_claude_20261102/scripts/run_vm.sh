#!/usr/bin/env bash
# Roda a comparação de backbones na VM (GPU). Uso, a partir da raiz do repo:
#   bash outputs/testes_claude_20261102/scripts/run_vm.sh [python] [nproc]
# Retoma de onde parou se interrompido (pula execuções já gravadas no .jsonl).
set -euo pipefail
cd "$(dirname "$0")"
PY=${1:-python}
NP=${2:-4}
export FS_DEVICE=${FS_DEVICE:-cuda} PYTHONDONTWRITEBYTECODE=1
OUT=../resultados

# 1) features (uma vez; baixa as bases pelo OpenML se não existirem em data/)
$PY extract.py dinov2s 224
$PY extract.py resnet18 128   # só usado se não houver o cache antigo em runs/

# 2) ResNet-18 de referência na mesma máquina, e DINOv2 ViT-S/14 (seed 7)
$PY runner.py bb:resnet18:128 $OUT/res_bb_vm.jsonl $NP
$PY runner.py bb:dinov2s:224  $OUT/res_bb_vm.jsonl $NP
echo "Pronto: $OUT/res_bb_vm.jsonl"
