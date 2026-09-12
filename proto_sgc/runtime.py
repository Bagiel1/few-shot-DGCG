"""Sementes aleatórias, escolha do dispositivo e compatibilidade ao carregar tensores."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fixa as fontes de aleatoriedade usadas por Python, NumPy e PyTorch.

    O modo deterministico do cuDNN favorece a reprodutibilidade. Dependendo do
    hardware, ele pode ser um pouco mais lento que a selecao automatica de
    algoritmos habilitada por ``benchmark=True``.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    """Resolve ``auto/cpu/cuda/mps`` e valida dispositivos pedidos à força."""

    # Para escolhas explicitas, falhar cedo e mais claro do que deixar o
    # primeiro tensor enviado ao dispositivo produzir um erro menos informativo.
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda foi solicitado, mas CUDA nao esta disponivel.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps foi solicitado, mas MPS nao esta disponivel.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    # Em ``auto``, CUDA tem prioridade sobre MPS e CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_torch_load(path: Path) -> object:
    """Carrega um arquivo aceitando PyTorch com ou sem ``weights_only``."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch anterior ao argumento weights_only
        return torch.load(path, map_location="cpu")
