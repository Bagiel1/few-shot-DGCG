"""Rankings com contexto de todo o treino; consultas de correlacao por episodio.

Nao cria grafo global nem usa classes de validacao/teste. Os nomes de classes
servem apenas para localizar exemplos, nunca para calcular similaridades.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .dgcg import dgcg_correlation_matrix


class TrainingRankings:
    """Guarda N x L indices em CPU, sem materializar correlacoes N x N."""

    @torch.no_grad()
    def __init__(
        self,
        features_by_class: dict[str, torch.Tensor],
        train_classes: Sequence[str],
        *,
        list_size: int,
        metric: str,
        batch_size: int = 256,
    ) -> None:
        if metric not in ("euclidean", "cosine"):
            raise ValueError("Metrica dos rankings deve ser euclidean ou cosine.")
        if list_size < 2 or batch_size < 1:
            raise ValueError("list_size deve ser >= 2 e batch_size >= 1.")
        if not train_classes or len(set(train_classes)) != len(train_classes):
            raise ValueError("Forneca classes de treino distintas e nao vazias.")
        blocks = []
        self.row_by_id: dict[tuple[str, int], int] = {}
        for label in sorted(train_classes):
            block = features_by_class[label].detach().cpu()
            for index in range(len(block)):
                self.row_by_id[(label, index)] = len(self.row_by_id)
            blocks.append(block)
        features = torch.cat(blocks)
        n = len(features)
        if n < 2:
            raise ValueError("Rankings de treino requerem ao menos duas imagens.")
        length = min(list_size, n)
        self.rankings = torch.empty((n, length), dtype=torch.long)
        vectors = F.normalize(features, p=2, dim=1) if metric == "cosine" else features
        # Limita a matriz temporaria de distancias. A busca ainda compara todos
        # os pares, mas mantem apenas L indices por imagem depois de cada bloco.
        block_size = min(batch_size, max(1, 4_000_000 // n))
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            if metric == "cosine":
                values = vectors[start:stop] @ vectors.T
            else:
                values = torch.cdist(vectors[start:stop], vectors, p=2)
            rows = torch.arange(stop - start)
            own_ids = torch.arange(start, stop)
            values[rows, own_ids] = torch.inf if metric == "cosine" else -torch.inf
            self.rankings[start:stop] = values.argsort(
                dim=1, descending=metric == "cosine", stable=True
            )[:, :length]

    @torch.no_grad()
    def correlations(
        self,
        sample_ids: Sequence[tuple[str, int]],
        *,
        correlation: str,
        top_k: int,
        rbo_p: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Seleciona rankings na ordem suporte+consulta e compara seus prefixos."""
        if not sample_ids:
            raise ValueError("O episodio precisa identificar seus exemplos para usar rankings de treino.")
        try:
            rows = [self.row_by_id[item] for item in sample_ids]
        except KeyError as error:
            raise ValueError("Episodio contem exemplo fora do conjunto de treino.") from error
        rankings = self.rankings[rows].to(device)
        return dgcg_correlation_matrix(
            rankings, correlation=correlation, top_k=top_k, rbo_p=rbo_p, dtype=dtype
        )
