"""Rankings com referencias de treino e consultas de correlacao por episodio.

Nao cria grafo global. A avaliacao pode acrescentar somente o episodio atual
ao universo de referencias. Os nomes de classes
servem apenas para localizar exemplos, nunca para calcular similaridades.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .dgcg import dgcg_correlation_matrix


@torch.no_grad()
def build_reference_ranked_lists(
    features: torch.Tensor, reference: torch.Tensor, *, list_size: int, metric: str,
) -> torch.Tensor:
    """Ordena episodio contra referencias + episodio, sem receber rotulos."""
    if features.ndim != 2 or not len(features) or reference.ndim != 2:
        raise ValueError("Forneca embeddings matriciais e episodio nao vazio.")
    if features.shape[1] != reference.shape[1]:
        raise ValueError("Dimensao das referencias difere do episodio.")
    if list_size < 2 or metric not in ("cosine", "euclidean"):
        raise ValueError("L deve ser >= 2 e metrica deve ser euclidean ou cosine.")
    reference = reference.to(device=features.device, dtype=features.dtype)
    universe = torch.cat((reference, features.detach()), dim=0)
    if metric == "cosine":
        values = F.normalize(features, p=2, dim=1) @ F.normalize(universe, p=2, dim=1).T
    else:
        values = torch.cdist(features, universe, p=2)
    rows = torch.arange(len(features), device=features.device)
    values[rows, len(reference) + rows] = torch.inf if metric == "cosine" else -torch.inf
    return values.argsort(dim=1, descending=metric == "cosine", stable=True)[
        :, :min(list_size, len(universe))
    ]


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
        self.features = features
        self.list_size = list_size
        self.metric = metric
        self._reference_by_device: dict[torch.device, torch.Tensor] = {}
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
    def episode_rankings(self, features: torch.Tensor) -> torch.Tensor:
        """Rankings dos nos episodicos no universo treino + episodio atual.

        IDs [0, N_treino) identificam referencias de treino; os IDs seguintes
        identificam suporte+consulta. Nenhum outro episodio entra neste universo.
        O proprio no e sempre o primeiro item. Nao recebe rotulos.
        """
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("Forneca embeddings de um episodio nao vazio.")
        if features.shape[1] != self.features.shape[1]:
            raise ValueError("Dimensao dos embeddings difere do conjunto de treino.")
        device = features.device
        reference = self._reference_by_device.get(device)
        if reference is None:
            reference = self.features.to(device)
            self._reference_by_device[device] = reference
        return build_reference_ranked_lists(
            features, reference, list_size=self.list_size, metric=self.metric,
        )

    @torch.no_grad()
    def episode_correlations(
        self, features: torch.Tensor, *, correlation: str, top_k: int,
        rbo_p: float, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compara apenas rankings dos nos do episodio, usando treino como referencia."""
        return dgcg_correlation_matrix(
            self.episode_rankings(features), correlation=correlation,
            top_k=top_k, rbo_p=rbo_p, dtype=dtype,
        )

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
