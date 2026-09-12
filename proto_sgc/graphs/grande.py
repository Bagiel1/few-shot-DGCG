"""Distâncias das arestas e grau GRaNDe usado na normalização da adjacência.

GRaNDe recebe os vetores já projetados por Theta; não escolhe as arestas.
Para a métrica RBO, o chamador fornece as correlações entre os rankings.
Adaptação de https://github.com/rduarte12/SBBD-2026_GRaNDe.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grande_edge_distances(
    features: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    correlations: torch.Tensor | None = None,
    *,
    metric: str,
) -> torch.Tensor:
    """Distancias das arestas usadas pelo grau GRaNDe.

    ``euclidean`` reproduz o repositorio oficial. ``cosine`` mede
    ``1 - cosseno`` entre os mesmos vetores, ficando invariante a norma dos
    descritores. ``rbo`` mede ``1 - RBO`` entre as listas ranqueadas
    desses vetores, ou seja, dois nos ficam proximos quando ordenam o
    episodio de forma parecida. Em todos os casos o restante do
    procedimento (min-max global e kernel gaussiano) nao muda.
    """
    if metric == "rbo":
        if correlations is None:
            raise ValueError("GRaNDe com RBO requer a matriz de correlacoes dos rankings.")
        return (1.0 - correlations[sources, targets]).clamp_min(0.0)
    if metric == "cosine":
        unit_features = F.normalize(features, p=2, dim=1)
        cosine_similarity = (
            (unit_features[sources] * unit_features[targets])
            .sum(dim=1)
            .clamp(-1.0, 1.0)
        )
        return (1.0 - cosine_similarity).clamp_min(0.0)
    return torch.linalg.vector_norm(
        features[sources] - features[targets],
        ord=2,
        dim=1,
    )


def grande_degree(
    features: torch.Tensor,
    adjacency: torch.Tensor,
    correlations: torch.Tensor | None = None,
    *,
    metric: str,
    sigma: float,
) -> torch.Tensor:
    """Calcula o GRaNDe de cada linha da adjacencia.

    A implementacao segue o repositorio oficial: conta as arestas (inclusive
    o autolaco), normaliza globalmente por min-max as distancias das arestas
    (Euclidianas por padrao; de cosseno ou por RBO conforme
    ``--grande-metric``) e soma ao grau a media do inverso do kernel
    ``exp(-distancia_normalizada^2 / sigma)``. Os valores das arestas nao
    entram no grau GRaNDe; eles continuam sendo usados na propagacao.
    """
    # GRaNDe considera a existencia da aresta. O peso da aresta continua na
    # adjacencia usada pela propagacao, mas nao na contagem de grau abaixo.
    edge_mask = adjacency.ne(0)
    sources, targets = edge_mask.nonzero(as_tuple=True)
    degree = edge_mask.sum(dim=1).to(dtype=features.dtype)

    edge_distances = grande_edge_distances(
        features, sources, targets, correlations, metric=metric
    ).clamp_min(1e-12)
    # O codigo oficial converte os extremos para escalares Python. O
    # ``detach`` preserva o mesmo gradiente sem forcar sincronizacao CPU/GPU.
    min_distance = edge_distances.amin().detach()
    distance_range = (edge_distances.amax() - min_distance).detach()
    normalized_distances = torch.where(
        distance_range > 1e-12,
        (edge_distances - min_distance) / distance_range.clamp_min(1e-12),
        torch.zeros_like(edge_distances),
    )
    # A formulacao oficial usa exp(-d^2/sigma), sem o fator 2 do RBF
    # gaussiano convencional.
    gaussian_similarities = torch.exp(
        -normalized_distances.square() / sigma
    )
    inverse_similarities = gaussian_similarities.clamp_min(1e-12).reciprocal()
    penalty_sum = torch.zeros_like(degree)
    penalty_sum.scatter_add_(0, sources, inverse_similarities)
    neighborhood_penalty = penalty_sum / degree.clamp_min(1.0)
    return degree + neighborhood_penalty
