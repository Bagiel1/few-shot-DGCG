"""Topologias kNN euclidianas e ponderação RBF opcional.

A função RBF também pode ponderar DGCG e DGCG+, preservando suas arestas.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def knn_adjacency(
    features: torch.Tensor,
    *,
    knn: int,
    graph_type: str,
) -> torch.Tensor:
    """Constroi uma adjacencia kNN euclidiana, binaria e sem autolacos.

    A convencao e ``A_out[i, j] > 0`` quando ``j`` pertence ao kNN de
    ``i`` (aresta ``i -> j``). Portanto, em ``A @ X``, ``knn-out`` faz o
    no ``i`` agregar os vizinhos que ele escolheu; ``knn-in`` agrega os
    nos que escolheram ``i``. ``torch.cdist(..., p=2)`` produz a mesma
    distancia Euclidiana do ``BallTree`` original (Minkowski com ``p=2``).
    """
    n_nodes = features.shape[0]
    if n_nodes < 2:
        raise ValueError("O episodio precisa ter ao menos dois nos.")
    k = min(knn, n_nodes - 1)

    distances = torch.cdist(features, features, p=2)
    diagonal = torch.eye(n_nodes, dtype=torch.bool, device=features.device)
    neighbors = distances.masked_fill(diagonal, torch.inf).topk(
        k=k, dim=1, largest=False
    ).indices
    adjacency_out = torch.zeros(
        (n_nodes, n_nodes), dtype=features.dtype, device=features.device
    )
    adjacency_out.scatter_(
        dim=1,
        index=neighbors,
        src=torch.ones_like(neighbors, dtype=features.dtype),
    )

    if graph_type == "knn-out":
        return adjacency_out
    if graph_type == "knn-in":
        return adjacency_out.T
    if graph_type == "knn-union":
        return torch.maximum(adjacency_out, adjacency_out.T)
    if graph_type == "knn-reciprocal":
        return torch.minimum(adjacency_out, adjacency_out.T)
    raise AssertionError(f"Variante de grafo inesperada: {graph_type}")


@torch.no_grad()
def apply_cosine_rbf_weights(
    features: torch.Tensor, adjacency: torch.Tensor, *, temperature: float,
) -> torch.Tensor:
    """Aplica pesos RBF da distancia cosseno apenas nas arestas existentes.

    Esta e uma ablação opcional, ativada apenas por ``--cosine-rbf-weight``.
    A mascara de ``adjacency`` preserva exatamente a topologia e a direcao
    do grafo. Pesos nativos do DGCG+ tambem sao substituidos quando a opcao
    e usada.
    """
    unit_features = F.normalize(features, p=2, dim=1)
    cosine_similarity = (unit_features @ unit_features.T).clamp(-1.0, 1.0)
    cosine_distance = (1.0 - cosine_similarity).clamp_min(0.0)
    rbf_weights = torch.exp(
        -cosine_distance.square() / (2.0 * temperature**2)
    )
    edge_mask = adjacency.ne(0).to(dtype=adjacency.dtype)
    return edge_mask * rbf_weights.to(dtype=adjacency.dtype)
