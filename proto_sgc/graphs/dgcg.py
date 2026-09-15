"""DGCG e DGCG+: rankings, correlações, seleção das arestas e pesos de vizinhança mútua.

As funções recebem as opções explicitamente e não dependem do modelo treinável.
``dtype`` mantém os cálculos de correlação no mesmo tipo numérico de Theta.
Adaptação de https://github.com/Bagiel1/DGCG e https://github.com/Bagiel1/DGCGplus.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_ranked_lists(
    features: torch.Tensor, list_size: int, *, metric: str,
) -> torch.Tensor:
    """Ordena os vizinhos de cada no segundo ``--dgcg-metric``.

    Usado pela topologia do DGCG/DGCG+ e, quando ``--grande-metric rbo``,
    tambem pelas listas que alimentam o grau GRaNDe.
    """
    n_nodes = features.shape[0]
    diagonal = torch.eye(n_nodes, dtype=torch.bool, device=features.device)
    # Garante que o proprio no ocupe a primeira posicao, inclusive quando
    # houver vetores duplicados ou empates numericos.
    if metric == "cosine":
        unit_features = F.normalize(features, p=2, dim=1)
        similarities = unit_features @ unit_features.T
        ranking_values = similarities.masked_fill(diagonal, torch.inf)
        return ranking_values.argsort(dim=1, descending=True)[:, :list_size]
    distances = torch.cdist(features, features, p=2)
    ranking_values = distances.masked_fill(diagonal, -torch.inf)
    return ranking_values.argsort(dim=1)[:, :list_size]


def dgcg_correlation_matrix(
    ranked_lists: torch.Tensor,
    *,
    correlation: str,
    rbo_p: float,
    top_k: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Calcula RBO/Jaccard entre todos os pares de listas ranqueadas.

    A formulacao segue as funcoes do repositorio oficial do DGCG. O
    calculo abaixo apenas vetoriza as intersecoes entre prefixos para
    evitar os lacos Python quadruplicados durante o treino episodico.
    ``correlation`` e ``rbo_p`` permitem que o GRaNDe use RBO com o seu
    proprio p, sem interferir na correlacao escolhida para a topologia.
    Cada linha pode conter IDs de um universo maior que o numero de linhas,
    como quando apenas os rankings dos nos episodicos sao retirados do treino.
    """
    n_nodes, list_size = ranked_lists.shape
    top_k = min(top_k, list_size)
    truncated = ranked_lists[:, :top_k]

    # prefixes[i, d] indica os itens presentes nos primeiros d+1 lugares
    # do ranking i. Cada ranking nao contem elementos repetidos.
    # Os rankings podem referenciar imagens fora do episodio. Compacta os IDs
    # presentes preservando igualdade/intersecoes, sem alocar pelo tamanho global.
    unique_ids, compact_ids = torch.unique(truncated, return_inverse=True)
    prefixes = F.one_hot(
        compact_ids.reshape_as(truncated), num_classes=unique_ids.numel()
    ).cumsum(dim=1)
    prefixes = prefixes.clamp_max(1).to(dtype=dtype)
    # Para cada profundidade d, overlaps[d, i, j] e o tamanho da intersecao
    # entre os prefixos dos rankings dos nos i e j.
    overlaps = torch.einsum("idv,jdv->dij", prefixes, prefixes)
    depths = torch.arange(
        1,
        top_k + 1,
        dtype=overlaps.dtype,
        device=overlaps.device,
    )

    if correlation == "rbo":
        # RBO pondera mais fortemente o topo. Quanto maior p, mais devagar
        # a importancia decai ao avançar no ranking.
        powers = torch.arange(
            top_k,
            dtype=overlaps.dtype,
            device=overlaps.device,
        )
        weights = (1.0 - rbo_p) * rbo_p**powers
        return (weights[:, None, None] * overlaps / depths[:, None, None]).sum(
            dim=0
        )

    # Jaccard = |intersecao| / |uniao| em cada profundidade do ranking.
    jaccard_by_depth = overlaps / (
        2.0 * depths[:, None, None] - overlaps
    ).clamp_min(1e-12)
    if correlation == "jaccardk":
        return jaccard_by_depth.mean(dim=0)
    if correlation == "jaccard-max":
        return jaccard_by_depth.amax(dim=0)
    if correlation == "jaccard-median":
        ordered = jaccard_by_depth.sort(dim=0).values
        middle = top_k // 2
        if top_k % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])
    raise AssertionError(f"Correlacao DGCG inesperada: {correlation!r}.")


def dgcg_mutual_neighborhood_scores(
    ranked_lists: torch.Tensor, *, top_k: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Calcula o escore por no usado como peso no DGCG+.

    Para cada no i, o repositorio do DGCG+ percorre os top-k vizinhos de i
    e suas respectivas listas top-k. Cada item que tambem aparece na lista
    de i contribui com o inverso de sua posicao nessa lista. A soma e
    dividida por top-k ao quadrado. Esta implementacao e algebricamente
    equivalente, mas vetoriza as intersecoes.
    """
    n_nodes, list_size = ranked_lists.shape
    top_k = min(top_k, list_size)
    truncated = ranked_lists[:, :top_k]
    score_dtype = dtype

    # membership[i, v] indica se v aparece no prefixo top-k do no i.
    membership = torch.zeros(
        (n_nodes, n_nodes),
        dtype=score_dtype,
        device=ranked_lists.device,
    )
    membership.scatter_(dim=1, index=truncated, value=1.0)

    # reciprocal_ranks[i, v] guarda 1/posicao de v no ranking de i.
    reciprocal_ranks = torch.zeros_like(membership)
    rank_weights = torch.arange(
        1,
        top_k + 1,
        dtype=score_dtype,
        device=ranked_lists.device,
    ).reciprocal()
    reciprocal_ranks.scatter_(
        dim=1,
        index=truncated,
        src=rank_weights.expand(n_nodes, -1),
    )

    # pair_scores[i, j] acumula 1/rank_i(v) para v em R_i e R_j.
    pair_scores = reciprocal_ranks @ membership.T
    return pair_scores.gather(dim=1, index=truncated).sum(dim=1) / float(
        top_k**2
    )


def effective_dgcg_degree_interval(
    candidate_k: int,
    *,
    target_degree: tuple[float, float],
    target_density: tuple[float, float] | None,
) -> tuple[float, float]:
    """Converte o alvo configurado para grau medio no episodio atual."""
    if target_density is not None:
        density_low, density_high = target_density
        return density_low * candidate_k, density_high * candidate_k
    degree_low, degree_high = target_degree
    return min(degree_low, candidate_k), min(degree_high, candidate_k)


def automatic_dgcg_selection(
    candidate_scores: torch.Tensor,
    *,
    target_degree: tuple[float, float],
    target_density: tuple[float, float] | None,
) -> torch.Tensor:
    """Seleciona o limiar observado mais proximo do intervalo de grau.

    Em episodios pequenos, RBO/Jaccard assume poucos valores distintos e
    uma bissecao numerica pode saltar o intervalo inteiro. Por isso, esta
    rotina avalia exatamente as densidades alcancaveis pelos escores do
    episodio, mantendo todos os empates no mesmo lado do limiar.
    """
    n_nodes, candidate_k = candidate_scores.shape
    degree_low, degree_high = effective_dgcg_degree_interval(
        candidate_k, target_degree=target_degree, target_density=target_density
    )
    target_midpoint = 0.5 * (degree_low + degree_high)

    unique_scores, counts = torch.unique(
        candidate_scores.reshape(-1),
        sorted=True,
        return_counts=True,
    )
    # Ao percorrer os escores do maior para o menor, o numero de arestas
    # selecionadas so cresce. Empates entram juntos e nunca sao quebrados.
    unique_scores = unique_scores.flip(0)
    selected_counts = counts.flip(0).cumsum(dim=0).to(candidate_scores.dtype)
    attainable_degrees = selected_counts / float(n_nodes)
    # Inclui o grafo sem arestas entre as alternativas; se ele for o mais
    # proximo, o fallback por no abaixo evita propagacao puramente identica.
    attainable_degrees = torch.cat(
        (attainable_degrees.new_zeros(1), attainable_degrees), dim=0
    )

    # Primeiro minimiza a distancia ao intervalo permitido; em caso de
    # empate, escolhe o grau mais proximo do centro desse intervalo.
    interval_distance = torch.where(
        attainable_degrees < degree_low,
        degree_low - attainable_degrees,
        torch.where(
            attainable_degrees > degree_high,
            attainable_degrees - degree_high,
            torch.zeros_like(attainable_degrees),
        ),
    )
    midpoint_distance = (attainable_degrees - target_midpoint).abs()
    best_interval_distance = interval_distance.min()
    midpoint_distance = torch.where(
        interval_distance == best_interval_distance,
        midpoint_distance,
        torch.full_like(midpoint_distance, torch.inf),
    )
    best_index = int(midpoint_distance.argmin().item())
    if best_index == 0:
        return torch.zeros_like(candidate_scores, dtype=torch.bool)

    cutoff = unique_scores[best_index - 1]
    return candidate_scores >= cutoff


def dgcg_adjacency(
    features: torch.Tensor,
    weighted: bool = False,
    *,
    metric: str,
    correlation: str,
    list_size: int,
    candidate_k: int,
    top_k: int,
    threshold: float | None,
    target_degree: tuple[float, float],
    target_density: tuple[float, float] | None,
    rbo_p: float,
    dtype: torch.dtype,
    correlations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Constroi a adjacencia do DGCG/DGCG+, sem autolacos.

    O DGCG inclui o proprio item na lista usada pela correlacao e conserva
    arestas cuja correlacao supera um limiar. A distancia Euclidiana e usada
    por padrao, como nas ranked lists geradas pelo ``BallTree`` original.
    Quando ``weighted`` e verdadeiro, a topologia permanece identica e
    cada aresta recebe a media dos escores de vizinhanca mutua dos seus
    extremos, exatamente como no DGCG+.
    ``correlations`` opcional substitui somente os escores de selecao por uma
    matriz externa na ordem dos nos do episodio. Candidatos, limiar e pesos
    de vizinhanca mutua continuam calculados a partir do episodio.
    Os autolacos sao retirados da selecao e adicionados depois, de forma
    uniforme para todos os tipos de grafo.

    O repositorio oficial armazena ``[origem, destino]`` para o PyG. Como
    este codigo propaga por ``A @ X``, retornamos o transposto para manter
    a mesma direcao de mensagens.
    """
    n_nodes = features.shape[0]
    if n_nodes < 2:
        raise ValueError("O episodio precisa ter ao menos dois nos.")

    list_size = min(list_size, n_nodes)
    candidate_k = min(candidate_k, list_size - 1, n_nodes - 1)
    if candidate_k < 1:
        raise ValueError(
            "O DGCG precisa de ao menos um candidato; use dgcg-list-size >= 2."
        )

    ranked_lists = build_ranked_lists(features, list_size, metric=metric)
    if correlations is None:
        correlations = dgcg_correlation_matrix(
            ranked_lists, correlation=correlation, rbo_p=rbo_p, top_k=top_k, dtype=dtype
        )
    else:
        if correlations.shape != (n_nodes, n_nodes):
            raise ValueError("Correlacoes externas devem ter formato [nos do episodio, nos do episodio].")
        correlations = correlations.detach().to(device=features.device, dtype=dtype)

    # A primeira coluna e o proprio no. As colunas seguintes sao os
    # candidatos que podem originar arestas na topologia DGCG.
    candidates = ranked_lists[:, 1 : candidate_k + 1]
    candidate_scores = correlations.gather(dim=1, index=candidates)
    if threshold is None:
        selected = automatic_dgcg_selection(
            candidate_scores, target_degree=target_degree, target_density=target_density
        )
        # Um alvo global pode deixar um no isolado. No modo automatico,
        # conserva para cada caso desses a maior correlacao disponivel.
        isolated = ~selected.any(dim=1)
        if bool(isolated.any()):
            best_candidates = candidate_scores.argmax(dim=1)
            isolated_rows = torch.arange(n_nodes, device=features.device)[isolated]
            selected[isolated_rows, best_candidates[isolated]] = True
    else:
        selected = candidate_scores > threshold

    # adjacency_out representa a convencao natural i -> candidato(i).
    adjacency_out = torch.zeros(
        (n_nodes, n_nodes), dtype=features.dtype, device=features.device
    )
    if weighted:
        # DGCG+ nao escolhe novas arestas: altera somente o valor das
        # arestas previamente aceitas pelo limiar DGCG.
        mutual_scores = dgcg_mutual_neighborhood_scores(
            ranked_lists, top_k=top_k, dtype=dtype
        )
        candidate_weights = 0.5 * (
            mutual_scores[:, None] + mutual_scores[candidates]
        ).to(dtype=adjacency_out.dtype)
        edge_values = selected.to(dtype=adjacency_out.dtype) * candidate_weights
    else:
        edge_values = selected.to(dtype=adjacency_out.dtype)
    adjacency_out.scatter_(
        dim=1,
        index=candidates,
        src=edge_values,
    )
    # A multiplicacao A @ X interpreta cada linha como os vetores que o no
    # agrega; por isso a matriz e transposta antes de sair deste metodo.
    return adjacency_out.T
