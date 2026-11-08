"""Homofilia posterior a previsao: os rotulos nao entram no grafo."""

import hashlib
import json

import numpy as np
import torch

from .episodes import episode_to_device


def edge_agreement(adjacency, labels, n_support):
    """Soma pesos entre classes iguais, excluindo autolacos.

    A @ X: linha i recebe atributos das colunas j. O recorte [n_support:]
    mede apenas as linhas de query, como na figura do PDF.
    """
    weights = adjacency.detach().clone()
    weights.fill_diagonal_(0)
    same = labels[:, None].eq(labels[None, :]).to(weights.dtype)
    return {
        "same_all": float((weights * same).sum().item()),
        "weight_all": float(weights.sum().item()),
        "same_query": float((weights[n_support:] * same[n_support:]).sum().item()),
        "weight_query": float(weights[n_support:].sum().item()),
    }


@torch.no_grad()
def measure_homophily(model, sampler, classes, n_way, episodes, device, seed):
    """Razao entre somas em todos os episodios, nao media de porcentagens.

    Usa a topologia efetiva antes de autolacos/normalizacao. Na configuracao
    padrao sem GRaNDe, essa topologia depende de X congelado, nao de Theta.
    """
    if episodes < 1 or model.model_name != "proto-sgc":
        raise ValueError("Homofilia requer grafo e numero positivo de episodios.")
    model.eval()
    rng = np.random.default_rng(seed)
    totals = dict.fromkeys(("same_all", "weight_all", "same_query", "weight_query"), 0.)
    digest = hashlib.sha256()
    for _ in range(episodes):
        sampled = sampler.sample(classes, rng)
        digest.update(json.dumps(sampled.sample_ids, separators=(",", ":")).encode())
        episode = episode_to_device(sampled, device)
        features = torch.cat((episode.support_x, episode.query_x))
        # O grafo e construido ANTES de ler os rotulos para o diagnostico.
        adjacency = model.graph_adjacency(features)
        labels = torch.cat((episode.support_y, episode.query_y))
        counts = edge_agreement(adjacency, labels, len(episode.support_x))
        for key, value in counts.items():
            totals[key] += value
    return dict(totals, homophily_all_percent=(100 * totals["same_all"] / totals["weight_all"]
                if totals["weight_all"] else None),
                homophily_query_percent=(100 * totals["same_query"] / totals["weight_query"]
                if totals["weight_query"] else None),
                diagnostic_episodes=episodes, episode_ids_sha256=digest.hexdigest())


def pearson(x, y):
    """Sem p-valor: os dois grafos da mesma base nao sao amostras independentes."""
    if len(x) != len(y):
        raise ValueError("Vetores de correlacao devem ter o mesmo tamanho.")
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])
