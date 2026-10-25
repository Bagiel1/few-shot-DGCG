"""Metricas episodicas e auditoria da topologia, sem dependencias adicionais."""

from contextlib import contextmanager

import numpy as np
import torch


def classification_scores(targets, predictions, n_classes):
    """F1 macro e recall macro (acuracia balanceada), em [0, 1].

    Classes locais 0..n_classes-1; divisao por zero produz zero no F1.
    Episodios deste projeto sempre possuem consultas de todas as classes.
    """
    confusion = torch.bincount(
        targets * n_classes + predictions, minlength=n_classes**2,
    ).reshape(n_classes, n_classes).double()
    true_positive = confusion.diag()
    actual = confusion.sum(1)
    predicted = confusion.sum(0)
    f1 = 2 * true_positive / (actual + predicted).clamp_min(1)
    recall = true_positive / actual.clamp_min(1)
    return f1.mean().item(), recall.mean().item()


class GraphStatistics:
    """Vizinhos nao nulos antes de autolacos/normalizacao, nao soma de pesos.

    Linha = vizinhos agregados por A @ X; coluna = nos que agregam aquele no.
    """

    def __init__(self):
        self.rows = []
        self.columns = []
        self.densities = []
        self.degree_target = None
        self.candidate_limit = None

    def update(self, adjacency):
        mask = adjacency.detach().ne(0).clone()
        mask.fill_diagonal_(False)
        self.rows.append(mask.sum(1).cpu().numpy())
        self.columns.append(mask.sum(0).cpu().numpy())
        n = len(mask)
        self.densities.append(float(mask.sum().item() / (n * (n - 1))) if n > 1 else 0.)

    def summary(self):
        if not self.rows:
            return None
        row = np.concatenate(self.rows)
        column = np.concatenate(self.columns)
        means = np.asarray([r.mean() for r in self.rows])
        result = {
            "episodes": len(self.rows), "nodes_per_episode": len(self.rows[0]),
            "degree_mean": float(row.mean()), "degree_std": float(row.std()),
            "degree_min": int(row.min()), "degree_max": int(row.max()),
            "degree_p05": float(np.quantile(row, .05)),
            "degree_p95": float(np.quantile(row, .95)),
            "episode_degree_mean_std": float(means.std()),
            "episode_degree_mean_min": float(means.min()),
            "episode_degree_mean_max": float(means.max()),
            "column_degree_min": int(column.min()), "column_degree_max": int(column.max()),
            "column_degree_std": float(column.std()),
            "zero_row_fraction": float((row == 0).mean()),
            "isolated_node_fraction": float(((row == 0) & (column == 0)).mean()),
            "density_mean": float(np.mean(self.densities)),
            "density_std": float(np.std(self.densities)),
        }
        if self.degree_target is not None:
            candidates = min(self.candidate_limit, len(self.rows[0]) - 1)
            low, high = (min(value, candidates) for value in self.degree_target)
            result.update({
                "candidate_k": candidates,
                "requested_degree_low": self.degree_target[0],
                "requested_degree_high": self.degree_target[1],
                "effective_degree_low": low, "effective_degree_high": high,
                "degree_target_clipped": tuple(self.degree_target) != (low, high),
                "episode_mean_outside_target_fraction": float(((means < low-1e-6) | (means > high+1e-6)).mean()),
            })
        return result


@contextmanager
def record_graphs(model):
    """Observa o grafo ja construido; nao recalcula nem altera a propagacao."""
    statistics = GraphStatistics()
    if (model.model_name == "proto-sgc" and model.graph_type in ("dgcg", "dgcg-plus")
            and model.dgcg_threshold is None and model.dgcg_target_density is None):
        statistics.degree_target = model.dgcg_target_degree
        statistics.candidate_limit = min(model.dgcg_candidate_k, model.dgcg_list_size - 1)
    previous = getattr(model, "graph_observer", None)
    model.graph_observer = statistics.update
    try:
        yield statistics
    finally:
        model.graph_observer = previous
