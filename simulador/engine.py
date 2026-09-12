"""Executa o modelo original e registra os tensores de cada passo para inspeção.

Não altera o treinamento do projeto. Os hooks abaixo apenas observam saídas;
o forward, a construção dos grafos, a amostragem e a avaliação são os reais.
O conjunto sintético substitui somente a extração da ResNet congelada.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from proto_sgc.config import DGCG_CORRELATIONS, DGCG_METRICS, GRAPH_TYPES
from proto_sgc.episodes import Episode, FeatureEpisodeSampler, split_classes
from proto_sgc.model import ProtoSGC
from proto_sgc.runtime import _safe_torch_load
from proto_sgc.training import evaluate


@dataclass(frozen=True)
class SimulationConfig:
    model_name: str = "proto-sgc"
    graph_type: str = "knn-union"
    grande_metric: str = "off"
    cosine_rbf_weight: bool = False
    n_way: int = 3
    n_shot: int = 2
    n_query: int = 3
    steps: int = 40
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    knn: int = 3
    sgc_hops: int = 2
    output_dim: int = 4
    graph_temperature: float = 0.5
    grande_sigma: float = 0.2
    dgcg_metric: str = "euclidean"
    dgcg_correlation: str = "rbo"
    dgcg_top_k: int = 10
    dgcg_threshold: float | None = None
    grande_rbo_p: float = 0.9
    noise: float = 0.65
    seed: int = 7


INTEGER_LIMITS = {
    "n_way": (2, 5), "n_shot": (1, 3), "n_query": (1, 5),
    "steps": (1, 120), "knn": (1, 12), "sgc_hops": (1, 5),
    "output_dim": (3, 12), "dgcg_top_k": (1, 40), "seed": (0, 100000),
}
FLOAT_LIMITS = {
    "learning_rate": (0.0001, 0.1), "weight_decay": (0.0, 0.1),
    "grad_clip": (0.01, 20.0), "graph_temperature": (0.05, 2.0),
    "grande_sigma": (0.05, 2.0), "grande_rbo_p": (0.1, 0.99),
    "noise": (0.05, 1.5),
}
CHOICES = {
    "model_name": ("proto-sgc", "protonet"), "graph_type": GRAPH_TYPES,
    "grande_metric": ("off", "euclidean", "cosine", "rbo"),
    "dgcg_metric": DGCG_METRICS, "dgcg_correlation": DGCG_CORRELATIONS,
}


def parse_config(values: dict) -> SimulationConfig:
    """Valida os controles antes de alocar tensores ou iniciar o treino."""
    if not isinstance(values, dict):
        raise ValueError("A configuração precisa ser um objeto JSON.")
    unknown = set(values) - {field.name for field in fields(SimulationConfig)}
    if unknown:
        raise ValueError(f"Parâmetros desconhecidos: {', '.join(sorted(unknown))}.")
    cfg = SimulationConfig(**values)
    for name, (low, high) in INTEGER_LIMITS.items():
        value = getattr(cfg, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name}: use um inteiro de {low} a {high}.")
    for name, (low, high) in FLOAT_LIMITS.items():
        value = getattr(cfg, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name}: use um número de {low} a {high}.")
    for name, choices in CHOICES.items():
        if getattr(cfg, name) not in choices:
            raise ValueError(f"{name}: escolha uma opção válida.")
    if type(cfg.cosine_rbf_weight) is not bool:
        raise ValueError("cosine_rbf_weight deve ser verdadeiro ou falso.")
    threshold = cfg.dgcg_threshold
    if threshold is not None and (type(threshold) not in (int, float) or not 0 <= threshold <= 1):
        raise ValueError("O limiar DGCG deve estar entre 0 e 1, ou vazio para automático.")
    if cfg.model_name == "protonet" and (cfg.grande_metric != "off" or cfg.cosine_rbf_weight):
        raise ValueError("ProtoNet sem grafo requer GRaNDe e pesos RBF desativados.")
    return cfg


def synthetic_features(cfg: SimulationConfig) -> dict[str, torch.Tensor]:
    """Cria embeddings fixos de 8 dimensões, com 4×N classes e 16 itens/classe."""
    rng = np.random.default_rng(cfg.seed)
    features = {}
    for index in range(4 * cfg.n_way):
        angle = 2 * math.pi * index / (4 * cfg.n_way)
        center = rng.normal(0, 1.1, 8)
        center[:2] = 2.2 * np.array([math.cos(angle), math.sin(angle)])
        features[f"C{index + 1:02d}"] = torch.tensor(
            center + rng.normal(0, cfg.noise, (16, 8)), dtype=torch.float32
        )
    return features


def load_features(path: Path) -> dict[str, torch.Tensor]:
    """Lê somente o cache de embeddings já extraído pelo programa principal."""
    cached = _safe_torch_load(path)
    if not isinstance(cached, dict) or not isinstance(cached.get("features_by_class"), dict):
        raise ValueError("O arquivo não contém features_by_class, como no cache da ResNet.")
    features = {}
    for label, value in cached["features_by_class"].items():
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or not torch.isfinite(value).all():
            raise ValueError(f"Embeddings inválidos na classe {label}.")
        features[str(label)] = value.detach().cpu().float().contiguous()
    dimensions = {value.shape[1] for value in features.values()}
    if len(dimensions) != 1 or min(dimensions) < 2:
        raise ValueError("Todas as classes precisam ter a mesma dimensão, com ao menos 2 atributos.")
    return features


class ObservedProtoSGC(ProtoSGC):
    """Acrescenta observação dos tensores, preservando o forward de ProtoSGC."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observed = {}
        self.normalization.register_forward_pre_hook(self._before_normalization)
        self.normalization.register_forward_hook(self._after_normalization)

    def _before_normalization(self, module, inputs):
        self.observed["projected_propagated"] = inputs[0].detach().clone()

    def _after_normalization(self, module, inputs, output):
        self.observed["layernorm"] = output.detach().clone()

    def graph_adjacency(self, features):
        adjacency = super().graph_adjacency(features)
        self.observed["adjacency"] = adjacency.detach().clone()
        return adjacency

    def normalized_graph_adjacency(self, graph_features, degree_features=None):
        adjacency = super().normalized_graph_adjacency(graph_features, degree_features)
        self.observed["normalized_adjacency"] = adjacency.detach().clone()
        return adjacency


def model_arguments(cfg: SimulationConfig, input_dim: int) -> dict:
    return dict(
        input_dim=input_dim, output_dim=cfg.output_dim, model_name=cfg.model_name,
        knn=cfg.knn, graph_type=cfg.graph_type, sgc_hops=cfg.sgc_hops,
        graph_temperature=cfg.graph_temperature, grande=cfg.grande_metric != "off",
        grande_metric="euclidean" if cfg.grande_metric == "off" else cfg.grande_metric,
        grande_sigma=cfg.grande_sigma, grande_rbo_p=cfg.grande_rbo_p,
        cosine_rbf_weight=cfg.cosine_rbf_weight, dgcg_metric=cfg.dgcg_metric,
        dgcg_correlation=cfg.dgcg_correlation, dgcg_top_k=cfg.dgcg_top_k,
        dgcg_threshold=cfg.dgcg_threshold, dgcg_target_degree=(4.0, 6.0),
    )


def tensor_values(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def parameters(model: ProtoSGC) -> dict:
    """Mostra Theta na convenção X @ Theta (transposta do peso nn.Linear)."""
    return {
        "theta": tensor_values(model.theta.weight.T[:8]),
        "gamma": tensor_values(model.normalization.weight),
        "beta": tensor_values(model.normalization.bias),
        "logit_scale": model.logit_scale.item(),
        "scale": model.logit_scale.exp().clamp(max=100).item(),
    }


@torch.no_grad()
def inspect_episode(model: ObservedProtoSGC, episode: Episode, cfg: SimulationConfig) -> dict:
    """Observa o forward e reconstrói apenas os saltos intermediários do SGC."""
    model.observed.clear()
    logits = model(episode.support_x, episode.support_y, episode.query_x, cfg.n_way)
    x = torch.cat((episode.support_x, episode.query_x))
    embeddings = F.normalize(model.observed["layernorm"], p=2, dim=1)
    n_support = len(episode.support_x)
    prototypes = torch.stack([
        embeddings[:n_support][episode.support_y == label].mean(0)
        for label in range(cfg.n_way)
    ])
    adjacency = model.observed.get("adjacency", torch.zeros(len(x), len(x)))
    normalized = model.observed.get("normalized_adjacency", torch.eye(len(x)))
    propagated = model.theta(x) if model.grande or cfg.model_name == "protonet" else x
    # Caches reais podem ter 512 dimensões. A interface recebe um recorte;
    # todas as operações do modelo e da propagação usam os tensores completos.
    hops = [tensor_values(propagated[:, :8])]
    if cfg.model_name == "proto-sgc":
        for _ in range(cfg.sgc_hops):
            propagated = normalized @ propagated
            hops.append(tensor_values(propagated[:, :8]))
    degree_adjacency = adjacency + torch.eye(len(x))
    if model.grande:
        projected = model.theta(x)
        correlations = model.grande_rank_correlations(projected) if model.grande_metric == "rbo" else None
        out_degree = model.grande_degree(projected, degree_adjacency, correlations)
        in_degree = model.grande_degree(projected, degree_adjacency.T, correlations)
    else:
        out_degree, in_degree = degree_adjacency.sum(1), degree_adjacency.sum(0)
    predictions = logits.argmax(1)
    return {
        "x": tensor_values(x[:, :8]), "adjacency": tensor_values(adjacency),
        "normalized_adjacency": tensor_values(normalized),
        "out_degree": tensor_values(out_degree), "in_degree": tensor_values(in_degree),
        "hops": hops, "projected_propagated": tensor_values(model.observed["projected_propagated"]),
        "layernorm": tensor_values(model.observed["layernorm"]),
        "embeddings": tensor_values(embeddings), "prototypes": tensor_values(prototypes),
        "distances": tensor_values(torch.cdist(embeddings[n_support:], prototypes).square()),
        "logits": tensor_values(logits), "probabilities": tensor_values(logits.softmax(1)),
        "predictions": tensor_values(predictions),
        "loss": F.cross_entropy(logits, episode.query_y).item(),
        "accuracy": (predictions == episode.query_y).float().mean().item(),
        "scale": model.logit_scale.exp().clamp(max=100).item(),
    }


def feature_lookup(features: dict[str, torch.Tensor]) -> dict[bytes, list[tuple[str, str]]]:
    """Indexa as identidades uma vez, sem repetir a busca em cada passo."""
    lookup = {}
    for label, values in features.items():
        for index, vector in enumerate(values):
            lookup.setdefault(vector.numpy().tobytes(), []).append((label, f"{label}/{index + 1:02d}"))
    return lookup


def episode_metadata(episode: Episode, lookup: dict) -> dict:
    """Associa os nós aos registros sem mudar a amostragem do projeto."""
    labels = torch.cat((episode.support_y, episode.query_y)).tolist()
    vectors = torch.cat((episode.support_x, episode.query_x))
    matches = [lookup[vector.numpy().tobytes()] for vector in vectors]
    class_names = {}
    for label in set(labels):
        candidates = [set(item[0] for item in values) for index, values in enumerate(matches) if labels[index] == label]
        common = set.intersection(*candidates)
        # Embeddings duplicados em várias classes não revelam a identidade;
        # nesse caso, exiba o rótulo local verdadeiro, sem inventar uma classe.
        class_names[label] = next(iter(common)) if len(common) == 1 else f"Classe local {label}"
    nodes = []
    for index, vector in enumerate(vectors):
        label = class_names[labels[index]]
        identities = [item[1] for item in matches[index] if item[0] == label]
        image_id = identities[0] if len(identities) == 1 else f"{label}/nó-{index}"
        nodes.append({"index": index, "id": image_id, "class_name": label,
                      "label": labels[index], "role": "support" if index < len(episode.support_x) else "query"})
    classes = [next(node["class_name"] for node in nodes if node["label"] == label) for label in range(max(labels) + 1)]
    return {"nodes": nodes, "classes": classes, "support_count": len(episode.support_x),
            "query_labels": episode.query_y.tolist()}


def simulate(cfg: SimulationConfig, cached_features: dict[str, torch.Tensor] | None = None) -> dict:
    """Treina passo a passo; a validação escolhe o estado usado no teste final."""
    features = cached_features if cached_features is not None else synthetic_features(cfg)
    lookup = feature_lookup(features)
    input_dim = next(iter(features.values())).shape[1]
    splits = split_classes(sorted(features), cfg.n_way, None, cfg.seed)
    sampler = FeatureEpisodeSampler(features, cfg.n_way, cfg.n_shot, cfg.n_query)
    probe = sampler.sample(splits["train"], np.random.default_rng(cfg.seed + 404))
    train_rng = np.random.default_rng(cfg.seed + 101)
    snapshots = []
    # O RNG do usuário que importar este módulo é restaurado ao terminar.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        model = ObservedProtoSGC(**model_arguments(cfg, input_dim))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        initial_parameters = parameters(model)
        initial_probe = inspect_episode(model, probe, cfg)
        best_accuracy, best_step, best_state = -math.inf, 0, copy.deepcopy(model.state_dict())
        for step in range(1, cfg.steps + 1):
            model.train()
            episode = sampler.sample(splits["train"], train_rng)
            before = inspect_episode(model, episode, cfg)
            before_parameters = parameters(model)
            old_weight = model.theta.weight.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            logits = model(episode.support_x, episode.support_y, episode.query_x, cfg.n_way)
            loss = F.cross_entropy(logits, episode.query_y)
            loss.backward()
            raw_gradient = model.theta.weight.grad.detach().clone()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            clipped_gradient = model.theta.weight.grad.detach().clone()
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("O treino ficou numericamente instável. Reduza a taxa de aprendizado.")
            after = inspect_episode(model, episode, cfg)
            probe_state = inspect_episode(model, probe, cfg)
            validation = None
            if step % 5 == 0 or step == cfg.steps:
                validation = asdict(evaluate(model, sampler, splits["val"], 10, cfg.n_way, torch.device("cpu"), cfg.seed + 202))
                if validation["accuracy"] > best_accuracy:
                    best_accuracy, best_step = validation["accuracy"], step
                    best_state = copy.deepcopy(model.state_dict())
            state = optimizer.state[model.theta.weight]
            snapshots.append({
                "step": step, "episode": episode_metadata(episode, lookup),
                "before": before, "after": after, "probe": probe_state,
                "parameters_before": before_parameters, "parameters_after": parameters(model),
                "gradient": tensor_values(raw_gradient.T[:8]),
                "clipped_gradient": tensor_values(clipped_gradient.T[:8]),
                "gradient_norm": norm, "clip_factor": min(1.0, cfg.grad_clip / (norm + 1e-6)),
                "weight_delta_norm": (model.theta.weight.detach() - old_weight).norm().item(),
                "adam_first_moment": state["exp_avg"][0, 0].item(),
                "adam_second_moment": state["exp_avg_sq"][0, 0].item(),
                "validation": validation,
            })
        final_parameters = parameters(model)
        model.load_state_dict(best_state)
        test = asdict(evaluate(model, sampler, splits["test"], 20, cfg.n_way, torch.device("cpu"), cfg.seed + 303))
    return {
        "config": asdict(cfg), "input_dim": input_dim,
        "source": "cache" if cached_features is not None else "synthetic",
        "splits": splits, "total_images": sum(len(v) for v in features.values()),
        "initial_parameters": initial_parameters, "initial_probe": initial_probe,
        "probe_episode": episode_metadata(probe, lookup), "steps": snapshots,
        "best_step": best_step, "best_validation_accuracy": best_accuracy,
        "test": test, "final_parameters": final_parameters,
    }
