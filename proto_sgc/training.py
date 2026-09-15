"""Meta-treino, avaliação episódica e seleção do melhor checkpoint pela validação."""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange

from .config import DGCG_GRAPH_TYPES
from .episodes import FeatureEpisodeSampler, episode_to_device
from .model import ProtoSGC
from .graphs.training_rankings import TrainingRankings


@dataclass(frozen=True)
class Metrics:
    """Resumo da avaliacao: entropia cruzada, acuracia media e IC de 95%."""

    loss: float
    accuracy: float
    ci95: float


@torch.no_grad()
def evaluate(
    model: ProtoSGC,
    sampler: FeatureEpisodeSampler,
    allowed_classes: Sequence[str],
    n_episodes: int,
    n_way: int,
    device: torch.device,
    seed: int,
) -> Metrics:
    """Avalia o modelo em episodios reproduziveis, sem calcular gradientes.

    A acuracia e calculada separadamente em cada episodio. O intervalo de 95%
    usa a variabilidade entre essas acuracias episodicas.
    """

    if n_episodes < 1:
        raise ValueError("O numero de episodios de avaliacao deve ser >= 1.")
    model.eval()
    # O gerador local evita que a avaliacao altere a sequencia aleatoria usada
    # pelo meta-treino ou por outra topologia.
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    accuracies: list[float] = []
    for _ in range(n_episodes):
        episode = episode_to_device(sampler.sample(allowed_classes, rng), device)
        logits = model(
            episode.support_x,
            episode.support_y,
            episode.query_x,
            n_way,
        )
        losses.append(F.cross_entropy(logits, episode.query_y).item())
        predictions = logits.argmax(dim=1)
        accuracies.append((predictions == episode.query_y).float().mean().item())

    accuracy_array = np.asarray(accuracies, dtype=np.float64)

    # Aproximacao normal: media +/- 1.96 * erro-padrao.
    ci95 = (
        1.96 * accuracy_array.std(ddof=1) / math.sqrt(n_episodes)
        if n_episodes > 1
        else 0.0
    )
    return Metrics(
        loss=float(np.mean(losses)),
        accuracy=float(accuracy_array.mean()),
        ci95=float(ci95),
    )


def train_model(
    model: ProtoSGC,
    sampler: FeatureEpisodeSampler,
    splits: dict[str, list[str]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ProtoSGC, Metrics, Path]:
    """Executa meta-treino episodico, selecao por validacao e meta-teste.

    O melhor estado e escolhido exclusivamente pela acuracia nas classes de
    validacao. Somente depois desse estado ser restaurado o teste e executado
    nas classes reservadas, evitando selecionar hiperparametros pelo teste.
    """

    model.to(device)
    use_global_rankings = (
        getattr(args, "dgcg_train_global_rankings", False)
        and args.model == "proto-sgc"
        and args.graph_type in DGCG_GRAPH_TYPES
    )
    training_rankings = None
    if use_global_rankings:
        print("Preparando rankings DGCG com todas as imagens de treino...")
        training_rankings = TrainingRankings(
            sampler.features_by_class, splits["train"],
            list_size=args.dgcg_list_size, metric=args.dgcg_metric,
        )
        print(
            f"Rankings de treino: {training_rankings.rankings.shape[0]} imagens, "
            f"L={training_rankings.rankings.shape[1]}. "
            "Somente correlacoes do DGCG usam contexto global; "
            "validacao/teste permanecem episodicos."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # Sementes diferentes separam conceitualmente meta-treino, validacao e
    # teste, enquanto continuam reproduziveis.
    train_rng = np.random.default_rng(args.seed + 101)
    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -math.inf
    # O sufixo registra no nome do checkpoint as escolhas que alteram o grafo.
    graph_suffix = args.graph_type if args.model == "proto-sgc" else "no-graph"
    if args.model == "proto-sgc" and args.graph_type.startswith("knn") and model.knn_metric != "euclidean":
        graph_suffix = f"{graph_suffix}-{model.knn_metric}"
    if args.model == "proto-sgc" and args.graph_type in DGCG_GRAPH_TYPES:
        if args.dgcg_threshold is not None:
            threshold_tag = f"t{args.dgcg_threshold:g}"
        elif args.dgcg_target_density is not None:
            density_low, density_high = args.dgcg_target_density
            threshold_tag = f"auto-rho{density_low:g}-{density_high:g}"
        else:
            degree_low, degree_high = args.dgcg_target_degree
            threshold_tag = f"auto-deg{degree_low:g}-{degree_high:g}"
        threshold_tag = threshold_tag.replace(".", "p")
        graph_suffix = (
            f"{args.graph_type}-{args.dgcg_metric}-"
            f"{args.dgcg_correlation}-{threshold_tag}"
        )
    if args.model == "proto-sgc" and args.cosine_rbf_weight:
        sigma_tag = f"{args.graph_temperature:g}".replace(".", "p")
        graph_suffix = f"{graph_suffix}-cosine-rbf-sigma{sigma_tag}"
    if args.model == "proto-sgc" and args.grande:
        sigma_tag = f"{args.grande_sigma:g}".replace(".", "p")
        metric_tag = args.grande_metric
        if args.grande_metric == "rbo":
            metric_tag = f"rbo-p{args.grande_rbo_p:g}".replace(".", "p")
        graph_suffix = f"{graph_suffix}-grande-{metric_tag}-sigma{sigma_tag}"
    if use_global_rankings:
        graph_suffix = f"{graph_suffix}-train-global-rankings"
    checkpoint_path = args.work_dir / f"best_{args.model}_{graph_suffix}.pt"
    # As medias moveis abaixo sao apenas para a barra de progresso; elas nao
    # interferem na loss otimizada nem na escolha do checkpoint.
    recent_losses: list[float] = []
    recent_accuracies: list[float] = []

    progress = trange(1, args.train_episodes + 1, desc="Meta-treino", unit="ep")
    for episode_index in progress:
        model.train()

        # Cada iteracao do otimizador corresponde a um novo episodio few-shot.
        episode = episode_to_device(sampler.sample(splits["train"], train_rng), device)
        graph_options = {}
        if training_rankings is not None:
            graph_options["dgcg_correlations"] = training_rankings.correlations(
                episode.sample_ids,
                correlation=args.dgcg_correlation,
                top_k=args.dgcg_top_k,
                rbo_p=args.dgcg_rbo_p,
                dtype=model.theta.weight.dtype,
                device=device,
            )
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            episode.support_x,
            episode.support_y,
            episode.query_x,
            args.n_way,
            **graph_options,
        )
        loss = F.cross_entropy(logits, episode.query_y)
        loss.backward()

        # O clipping reduz o efeito de episodios com gradientes atipicamente altos.
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        accuracy = (logits.argmax(dim=1) == episode.query_y).float().mean().item()
        recent_losses.append(loss.item())
        recent_accuracies.append(accuracy)
        if len(recent_losses) > 50:
            recent_losses.pop(0)
            recent_accuracies.pop(0)
        progress.set_postfix(
            loss=f"{np.mean(recent_losses):.3f}",
            acc=f"{100 * np.mean(recent_accuracies):.1f}%",
        )

        validate_now = episode_index % args.val_every == 0 or episode_index == args.train_episodes
        if validate_now:
            # Seed fixa: cada checkpoint e comparado exatamente nos mesmos episodios.
            val_metrics = evaluate(
                model,
                sampler,
                splits["val"],
                args.val_episodes,
                args.n_way,
                device,
                seed=args.seed + 202,
            )
            print(
                f"\nEp. {episode_index:4d} | validacao: "
                f"{100 * val_metrics.accuracy:.2f}% +/- "
                f"{100 * val_metrics.ci95:.2f}% | loss={val_metrics.loss:.4f}"
            )
            if val_metrics.accuracy > best_val_accuracy:
                # Mantem uma copia em memoria e outra em disco. O teste final
                # usara exatamente este estado, nao o estado da ultima epoca.
                best_val_accuracy = val_metrics.accuracy
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "input_dim": model.theta.in_features,
                        "args": vars(args),
                        "class_splits": splits,
                        "validation": asdict(val_metrics),
                    },
                    checkpoint_path,
                )

    # Meta-teste: somente classes de ``splits["test"]``, nunca vistas acima.
    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        sampler,
        splits["test"],
        args.test_episodes,
        args.n_way,
        device,
        seed=args.seed + 303,
    )
    return model, test_metrics, checkpoint_path
