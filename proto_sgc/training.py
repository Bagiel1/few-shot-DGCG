"""Meta-treino, avaliação episódica e seleção do melhor checkpoint pela validação."""

from __future__ import annotations

import argparse
import copy
import json
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
from .graphs.training_rankings import build_reference_ranked_lists
from .graphs.dgcg import dgcg_correlation_matrix
from .reference_bank import ExtraReferenceBank
from .metrics import classification_scores, record_graphs


@dataclass(frozen=True)
class Metrics:
    """Medias episodicas, IC95 e topologia; novos campos opcionais para legado."""

    loss: float
    accuracy: float
    ci95: float
    macro_f1: float | None = None
    macro_f1_ci95: float | None = None
    balanced_accuracy: float | None = None
    balanced_accuracy_ci95: float | None = None
    graph: dict | None = None


def extra_graph_options(model, episode, bank, mode="rankings"):
    """Nao recebe rotulos das extras; preserva a ordem suporte/query/extra."""
    if mode not in ("rankings", "propagation"):
        raise ValueError("Modo de extras desconhecido.")
    if bank is None or not bank.count:
        return {}
    extras = bank.extra_features(episode).to(episode.support_x)
    if mode == "propagation":
        return {"extra_x": extras}
    if model.model_name != "proto-sgc" or model.graph_type not in DGCG_GRAPH_TYPES:
        raise ValueError("Extras somente nos rankings requerem DGCG.")
    rankings = build_reference_ranked_lists(
        torch.cat((episode.support_x, episode.query_x)), extras,
        list_size=model.dgcg_list_size, metric=model.dgcg_metric,
    )
    return {"dgcg_correlations": dgcg_correlation_matrix(
        rankings, correlation=model.dgcg_correlation, top_k=model.dgcg_top_k,
        rbo_p=model.dgcg_rbo_p, dtype=model.theta.weight.dtype,
    )}


def prepare_extra_banks(sampler, splits, args):
    """Reserva todas as fases para comparacoes pareadas; usa so as habilitadas."""
    count = getattr(args, "extra_images", None)
    if count is None:
        return None
    all_classes = [label for phase in ("train", "val", "test") for label in splits[phase]]
    if len(set(all_classes)) != len(all_classes):
        raise ValueError("Classes de treino, validacao e teste devem ser disjuntas.")
    return {
        phase: ExtraReferenceBank(
            sampler, splits[phase], count=count if phase in args.extra_phases else 0,
            per_class=args.extra_bank_per_class, seed=args.seed + offset,
        )
        for phase, offset in [("train", 4101), ("val", 4202), ("test", 4303)]
    }


@torch.no_grad()
def evaluate(model, *args, **kwargs):
    # Restaura o observador mesmo se a avaliacao falhar.
    with record_graphs(model) as graph_statistics:
        return _evaluate(model, *args, graph_statistics=graph_statistics, **kwargs)


def _evaluate(
    model: ProtoSGC,
    sampler: FeatureEpisodeSampler,
    allowed_classes: Sequence[str],
    n_episodes: int,
    n_way: int,
    device: torch.device,
    seed: int,
    *,
    training_rankings: TrainingRankings | None = None,
    extra_reference_bank: ExtraReferenceBank | None = None,
    extra_mode: str = "rankings",
    graph_statistics=None,
) -> Metrics:
    """Avalia o modelo em episodios reproduziveis, sem calcular gradientes.

    A acuracia e calculada separadamente em cada episodio. O intervalo de 95%
    usa a variabilidade entre essas acuracias episodicas.
    """

    if n_episodes < 1:
        raise ValueError("O numero de episodios de avaliacao deve ser >= 1.")
    if training_rankings is not None and extra_reference_bank is not None:
        raise ValueError("Escolha um unico contexto extra de avaliacao.")
    if extra_reference_bank is not None and set(allowed_classes) != set(extra_reference_bank.classes):
        raise ValueError("Banco de referencias pertence a outra particao de classes.")
    model.eval()
    # O gerador local evita que a avaliacao altere a sequencia aleatoria usada
    # pelo meta-treino ou por outra topologia.
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    accuracies: list[float] = []
    f1_scores: list[float] = []
    balanced_scores: list[float] = []
    for _ in range(n_episodes):
        sampled = (extra_reference_bank.sample(rng) if extra_reference_bank is not None
                   else sampler.sample(allowed_classes, rng))
        episode = episode_to_device(sampled, device)
        graph_options = extra_graph_options(model, episode, extra_reference_bank, extra_mode)
        if training_rankings is not None:
            if any(item in training_rankings.row_by_id for item in episode.sample_ids):
                raise ValueError("Avaliacao de referencia requer episodio disjunto do treino.")
            graph_options["dgcg_correlations"] = training_rankings.episode_correlations(
                torch.cat((episode.support_x, episode.query_x)),
                correlation=model.dgcg_correlation, top_k=model.dgcg_top_k,
                rbo_p=model.dgcg_rbo_p, dtype=model.theta.weight.dtype,
            )
        logits = model(
            episode.support_x,
            episode.support_y,
            episode.query_x,
            n_way,
            **graph_options,
        )
        losses.append(F.cross_entropy(logits, episode.query_y).item())
        predictions = logits.argmax(dim=1)
        accuracies.append((predictions == episode.query_y).float().mean().item())
        f1, balanced = classification_scores(episode.query_y, predictions, n_way)
        f1_scores.append(f1)
        balanced_scores.append(balanced)

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
        macro_f1=float(np.mean(f1_scores)),
        macro_f1_ci95=_ci95(f1_scores),
        balanced_accuracy=float(np.mean(balanced_scores)),
        balanced_accuracy_ci95=_ci95(balanced_scores),
        graph=graph_statistics.summary(),
    )


def _ci95(values):
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.


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
    extra_count = getattr(args, "dgcg_extra_references", None)
    train_bank = val_bank = test_bank = None
    context_banks = prepare_extra_banks(sampler, splits, args)
    extra_mode = args.extra_mode if context_banks is not None else "rankings"
    if context_banks is not None:
        train_bank, val_bank, test_bank = (context_banks[p] for p in ("train", "val", "test"))
        print(f"Extras: modo={extra_mode}; fases={','.join(args.extra_phases)}; "
              f"totais efetivos por episodio="
              f"{ {p: b.count for p, b in context_banks.items()} }; "
              f"reserva={args.extra_bank_per_class}/classe em cada split. "
              "Sem rotulos extras, sem pseudorrotulos, loss/metricas somente nas queries.")
    if extra_count is not None:
        val_bank = ExtraReferenceBank(sampler, splits["val"], count=extra_count,
                                     per_class=args.dgcg_extra_bank_per_class, seed=args.seed + 4202)
        test_bank = ExtraReferenceBank(sampler, splits["test"], count=extra_count,
                                      per_class=args.dgcg_extra_bank_per_class, seed=args.seed + 4303)
        print(f"Referencias extras na avaliacao: {extra_count} por episodio; "
              f"banco fixo de {args.dgcg_extra_bank_per_class} imagens por classe, "
              "excluido de suporte/consultas. Treino nao usa esse banco.")
    reference_evaluation = getattr(args, "dgcg_reference_train_rankings", False)
    use_global_rankings = (
        (getattr(args, "dgcg_train_global_rankings", False) or reference_evaluation)
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
        evaluation_context = (
            "validacao/teste usam treino + episodio atual." if reference_evaluation
            else "validacao/teste usam episodio + banco reservado." if val_bank is not None
            else "validacao/teste permanecem episodicos."
        )
        print(
            f"Rankings de treino: {training_rankings.rankings.shape[0]} imagens, "
            f"L={training_rankings.rankings.shape[1]}. "
            "Somente correlacoes do DGCG usam contexto global; "
            + evaluation_context
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
    best_val_metrics = None
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
    if use_global_rankings and reference_evaluation:
        graph_suffix = f"{graph_suffix}-reference-train-rankings"
    elif use_global_rankings:
        graph_suffix = f"{graph_suffix}-train-global-rankings"
    if extra_count is not None:
        graph_suffix += f"-extra{extra_count}-bank{args.dgcg_extra_bank_per_class}"
    if context_banks is not None:
        phases_tag = '-'.join(p for p in ("train", "val", "test") if p in args.extra_phases)
        graph_suffix += (f"-context{args.extra_images}-{extra_mode}-{phases_tag}"
                         f"-bank{args.extra_bank_per_class}")
    checkpoint_path = args.work_dir / f"best_{args.model}_{graph_suffix}.pt"
    # As medias moveis abaixo sao apenas para a barra de progresso; elas nao
    # interferem na loss otimizada nem na escolha do checkpoint.
    recent_losses: list[float] = []
    recent_accuracies: list[float] = []

    progress = trange(1, args.train_episodes + 1, desc="Meta-treino", unit="ep")
    for episode_index in progress:
        model.train()

        # Cada iteracao do otimizador corresponde a um novo episodio few-shot.
        sampled = (train_bank.sample(train_rng) if train_bank is not None
                   else sampler.sample(splits["train"], train_rng))
        episode = episode_to_device(sampled, device)
        graph_options = extra_graph_options(model, episode, train_bank, extra_mode)
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
                training_rankings=training_rankings if reference_evaluation else None,
                extra_reference_bank=val_bank,
                extra_mode=extra_mode,
            )
            print(
                f"\nEp. {episode_index:4d} | validacao: "
                f"{100 * val_metrics.accuracy:.2f}% +/- "
                f"{100 * val_metrics.ci95:.2f}% | loss={val_metrics.loss:.4f} | "
                f"f1_macro={100 * val_metrics.macro_f1:.2f}% | "
                f"balanced_accuracy={100 * val_metrics.balanced_accuracy:.2f}%"
            )
            if val_metrics.accuracy > best_val_accuracy:
                # Mantem uma copia em memoria e outra em disco. O teste final
                # usara exatamente este estado, nao o estado da ultima epoca.
                best_val_accuracy = val_metrics.accuracy
                best_val_metrics = val_metrics
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "input_dim": model.theta.in_features,
                        "args": vars(args),
                        "class_splits": splits,
                        "validation": asdict(val_metrics),
                        "extra_reference_banks": (
                            {p: b.metadata() for p, b in context_banks.items()}
                            if context_banks is not None else
                            {"val":val_bank.metadata(), "test":test_bank.metadata()}
                            if val_bank is not None else None
                        ),
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
        training_rankings=training_rankings if reference_evaluation else None,
        extra_reference_bank=test_bank,
        extra_mode=extra_mode,
    )
    # Saida estruturada: dispensa extrair metricas de barras de progresso/logs.
    result = {
        "schema_version": 2,
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "checkpoint": str(checkpoint_path), "class_splits": splits,
        "selection_metric": "validation.accuracy",
        "metric_aggregation": "mean_per_episode; values in [0,1]; normal CI95 across episodes",
        "graph_definition": "nonzero neighbors per adjacency row before self-loops and normalization",
        "validation": asdict(best_val_metrics), "test": asdict(test_metrics),
        "extra_context": ({
            "mode": extra_mode, "phases": args.extra_phases,
            "requested_total": args.extra_images,
            "banks": {p: b.metadata() for p, b in context_banks.items()},
            "labels_exposed_to_model": False, "pseudo_labels": False,
            "sampling": "same_episode_classes_balanced; labels_used_only_by_sampler",
            "supervision_and_metrics": "query_only; prototypes_from_support_only",
        } if context_banks is not None else None),
    }
    checkpoint_path.with_suffix(".metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    return model, test_metrics, checkpoint_path
