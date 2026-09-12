"""Fluxo completo do experimento: dados → embeddings → episódios → modelo → treino → teste."""

from __future__ import annotations

import copy
from pathlib import Path

from .config import DGCG_GRAPH_TYPES, GRAPH_TYPES, parse_args, validate_hyperparameters
from .data import download_meta_album, load_local_meta_album
from .episodes import FeatureEpisodeSampler, split_classes
from .features import extract_or_load_features
from .model import ProtoSGC
from .runtime import seed_everything, select_device
from .training import Metrics, train_model


def main() -> None:
    """Orquestra dados, episodios, variantes de grafo, treino e relatorio."""

    # -----------------------------------------------------------------------
    # 1. Configuracao reproduzivel do experimento
    # -----------------------------------------------------------------------
    args = parse_args()
    validate_hyperparameters(args)
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    seed_everything(args.seed)
    device = select_device(args.device)

    # -----------------------------------------------------------------------
    # 2. Leitura das imagens e extracao (ou carga) dos embeddings congelados
    # -----------------------------------------------------------------------
    records = (
        load_local_meta_album(args.data_root)
        if args.data_root is not None
        else download_meta_album(args.openml_id, args.cache_dir)
    )
    features_by_class, input_dim = extract_or_load_features(records, args, device)
    # A divisao e feita por classe antes de qualquer amostragem episodica.
    class_names = sorted(features_by_class)
    splits = split_classes(class_names, args.n_way, args.class_split, args.seed)
    print(f"Classes ({len(class_names)}): treino/val/teste = "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    for split_name, names in splits.items():
        print(f"  {split_name:5s}: {', '.join(names)}")

    # -----------------------------------------------------------------------
    # 3. Amostrador comum às execucoes e validacao de opcoes incompatíveis
    # -----------------------------------------------------------------------
    sampler = FeatureEpisodeSampler(
        features_by_class,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
    )
    if args.model == "protonet" and args.graph_type == "all":
        raise ValueError("--graph-type all requer --model proto-sgc; protonet nao usa grafo.")
    if args.model == "protonet" and args.cosine_rbf_weight:
        raise ValueError("--cosine-rbf-weight requer --model proto-sgc.")
    if args.model == "protonet" and args.grande:
        raise ValueError("--grande requer --model proto-sgc.")

    # ``all`` expande para as topologias na ordem declarada em GRAPH_TYPES.
    graph_types = list(GRAPH_TYPES) if args.graph_type == "all" else [args.graph_type]
    results: list[tuple[str, Metrics, Path]] = []

    # -----------------------------------------------------------------------
    # 4. Meta-treino e meta-teste de cada topologia solicitada
    # -----------------------------------------------------------------------
    for graph_type in graph_types:
        # Reinicia a inicializacao e reutiliza as mesmas seeds episodicas para
        # que a comparacao entre grafos nao seja confundida por outra aleatoriedade.
        seed_everything(args.seed)
        # A copia rasa e suficiente: somente graph_type e substituido.
        run_args = copy.copy(args)
        run_args.graph_type = graph_type
        run_name = graph_type if args.model == "proto-sgc" else "sem-grafo"
        if args.model == "proto-sgc" and args.cosine_rbf_weight:
            run_name = f"{run_name}+cosine-rbf"
        if args.model == "proto-sgc" and args.grande:
            run_name = f"{run_name}+grande-{args.grande_metric}"
        print(f"\n=== Execucao: {args.model} / {run_name} ===")
        if args.model == "proto-sgc" and args.cosine_rbf_weight:
            print(
                "Pesos de aresta: RBF da distancia cosseno "
                f"(sigma={args.graph_temperature:g}); topologia preservada."
            )
        if args.model == "proto-sgc" and args.grande:
            metric_description = args.grande_metric
            if args.grande_metric == "rbo":
                metric_description = (
                    f"rbo (p={args.grande_rbo_p:g}, listas por "
                    f"{args.dgcg_metric})"
                )
            print(
                "Normalizacao: GRaNDe "
                f"(distancia={metric_description}, sigma={args.grande_sigma:g}); "
                "topologia e pesos preservados."
            )
        if args.model == "proto-sgc" and graph_type in DGCG_GRAPH_TYPES:
            # Estes valores sao apenas descritivos. A classe ProtoSGC refaz os
            # mesmos limites ao construir cada grafo episodico.
            episode_nodes = args.n_way * (args.n_shot + args.n_query)
            effective_list_size = min(args.dgcg_list_size, episode_nodes)
            effective_candidate_k = min(
                args.dgcg_candidate_k,
                effective_list_size - 1,
                episode_nodes - 1,
            )
            effective_top_k = min(args.dgcg_top_k, effective_list_size)
            if args.dgcg_threshold is not None:
                threshold_description = f"manual={args.dgcg_threshold:g}"
            elif args.dgcg_target_density is not None:
                density_low, density_high = args.dgcg_target_density
                threshold_description = (
                    f"automatico por densidade [{density_low:g}, {density_high:g}]"
                )
            else:
                degree_low = min(args.dgcg_target_degree[0], effective_candidate_k)
                degree_high = min(args.dgcg_target_degree[1], effective_candidate_k)
                density_low = degree_low / effective_candidate_k
                density_high = degree_high / effective_candidate_k
                threshold_description = (
                    f"automatico por grau medio [{degree_low:g}, {degree_high:g}] "
                    f"(densidade efetiva [{density_low:.3f}, {density_high:.3f}])"
                )
            graph_label = "DGCG+" if graph_type == "dgcg-plus" else "DGCG"
            if args.cosine_rbf_weight:
                weighting_description = ", pesos=RBF da distancia cosseno"
            elif graph_type == "dgcg-plus":
                weighting_description = ", pesos=vizinhanca mutua (media dos extremos)"
            else:
                weighting_description = ", sem pesos de aresta"
            print(
                f"{graph_label} efetivo por episodio: "
                f"metrica={args.dgcg_metric}, correlacao={args.dgcg_correlation}, "
                f"L={effective_list_size}, "
                f"candidatos={effective_candidate_k}, top-k={effective_top_k}, "
                f"limiar={threshold_description}{weighting_description}."
            )
        # Cada variante recebe uma nova instancia com a mesma inicializacao
        # aleatoria, garantida por seed_everything no inicio do laco.
        model = ProtoSGC(
            input_dim=input_dim,
            output_dim=args.feature_dim,
            model_name=args.model,
            knn=args.knn,
            graph_type=graph_type,
            sgc_hops=args.sgc_hops,
            graph_temperature=args.graph_temperature,
            grande=args.grande,
            grande_sigma=args.grande_sigma,
            grande_metric=args.grande_metric,
            grande_rbo_p=args.grande_rbo_p,
            cosine_rbf_weight=args.cosine_rbf_weight,
            dgcg_metric=args.dgcg_metric,
            dgcg_correlation=args.dgcg_correlation,
            dgcg_list_size=args.dgcg_list_size,
            dgcg_candidate_k=args.dgcg_candidate_k,
            dgcg_top_k=args.dgcg_top_k,
            dgcg_threshold=args.dgcg_threshold,
            dgcg_target_degree=args.dgcg_target_degree,
            dgcg_target_density=args.dgcg_target_density,
            dgcg_rbo_p=args.dgcg_rbo_p,
        )
        _, test_metrics, checkpoint_path = train_model(
            model, sampler, splits, run_args, device
        )
        results.append((run_name, test_metrics, checkpoint_path))

    # -----------------------------------------------------------------------
    # 5. Resumo comparavel das execucoes em classes nunca vistas
    # -----------------------------------------------------------------------
    print("\nResultado final em classes nunca vistas no meta-treino:")
    for run_name, test_metrics, checkpoint_path in results:
        print(
            f"  {args.model}/{run_name}: {100 * test_metrics.accuracy:.2f}% +/- "
            f"{100 * test_metrics.ci95:.2f}% (IC 95%) | "
            f"loss={test_metrics.loss:.4f}"
        )
        print(f"    checkpoint: {checkpoint_path}")
