"""Nomes e mensagens dos experimentos, sem alterar modelos ou amostragem."""

from .config import DGCG_GRAPH_TYPES


def build_run_name(args, graph_type: str) -> str:
    """Nome historico usado nos logs e pelos scripts de consolidacao."""
    run_name = graph_type if args.model == "proto-sgc" else "sem-grafo"
    if args.model == "proto-sgc" and graph_type.startswith("knn"):
        run_name = f"{run_name}+{args.knn_metric}"
    if args.dgcg_reference_train_rankings and graph_type in DGCG_GRAPH_TYPES:
        run_name = f"{run_name}+reference-train-rankings"
    elif args.dgcg_train_global_rankings and graph_type in DGCG_GRAPH_TYPES:
        run_name = f"{run_name}+train-global-rankings"
    if args.dgcg_extra_references is not None:
        run_name += f"+extra{args.dgcg_extra_references}-bank{args.dgcg_extra_bank_per_class}"
    if args.extra_images is not None:
        run_name += f"+context{args.extra_images}-{args.extra_mode}-{'-'.join(args.extra_phases)}"
        if args.extra_source == "queries":
            run_name += f"-queries-pool{args.extra_bank_per_class}"
    if args.model == "proto-sgc" and args.cosine_rbf_weight:
        run_name = f"{run_name}+cosine-rbf"
    if args.model == "proto-sgc" and args.grande:
        run_name = f"{run_name}+grande-{args.grande_metric}"
    return run_name


def describe_run(args, graph_type, features_by_class, splits, run_name) -> None:
    """Mostra as escolhas efetivas do grafo e do contexto episodico."""
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
        train_extra = (args.extra_images or 0) if 'train' in args.extra_phases else 0
        train_graph_nodes = episode_nodes + (train_extra if args.extra_mode == 'propagation' else 0)
        effective_list_size = min(args.dgcg_list_size, train_graph_nodes)
        effective_candidate_k = min(
            args.dgcg_candidate_k,
            effective_list_size - 1,
            train_graph_nodes - 1,
        )
        # Candidatos continuam locais. Apenas os rankings usados na
        # correlacao podem ter mais itens que o episodio de treino.
        ranking_nodes = (
            sum(len(features_by_class[name]) for name in splits["train"])
            if (args.dgcg_train_global_rankings or args.dgcg_reference_train_rankings)
            else episode_nodes + train_extra
        )
        correlation_list_size = min(args.dgcg_list_size, ranking_nodes)
        effective_top_k = min(args.dgcg_top_k, correlation_list_size)
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
            f"L da correlacao no treino={correlation_list_size}, "
            f"candidatos={effective_candidate_k}, top-k={effective_top_k}, "
            f"limiar={threshold_description}{weighting_description}."
        )
        if args.dgcg_reference_train_rankings:
            print(
                "Rankings na validacao/teste: treino + episodio atual; "
                f"L={min(args.dgcg_list_size, ranking_nodes + episode_nodes)}. "
                "Somente os nos episodicos participam do grafo."
            )
        if args.dgcg_extra_references is not None:
            print(
                f"Rankings na validacao/teste: episodio + {args.dgcg_extra_references} extras; "
                f"L={min(args.dgcg_list_size, episode_nodes + args.dgcg_extra_references)}. "
                "Banco separado de suporte/consultas; sem propagacao nas extras."
            )
    if args.extra_images is not None:
        if args.extra_source == "queries":
            print(
                "Protocolo de contexto de queries: "
                f"{args.n_query} avaliada(s) + "
                f"{args.extra_images // args.n_way} nao rotulada(s) por classe; "
                f"pool mestre={args.n_query + args.extra_bank_per_class}/classe; "
                f"fases={','.join(args.extra_phases)}."
            )
        else:
            print(f"Protocolo de extras: {args.extra_images} totais, modo={args.extra_mode}, "
                  f"fases={','.join(args.extra_phases)}; reserva={args.extra_bank_per_class}/classe "
                  "em TODOS os splits para manter suporte/query pareados.")


def print_results(results, model_name: str) -> None:
    """Preserva o formato das linhas consumidas pelas baterias existentes."""
    print("\nResultado final em classes nunca vistas no meta-treino:")
    for run_name, test_metrics, checkpoint_path in results:
        print(
            f"  {model_name}/{run_name}: {100 * test_metrics.accuracy:.2f}% +/- "
            f"{100 * test_metrics.ci95:.2f}% (IC 95%) | "
            f"loss={test_metrics.loss:.4f}"
        )
        # Preserva a linha acima: scripts antigos extraem acuracia/loss dela.
        print(
            f"    f1_macro={100 * test_metrics.macro_f1:.2f}% +/- {100 * test_metrics.macro_f1_ci95:.2f}% | "
            f"balanced_accuracy={100 * test_metrics.balanced_accuracy:.2f}% "
            f"+/- {100 * test_metrics.balanced_accuracy_ci95:.2f}%"
        )
        print(f"    checkpoint: {checkpoint_path}")
        print(f"    metricas: {checkpoint_path.with_suffix('.metrics.json')}")
        graph = test_metrics.graph
        print(f"    grafo: grau medio={graph['degree_mean']:.3f}; "
              f"p05-p95=[{graph['degree_p05']:.1f}, {graph['degree_p95']:.1f}]; "
              f"densidade={graph['density_mean']:.4f}")
