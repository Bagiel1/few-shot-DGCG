"""Fluxo completo do experimento: dados → embeddings → episódios → modelo → treino → teste."""

from __future__ import annotations

import copy
from pathlib import Path

from .config import GRAPH_TYPES, parse_args, validate_hyperparameters
from .data import download_meta_album, load_local_meta_album
from .episodes import FeatureEpisodeSampler, split_classes
from .features import extract_or_load_features, feature_spec
from .model import ProtoSGC
from .reporting import build_run_name, describe_run, print_results
from .runtime import seed_everything, select_device
from .training import Metrics, train_model, evaluate_without_training


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
    args.feature_spec = feature_spec(args)
    args.feature_cache_dir = str(args.work_dir)
    # Compartilha features entre os modos, mas separa resultados e checkpoints.
    mode = "train-free" if args.train_free else "trained"
    prep = args.feature_spec["preprocessing"]
    # As baterias antigas procuram JSONs diretamente no work-dir da ResNet/128.
    if args.backbone != "resnet18" or args.image_size != 128 or args.train_free:
        args.work_dir = args.work_dir / (
            f"{args.backbone}_{args.weights}_{args.image_size}px_{prep}"
        ) / mode / f"{args.n_way}w{args.n_shot}s{args.n_query}q_seed{args.seed}"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.train_free:
        print(f"Sem treino adicional: Theta identidade {input_dim}x{input_dim}; "
              "LayerNorm e escala fixas. Nenhum otimizador sera criado.")
        args.feature_dim = input_dim
    else:
        print(f"Backbone congelado; treino de Theta {input_dim}->{args.feature_dim}, "
              "LayerNorm e escala dos logits.")
    print(f"Resultados desta execucao: {args.work_dir}")
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
        name = build_run_name(args, graph_type)
        describe_run(args, graph_type, features_by_class, splits, name)
        # Cada variante recebe uma nova instancia com a mesma inicializacao
        # aleatoria, garantida por seed_everything no inicio do laco.
        model = ProtoSGC(
            input_dim=input_dim,
            output_dim=args.feature_dim,
            model_name=args.model,
            knn=args.knn,
            knn_metric=args.knn_metric,
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
        runner = evaluate_without_training if args.train_free else train_model
        _, test_metrics, checkpoint_path = runner(
            model, sampler, splits, run_args, device
        )
        results.append((name, test_metrics, checkpoint_path))

    # -----------------------------------------------------------------------
    # 5. Resumo comparavel das execucoes em classes nunca vistas
    # -----------------------------------------------------------------------
    print_results(results, args.model)
