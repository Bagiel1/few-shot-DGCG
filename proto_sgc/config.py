"""Opções da linha de comando, constantes e validação dos hiperparâmetros."""

from __future__ import annotations

import argparse
from pathlib import Path


# Dataset usado quando nenhum --data-root e informado.
FLOWERS_MICRO_OPENML_ID = 44239

# Extensoes reconhecidas ao procurar as imagens baixadas/localmente salvas.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# As duas variantes compartilham a mesma topologia; DGCG+ acrescenta pesos.
DGCG_GRAPH_TYPES = ("dgcg", "dgcg-plus")

# Mutual e reciprocal sao a mesma intersecao; nao duplicar no modo all.
GRAPH_ALIASES = ("knn-mutual",)

# ``all`` nao aparece aqui porque e uma instrucao para executar, em sequencia,
# todas as topologias abaixo; ele nao representa uma topologia propriamente dita.
GRAPH_TYPES = (
    "knn-union",
    "knn-out",
    "knn-in",
    "knn-reciprocal",
    *DGCG_GRAPH_TYPES,
)
# DGCG adaptado de https://github.com/Bagiel1/DGCG (Brito e Valem, 2025).
# Peso de vizinhanca mutua do DGCG+ adaptado de
# https://github.com/Bagiel1/DGCGplus.
# GRaNDe adaptado de https://github.com/rduarte12/SBBD-2026_GRaNDe.
DGCG_CORRELATIONS = ("rbo", "jaccardk", "jaccard-median", "jaccard-max")
DGCG_METRICS = ("cosine", "euclidean")
KNN_METRICS = ("euclidean", "cosine")
# Distancia usada nas arestas pelo GRaNDe; euclidean reproduz o codigo oficial.
GRANDE_METRICS = ("euclidean", "cosine", "rbo")


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Registra as opcoes de dados e backbone."""
    group = parser.add_argument_group("dados e backbone")
    group.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Pasta Meta-Album ja baixada, contendo labels.csv e images/. "
        "Se omitida, o Flowers Micro sera baixado via OpenML.",
    )
    group.add_argument(
        "--openml-id",
        type=int,
        default=FLOWERS_MICRO_OPENML_ID,
        help="ID OpenML; o padrao 44239 corresponde ao Flowers Micro.",
    )
    group.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/openml"),
        help="Cache do OpenML.",
    )
    group.add_argument(
        "--work-dir",
        type=Path,
        default=Path("runs/proto_sgc_meta_album"),
        help="Embeddings, checkpoint e resultados.",
    )
    group.add_argument("--backbone", choices=("resnet18", "vit-b-16"), default="resnet18")
    group.add_argument("--image-size", type=int, default=None,
                      help="ResNet: padrao 128. ViT-B/16: somente 224 (padrao).")
    group.add_argument("--batch-size", type=int, default=64)
    group.add_argument("--num-workers", type=int, default=0)
    group.add_argument(
        "--weights",
        choices=("imagenet", "random"),
        default="imagenet",
        help="imagenet usa IMAGENET1K_V1 do backbone escolhido; random somente para teste/offline.",
    )
    group.add_argument(
        "--recompute-features",
        action="store_true",
        help="Ignora o cache local de embeddings do backbone.",
    )


def _add_episode_arguments(parser: argparse.ArgumentParser) -> None:
    """Registra as opcoes de episodios."""
    group = parser.add_argument_group("episodios")
    group.add_argument("--n-way", type=int, default=5)
    group.add_argument("--n-shot", type=int, default=1)
    group.add_argument("--n-query", type=int, default=5)
    group.add_argument(
        "--class-split",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=None,
        help="Numero de classes por particao. Padrao para 20 classes: 10 5 5.",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Registra as opcoes de ProtoNet/SGC."""
    group = parser.add_argument_group("ProtoNet/SGC")
    group.add_argument(
        "--model",
        choices=("proto-sgc", "protonet"),
        default="proto-sgc",
        help="proto-sgc usa A_hat^K X Theta; protonet usa X Theta.",
    )
    group.add_argument("--feature-dim", type=int, default=256)
    group.add_argument(
        "--knn",
        type=int,
        default=5,
        help=(
            "Numero de vizinhos das variantes kNN; "
            "nao e usado por DGCG/DGCG+."
        ),
    )
    group.add_argument(
        "--knn-metric",
        choices=KNN_METRICS,
        default="euclidean",
        help="Metrica para selecionar vizinhos kNN; independente de --dgcg-metric. Pesos padrao binarios.",
    )
    group.add_argument(
        "--graph-type",
        choices=(*GRAPH_TYPES, *GRAPH_ALIASES, "all"),
        default="knn-union",
        help=(
            "Variante do grafo: knn-union reproduz o grafo atual (uniao com o "
            "transposto); knn-out usa i->kNN(i); knn-in inverte essas arestas; "
            "knn-reciprocal (alias knn-mutual) mantem apenas vizinhos mutuos; dgcg usa correlacao "
            "entre listas ranqueadas e limiar guiado por grau; dgcg-plus mantem "
            "essa topologia e pondera as arestas por vizinhanca mutua; all testa "
            "todas."
        ),
    )
    group.add_argument("--sgc-hops", type=int, default=2)


def _add_dgcg_arguments(parser: argparse.ArgumentParser) -> None:
    """Registra as opcoes de DGCG."""
    group = parser.add_argument_group("DGCG")
    group.add_argument(
        "--dgcg-metric",
        choices=DGCG_METRICS,
        default="euclidean",
        help=(
            "Metrica que gera os rankings do DGCG. Euclidean e o padrao e "
            "reproduz as ranked lists geradas pelo BallTree original."
        ),
    )
    group.add_argument(
        "--dgcg-correlation",
        choices=DGCG_CORRELATIONS,
        default="rbo",
        help="Correlacao entre rankings usada pelo DGCG.",
    )
    group.add_argument(
        "--dgcg-list-size",
        type=int,
        default=200,
        help="Tamanho maximo L; limitado ao universo de imagens usado nos rankings.",
    )
    group.add_argument(
        "--dgcg-candidate-k",
        type=int,
        default=200,
        help="Numero maximo de candidatos; sempre limitado a L-1.",
    )
    group.add_argument(
        "--dgcg-top-k",
        type=int,
        default=40,
        help="Profundidade das listas na correlacao (padrao oficial: 40).",
    )
    group.add_argument(
        "--dgcg-threshold",
        type=float,
        default=None,
        help="Limiar manual do DGCG. Se omitido, usa busca guiada por grau medio.",
    )
    group.add_argument(
        "--dgcg-target-degree",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(4.0, 6.0),
        help=(
            "Intervalo-alvo do grau medio no modo automatico (padrao: 4--6)."
        ),
    )
    group.add_argument(
        "--dgcg-target-density",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Densidade absoluta opcional; se informada, substitui "
            "--dgcg-target-degree. Util para reproduzir o protocolo nao episodico."
        ),
    )
    group.add_argument(
        "--dgcg-rbo-p",
        type=float,
        default=0.9,
        help="Persistencia p do RBO (padrao oficial: 0.9).",
    )


def _add_context_arguments(parser: argparse.ArgumentParser) -> None:
    """Registra as opcoes de contexto e referencias extras."""
    group = parser.add_argument_group("contexto e referencias extras")
    group.add_argument(
        "--dgcg-train-global-rankings",
        action="store_true",
        help=(
            "Usa rankings de todas as imagens de treino nas correlacoes do DGCG/DGCG+. "
            "Candidatos, limiar e pesos DGCG+ continuam episodicos. "
            "Validacao e teste usam somente rankings do episodio."
        ),
    )
    group.add_argument(
        "--dgcg-reference-train-rankings",
        action="store_true",
        help=(
            "Treino: rankings contra todas as imagens de treino. Validacao/teste: "
            "rankings contra treino + somente o episodio atual. "
            "Grafo e candidatos continuam episodicos; sem rotulos nas correlacoes."
        ),
    )
    group.add_argument(
        "--dgcg-extra-references",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Validacao/teste: N referencias nao rotuladas das classes do episodio, "
            "reservadas fora de suporte/consultas; somente nos rankings. "
            "Omitir desativa. N=0 e o controle com o mesmo banco reservado."
        ),
    )
    group.add_argument(
        "--dgcg-extra-bank-per-class",
        type=int,
        default=20,
        help="Imagens reservadas por classe na avaliacao com referencias extras (default 20).",
    )
    group.add_argument(
        "--extra-images", type=int, default=None, metavar="N",
        help="Novo protocolo: N extras totais por episodio. 0 reserva os mesmos bancos sem usar extras.",
    )
    group.add_argument(
        "--extra-mode", choices=("rankings", "propagation"), default="rankings",
        help="rankings: somente correlacoes DGCG; propagation: extras tambem sao nos do grafo.",
    )
    group.add_argument(
        "--extra-phases", nargs="+", choices=("train", "val", "test"),
        default=["train", "val", "test"],
        help="Fases que usam extras. Bancos separados por split; nunca mistura classes entre fases.",
    )
    group.add_argument(
        "--extra-bank-per-class", type=int, default=20,
        help=(
            "Com --extra-source reserved, reserva fixa por classe. Com queries, "
            "define o pool fixo de queries de contexto por classe."
        ),
    )
    group.add_argument(
        "--extra-source", choices=("reserved", "queries"), default="reserved",
        help=(
            "reserved usa o banco separado antigo; queries separa as extras das "
            "queries sorteadas no mesmo episodio mestre."
        ),
    )


def _add_weighting_arguments(parser: argparse.ArgumentParser) -> None:
    """Registra as opcoes de ponderacao e GRaNDe."""
    group = parser.add_argument_group("ponderacao e GRaNDe")
    group.add_argument(
        "--grande",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ativa GRaNDe na normalizacao de qualquer topologia. A flag "
            "--no-grande o desativa explicitamente (padrao: desativado)."
        ),
    )
    group.add_argument(
        "--grande-sigma",
        type=float,
        default=0.2,
        help="Sigma do kernel gaussiano do GRaNDe (padrao oficial: 0.2).",
    )
    group.add_argument(
        "--grande-metric",
        choices=GRANDE_METRICS,
        default="euclidean",
        help=(
            "Distancia das arestas usada pelo GRaNDe. euclidean reproduz o "
            "codigo oficial; cosine usa 1 - cosseno entre os mesmos vetores; "
            "rbo usa 1 - RBO entre as listas ranqueadas desses vetores. Em "
            "todos os casos o min-max global e o kernel gaussiano nao mudam."
        ),
    )
    group.add_argument(
        "--grande-rbo-p",
        type=float,
        default=0.9,
        help=(
            "Persistencia p do RBO do GRaNDe, usada apenas com "
            "--grande-metric rbo. As listas ranqueadas seguem --dgcg-metric e "
            "os cortes --dgcg-list-size/--dgcg-top-k, mesmo em topologias kNN."
        ),
    )
    group.add_argument(
        "--cosine-rbf-weight",
        action="store_true",
        help=(
            "Ablacao opcional: substitui os valores das arestas existentes por "
            "exp(-(1-cosseno)^2 / (2*sigma^2)), sem alterar as arestas; "
            "--graph-temperature define sigma."
        ),
    )
    group.add_argument(
        "--graph-temperature",
        type=float,
        default=0.2,
        help=(
            "Sigma usado somente pela ablacao --cosine-rbf-weight. "
            "Nao interfere nos grafos padrao."
        ),
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Registra as opcoes de otimizacao e avaliacao."""
    group = parser.add_argument_group("otimizacao e avaliacao")
    group.add_argument("--train-free", action="store_true",
                       help="Sem otimizacao: Theta identidade na dimensao do backbone; LayerNorm e escala fixas.")
    group.add_argument("--train-episodes", type=int, default=2000)
    group.add_argument("--val-episodes", type=int, default=200)
    group.add_argument("--test-episodes", type=int, default=1000)
    group.add_argument("--val-every", type=int, default=200)
    group.add_argument("--learning-rate", type=float, default=1e-3)
    group.add_argument("--weight-decay", type=float, default=1e-4)
    group.add_argument("--grad-clip", type=float, default=5.0)
    group.add_argument("--seed", type=int, default=7)
    group.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    group.add_argument(
        "--quick",
        action="store_true",
        help="Executa 200/50/200 episodios para um teste rapido.",
    )


def parse_args() -> argparse.Namespace:
    """Le os argumentos; cada familia de opcoes e definida separadamente."""
    parser = argparse.ArgumentParser(
        description="ProtoNet + SGC transdutivo no Meta-Album Flowers Micro."
    )
    _add_data_arguments(parser)
    _add_episode_arguments(parser)
    _add_model_arguments(parser)
    _add_dgcg_arguments(parser)
    _add_context_arguments(parser)
    _add_weighting_arguments(parser)
    _add_training_arguments(parser)

    args = parser.parse_args()
    if args.image_size is None:
        args.image_size = 224 if args.backbone == "vit-b-16" else 128

    # ``--quick`` altera apenas o custo do experimento; todo o pipeline e
    # mantido, o que o torna util como teste de funcionamento ponta a ponta.
    if args.quick:
        args.train_episodes = 200
        args.val_episodes = 50
        args.test_episodes = 200
        args.val_every = 50
    return args


def validate_hyperparameters(args: argparse.Namespace) -> None:
    """Valida relacoes numericas que o argparse nao expressa sozinho."""

    if getattr(args, "backbone", "resnet18") == "vit-b-16" and args.image_size != 224:
        raise ValueError("ViT-B/16 usa entrada 224x224; remova --image-size ou informe 224.")
    if getattr(args, "train_free", False) and (
        getattr(args, "extra_images", None) is not None
        or getattr(args, "dgcg_extra_references", None) is not None
        or args.dgcg_train_global_rankings or args.dgcg_reference_train_rankings
    ):
        raise ValueError("--train-free nesta comparacao usa episodios padrao, sem extras/rankings globais.")
    context_count = getattr(args, "extra_images", None)
    if context_count is not None:
        if context_count < 0 or args.extra_bank_per_class < 1:
            raise ValueError("extra-images deve ser >= 0 e extra-bank-per-class >= 1.")
        if context_count > args.n_way * args.extra_bank_per_class:
            raise ValueError("Extras excedem n-way * extra-bank-per-class.")
        if len(set(args.extra_phases)) != len(args.extra_phases):
            raise ValueError("extra-phases nao aceita fases repetidas.")
        if (getattr(args, "dgcg_extra_references", None) is not None
                or args.dgcg_train_global_rankings or args.dgcg_reference_train_rankings):
            raise ValueError("Nao misture --extra-images com os modos legados de referencias/rankings globais.")
        if context_count and args.model != "proto-sgc":
            raise ValueError("Extras positivas requerem proto-sgc; no-graph aceita apenas --extra-images 0.")
        if context_count and args.extra_mode == "rankings" and args.graph_type not in DGCG_GRAPH_TYPES:
            raise ValueError("Extras somente nos rankings requerem dgcg ou dgcg-plus.")
        if args.extra_source == "queries":
            if context_count % args.n_way:
                raise ValueError("Extras vindas de queries devem ser divisiveis por n-way.")
            if args.extra_mode != "propagation":
                raise ValueError("Extras vindas de queries usam somente o modo propagation.")
    extra = getattr(args, "dgcg_extra_references", None)
    if extra is not None:
        if args.model != "proto-sgc" or args.graph_type not in DGCG_GRAPH_TYPES:
            raise ValueError("--dgcg-extra-references requer proto-sgc com dgcg ou dgcg-plus.")
        if extra < 0 or args.dgcg_extra_bank_per_class < 1:
            raise ValueError("Referencias extras devem ser >= 0; banco por classe deve ser >= 1.")
        if extra > args.n_way * args.dgcg_extra_bank_per_class:
            raise ValueError("Referencias extras excedem n-way * dgcg-extra-bank-per-class.")
        if getattr(args, "dgcg_reference_train_rankings", False):
            raise ValueError("Escolha referencias extras OU referencias de treino na avaliacao.")

    if (getattr(args, "dgcg_train_global_rankings", False) or
        getattr(args, "dgcg_reference_train_rankings", False)) and (
        args.model != "proto-sgc" or args.graph_type not in (*DGCG_GRAPH_TYPES, "all")
    ):
        raise ValueError("Rankings de treino requerem proto-sgc com dgcg, dgcg-plus ou all.")
    # ``argparse`` valida tipos e escolhas; limites como > 0 precisam ser
    # verificados explicitamente depois da leitura.
    positive_ints = {
        "image-size": args.image_size,
        "batch-size": args.batch_size,
        "n-way": args.n_way,
        "n-shot": args.n_shot,
        "n-query": args.n_query,
        "feature-dim": args.feature_dim,
        "knn": args.knn,
        "dgcg-candidate-k": args.dgcg_candidate_k,
        "dgcg-top-k": args.dgcg_top_k,
        "train-episodes": args.train_episodes,
        "val-episodes": args.val_episodes,
        "test-episodes": args.test_episodes,
        "val-every": args.val_every,
    }
    invalid = {name: value for name, value in positive_ints.items() if value < 1}
    if invalid:
        raise ValueError(f"Estes parametros devem ser >= 1: {invalid}")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("learning-rate e grad-clip devem ser > 0; weight-decay deve ser >= 0.")
    if args.cosine_rbf_weight and args.graph_temperature <= 0:
        raise ValueError("graph-temperature deve ser > 0.")
    if args.grande_sigma <= 0:
        raise ValueError("grande-sigma deve ser > 0.")
    if not 0.0 < args.grande_rbo_p < 1.0:
        raise ValueError("grande-rbo-p deve estar estritamente entre 0 e 1.")
    if args.dgcg_list_size < 2:
        raise ValueError("dgcg-list-size deve ser >= 2.")
    if args.dgcg_threshold is not None and not 0.0 <= args.dgcg_threshold <= 1.0:
        raise ValueError("dgcg-threshold deve estar em [0, 1].")
    degree_low, degree_high = args.dgcg_target_degree
    if not 0.0 < degree_low <= degree_high:
        raise ValueError(
            "dgcg-target-degree deve satisfazer 0 < MIN <= MAX."
        )
    if args.dgcg_target_density is not None:
        density_low, density_high = args.dgcg_target_density
        if not 0.0 < density_low <= density_high < 1.0:
            raise ValueError(
                "dgcg-target-density deve satisfazer 0 < MIN <= MAX < 1."
            )
    if not 0.0 < args.dgcg_rbo_p < 1.0:
        raise ValueError("dgcg-rbo-p deve estar estritamente entre 0 e 1.")
