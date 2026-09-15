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


def parse_args() -> argparse.Namespace:
    """Define a interface de linha de comando e devolve seus argumentos.

    As opcoes sao separadas em quatro grupos para que ``--help`` tambem sirva
    como uma referencia rapida do experimento.
    """

    parser = argparse.ArgumentParser(
        description="ProtoNet + SGC transdutivo no Meta-Album Flowers Micro."
    )
    # 1) Origem das imagens, cache e extrator visual congelado.
    data = parser.add_argument_group("dados e backbone")
    data.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Pasta Meta-Album ja baixada, contendo labels.csv e images/. "
        "Se omitida, o Flowers Micro sera baixado via OpenML.",
    )
    data.add_argument(
        "--openml-id",
        type=int,
        default=FLOWERS_MICRO_OPENML_ID,
        help="ID OpenML; o padrao 44239 corresponde ao Flowers Micro.",
    )
    data.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/openml"),
        help="Cache do OpenML.",
    )
    data.add_argument(
        "--work-dir",
        type=Path,
        default=Path("runs/proto_sgc_meta_album"),
        help="Embeddings, checkpoint e resultados.",
    )
    data.add_argument("--image-size", type=int, default=128)
    data.add_argument("--batch-size", type=int, default=64)
    data.add_argument("--num-workers", type=int, default=0)
    data.add_argument(
        "--weights",
        choices=("imagenet", "random"),
        default="imagenet",
        help="Pesos da ResNet-18. Use random apenas para teste/offline.",
    )
    data.add_argument(
        "--recompute-features",
        action="store_true",
        help="Ignora o cache local de embeddings da ResNet.",
    )

    # 2) Formato N-way/K-shot dos episodios e divisao das classes.
    episode = parser.add_argument_group("episodios")
    episode.add_argument("--n-way", type=int, default=5)
    episode.add_argument("--n-shot", type=int, default=1)
    episode.add_argument("--n-query", type=int, default=5)
    episode.add_argument(
        "--class-split",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=None,
        help="Numero de classes por particao. Padrao para 20 classes: 10 5 5.",
    )

    # 3) Modelo, construcao de grafo, DGCG/DGCG+ e GRaNDe.
    graph = parser.add_argument_group("ProtoNet/SGC")
    graph.add_argument(
        "--model",
        choices=("proto-sgc", "protonet"),
        default="proto-sgc",
        help="proto-sgc usa A_hat^K X Theta; protonet usa X Theta.",
    )
    graph.add_argument("--feature-dim", type=int, default=256)
    graph.add_argument(
        "--knn",
        type=int,
        default=5,
        help=(
            "Numero de vizinhos das variantes kNN; "
            "nao e usado por DGCG/DGCG+."
        ),
    )
    graph.add_argument(
        "--knn-metric",
        choices=KNN_METRICS,
        default="euclidean",
        help="Metrica para selecionar vizinhos kNN; independente de --dgcg-metric. Pesos padrao binarios.",
    )
    graph.add_argument(
        "--graph-type",
        choices=(*GRAPH_TYPES, "all"),
        default="knn-union",
        help=(
            "Variante do grafo: knn-union reproduz o grafo atual (uniao com o "
            "transposto); knn-out usa i->kNN(i); knn-in inverte essas arestas; "
            "knn-reciprocal mantem apenas vizinhos mutuos; dgcg usa correlacao "
            "entre listas ranqueadas e limiar guiado por grau; dgcg-plus mantem "
            "essa topologia e pondera as arestas por vizinhanca mutua; all testa "
            "todas."
        ),
    )
    graph.add_argument("--sgc-hops", type=int, default=2)
    graph.add_argument(
        "--grande",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ativa GRaNDe na normalizacao de qualquer topologia. A flag "
            "--no-grande o desativa explicitamente (padrao: desativado)."
        ),
    )
    graph.add_argument(
        "--grande-sigma",
        type=float,
        default=0.2,
        help="Sigma do kernel gaussiano do GRaNDe (padrao oficial: 0.2).",
    )
    graph.add_argument(
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
    graph.add_argument(
        "--grande-rbo-p",
        type=float,
        default=0.9,
        help=(
            "Persistencia p do RBO do GRaNDe, usada apenas com "
            "--grande-metric rbo. As listas ranqueadas seguem --dgcg-metric e "
            "os cortes --dgcg-list-size/--dgcg-top-k, mesmo em topologias kNN."
        ),
    )
    graph.add_argument(
        "--cosine-rbf-weight",
        action="store_true",
        help=(
            "Ablacao opcional: substitui os valores das arestas existentes por "
            "exp(-(1-cosseno)^2 / (2*sigma^2)), sem alterar as arestas; "
            "--graph-temperature define sigma."
        ),
    )
    graph.add_argument(
        "--graph-temperature",
        type=float,
        default=0.2,
        help=(
            "Sigma usado somente pela ablacao --cosine-rbf-weight. "
            "Nao interfere nos grafos padrao."
        ),
    )
    graph.add_argument(
        "--dgcg-metric",
        choices=DGCG_METRICS,
        default="euclidean",
        help=(
            "Metrica que gera os rankings do DGCG. Euclidean e o padrao e "
            "reproduz as ranked lists geradas pelo BallTree original."
        ),
    )
    graph.add_argument(
        "--dgcg-train-global-rankings",
        action="store_true",
        help=(
            "Usa rankings de todas as imagens de treino nas correlacoes do DGCG/DGCG+. "
            "Candidatos, limiar e pesos DGCG+ continuam episodicos. "
            "Validacao e teste usam somente rankings do episodio."
        ),
    )
    graph.add_argument(
        "--dgcg-correlation",
        choices=DGCG_CORRELATIONS,
        default="rbo",
        help="Correlacao entre rankings usada pelo DGCG.",
    )
    graph.add_argument(
        "--dgcg-list-size",
        type=int,
        default=200,
        help="Tamanho maximo L; sempre limitado ao numero de nos do episodio.",
    )
    graph.add_argument(
        "--dgcg-candidate-k",
        type=int,
        default=200,
        help="Numero maximo de candidatos; sempre limitado a L-1.",
    )
    graph.add_argument(
        "--dgcg-top-k",
        type=int,
        default=40,
        help="Profundidade das listas na correlacao (padrao oficial: 40).",
    )
    graph.add_argument(
        "--dgcg-threshold",
        type=float,
        default=None,
        help="Limiar manual do DGCG. Se omitido, usa busca guiada por grau medio.",
    )
    graph.add_argument(
        "--dgcg-target-degree",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(4.0, 6.0),
        help=(
            "Intervalo-alvo do grau medio no modo automatico (padrao: 4--6)."
        ),
    )
    graph.add_argument(
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
    graph.add_argument(
        "--dgcg-rbo-p",
        type=float,
        default=0.9,
        help="Persistencia p do RBO (padrao oficial: 0.9).",
    )
    # 4) Quantidade de episodios, otimizador, semente e dispositivo.
    train = parser.add_argument_group("otimizacao e avaliacao")
    train.add_argument("--train-episodes", type=int, default=2000)
    train.add_argument("--val-episodes", type=int, default=200)
    train.add_argument("--test-episodes", type=int, default=1000)
    train.add_argument("--val-every", type=int, default=200)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    train.add_argument(
        "--quick",
        action="store_true",
        help="Executa 200/50/200 episodios para um teste rapido.",
    )
    args = parser.parse_args()

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

    if getattr(args, "dgcg_train_global_rankings", False) and (
        args.model != "proto-sgc" or args.graph_type not in (*DGCG_GRAPH_TYPES, "all")
    ):
        raise ValueError("--dgcg-train-global-rankings requer proto-sgc com dgcg, dgcg-plus ou all.")
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
