"""Modelo ProtoNet/SGC: projeção treinável, propagação e classificação por protótipos.

Leia primeiro ``forward`` e ``encode_episode`` para entender o modelo.
Os métodos de grafo repassam suas opções às funções de ``graphs/``; as
assinaturas e os atributos originais são mantidos para uso em notebooks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DGCG_CORRELATIONS,
    DGCG_GRAPH_TYPES,
    DGCG_METRICS,
    GRANDE_METRICS,
    GRAPH_TYPES,
)
from .graphs import dgcg, grande, knn


class ProtoSGC(nn.Module):
    """ProtoNet sobre embeddings propagados por Simplified Graph Convolution.

    Para ``proto-sgc``, a representacao de cada no e

        H = normalize(LayerNorm(A_hat^K X Theta)).

    Para ``protonet``, A_hat^K e substituida pela identidade. O grafo e
    transdutivo e contem suporte e consultas, mas nao utiliza rotulos para
    criar arestas.
    """

    # Configuração e parâmetros treináveis.
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_name: str,
        knn: int,
        graph_type: str,
        sgc_hops: int,
        graph_temperature: float,
        grande: bool = False,
        grande_sigma: float = 0.2,
        grande_metric: str = "euclidean",
        grande_rbo_p: float = 0.9,
        cosine_rbf_weight: bool = False,
        dgcg_metric: str = "euclidean",
        dgcg_correlation: str = "rbo",
        dgcg_list_size: int = 200,
        dgcg_candidate_k: int = 200,
        dgcg_top_k: int = 40,
        dgcg_threshold: float | None = None,
        dgcg_target_degree: tuple[float, float] | list[float] = (4.0, 6.0),
        dgcg_target_density: tuple[float, float] | list[float] | None = None,
        dgcg_rbo_p: float = 0.9,
    ) -> None:
        """Configura a projecao aprendivel e todas as variantes de grafo.

        A maior parte dos testes abaixo repete as validacoes da linha de
        comando porque a classe tambem pode ser instanciada diretamente por
        outro codigo, sem passar por :func:`validate_hyperparameters`.
        """

        super().__init__()

        # Validacoes independentes dos dados: detectam configuracoes invalidas
        # antes que um episodio seja amostrado ou um tensor seja alocado.
        if knn < 1:
            raise ValueError("knn deve ser >= 1.")
        if sgc_hops < 1:
            raise ValueError("sgc-hops deve ser >= 1.")
        if cosine_rbf_weight and graph_temperature <= 0:
            raise ValueError("graph-temperature deve ser > 0.")
        if grande_sigma <= 0:
            raise ValueError("grande-sigma deve ser > 0.")
        if grande_metric not in GRANDE_METRICS:
            raise ValueError(
                f"grande-metric deve ser um de {GRANDE_METRICS}; "
                f"recebido: {grande_metric!r}."
            )
        if not 0.0 < grande_rbo_p < 1.0:
            raise ValueError("grande-rbo-p deve estar estritamente entre 0 e 1.")
        if graph_type not in GRAPH_TYPES:
            raise ValueError(
                f"graph-type deve ser um de {GRAPH_TYPES}; recebido: {graph_type!r}."
            )
        if dgcg_metric not in DGCG_METRICS:
            raise ValueError(
                f"dgcg-metric deve ser um de {DGCG_METRICS}; recebido: {dgcg_metric!r}."
            )
        if dgcg_correlation not in DGCG_CORRELATIONS:
            raise ValueError(
                "dgcg-correlation deve ser um de "
                f"{DGCG_CORRELATIONS}; recebido: {dgcg_correlation!r}."
            )
        if dgcg_list_size < 2 or dgcg_candidate_k < 1 or dgcg_top_k < 1:
            raise ValueError(
                "dgcg-list-size deve ser >= 2; dgcg-candidate-k e "
                "dgcg-top-k devem ser >= 1."
            )
        if dgcg_threshold is not None and not 0.0 <= dgcg_threshold <= 1.0:
            raise ValueError("dgcg-threshold deve estar em [0, 1].")
        degree_low, degree_high = dgcg_target_degree
        if not 0.0 < degree_low <= degree_high:
            raise ValueError(
                "dgcg-target-degree deve satisfazer 0 < MIN <= MAX."
            )
        if dgcg_target_density is not None:
            density_low, density_high = dgcg_target_density
            if not 0.0 < density_low <= density_high < 1.0:
                raise ValueError(
                    "dgcg-target-density deve satisfazer 0 < MIN <= MAX < 1."
                )
        if not 0.0 < dgcg_rbo_p < 1.0:
            raise ValueError("dgcg-rbo-p deve estar estritamente entre 0 e 1.")
        # Opcoes fixas do experimento. Elas nao sao parametros aprendiveis,
        # mas determinam como a adjacencia de cada episodio sera construida.
        self.model_name = model_name
        self.knn = knn
        self.graph_type = graph_type
        self.sgc_hops = sgc_hops
        self.graph_temperature = graph_temperature
        self.grande = grande
        self.grande_sigma = grande_sigma
        self.grande_metric = grande_metric
        self.grande_rbo_p = grande_rbo_p
        self.cosine_rbf_weight = cosine_rbf_weight
        self.dgcg_metric = dgcg_metric
        self.dgcg_correlation = dgcg_correlation
        self.dgcg_list_size = dgcg_list_size
        self.dgcg_candidate_k = dgcg_candidate_k
        self.dgcg_top_k = dgcg_top_k
        self.dgcg_threshold = dgcg_threshold
        self.dgcg_target_degree = (
            float(dgcg_target_degree[0]),
            float(dgcg_target_degree[1]),
        )
        self.dgcg_target_density = (
            None
            if dgcg_target_density is None
            else (
                float(dgcg_target_density[0]),
                float(dgcg_target_density[1]),
            )
        )
        self.dgcg_rbo_p = dgcg_rbo_p
        # Estes tres objetos compoem a parte treinavel. ``theta`` projeta os
        # atributos, LayerNorm estabiliza a representacao e logit_scale ajusta
        # a concentracao dos logits baseados em distancia.
        self.theta = nn.Linear(input_dim, output_dim, bias=False)
        self.normalization = nn.LayerNorm(output_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    # Fluxo de chamadas: forward → encode_episode → normalized_graph_adjacency.
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """Constroi prototipos do suporte e devolve logits para as consultas.

        Os vetores de consulta entram no grafo, mas ``query_y`` nem sequer e
        recebido por este metodo. Portanto, seus rotulos nao podem influenciar
        a propagacao, a formacao dos prototipos ou os logits.
        """

        n_support = support_x.shape[0]

        # A concatenacao e necessaria para construir um unico grafo transdutivo.
        embeddings = self.encode_episode(torch.cat((support_x, query_x), dim=0))
        support_embeddings = embeddings[:n_support]
        query_embeddings = embeddings[n_support:]

        # Prototipo da classe = centroide dos exemplos de suporte dessa classe.
        prototypes = []
        for class_id in range(n_way):
            class_support = support_embeddings[support_y == class_id]
            if len(class_support) == 0:
                raise ValueError(f"Classe episodica {class_id} sem exemplos de suporte.")
            prototypes.append(class_support.mean(dim=0))
        prototypes_tensor = torch.stack(prototypes, dim=0)

        # Distancias menores devem produzir logits maiores; por isso o sinal
        # negativo. A escala positiva e aprendida junto com Theta.
        squared_distances = torch.cdist(query_embeddings, prototypes_tensor).square()
        scale = self.logit_scale.exp().clamp(max=100.0)
        return -scale * squared_distances

    def encode_episode(self, features: torch.Tensor) -> torch.Tensor:
        """Produz uma representacao normalizada para todos os nos do episodio.

        Em ``proto-sgc``, suporte e consultas participam conjuntamente de
        ``A_hat^K X Theta``: este e o componente transdutivo do metodo. Em
        ``protonet``, nao ha adjacencia nem troca de informacao entre exemplos.
        """

        if self.model_name == "proto-sgc":
            if self.grande:
                # O GRaNDe oficial mede distancias em H^(0) = X Theta e as
                # atualiza juntamente com Theta durante o meta-treino.
                propagated = self.theta(features)
                adjacency = self.normalized_graph_adjacency(
                    graph_features=features,
                    degree_features=propagated,
                )
            else:
                # Sem GRaNDe, a adjacencia independe de Theta. Assim, a
                # propagacao pode ocorrer em X antes da projecao linear.
                propagated = features
                adjacency = self.normalized_graph_adjacency(features)

            # SGC remove ativacoes e pesos entre camadas: cada salto e apenas
            # uma nova multiplicacao pela mesma adjacencia normalizada.
            for _ in range(self.sgc_hops):
                propagated = adjacency @ propagated
            if not self.grande:
                propagated = self.theta(propagated)
        else:
            # Baseline ProtoNet: cada exemplo e projetado isoladamente.
            propagated = self.theta(features)

        # LayerNorm atua por vetor; a normalizacao L2 posterior permite
        # comparar as representacoes em uma escala comum.
        embeddings = self.normalization(propagated)
        return F.normalize(embeddings, p=2, dim=1)

    def normalized_graph_adjacency(
        self,
        graph_features: torch.Tensor,
        degree_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Adiciona autolacos e aplica normalizacao por graus de saida/entrada.

        Para grafos dirigidos, usa ``D_out^-1/2 A D_in^-1/2``. Nas variantes
        simetricas (uniao e reciproco), a expressao coincide com a normalizacao
        simetrica original ``D^-1/2 A D^-1/2``. Com GRaNDe, cada grau e
        substituido pelo grau central acrescido da penalidade gaussiana de sua
        vizinhanca. Assim como no ``GrandeSGConv`` oficial, as distancias do
        GRaNDe sao calculadas depois da transformacao linear; a topologia segue
        sendo construida a partir dos descritores de entrada. Nos grafos
        dirigidos, as vizinhancas de saida e entrada sao tratadas separadamente.
        """
        n_nodes = graph_features.shape[0]
        adjacency = self.graph_adjacency(graph_features)
        # Autolacos sao adicionados aqui, de forma identica para todas as
        # topologias e depois da eventual substituicao dos pesos das arestas.
        adjacency = adjacency + torch.eye(
            n_nodes, dtype=adjacency.dtype, device=adjacency.device
        )
        if self.grande:
            if degree_features is None:
                raise ValueError(
                    "degree_features e obrigatorio quando GRaNDe esta ativo."
                )
            correlations = (
                self.grande_rank_correlations(degree_features)
                if self.grande_metric == "rbo"
                else None
            )
            out_degree = self.grande_degree(
                degree_features, adjacency, correlations
            )
            in_degree = self.grande_degree(
                degree_features, adjacency.T, correlations
            )
        else:
            out_degree = adjacency.sum(dim=1)
            in_degree = adjacency.sum(dim=0)
        # Forma matricial: D_out^(-1/2) A D_in^(-1/2).
        out_degree_inv_sqrt = out_degree.clamp_min(1e-12).pow(-0.5)
        in_degree_inv_sqrt = in_degree.clamp_min(1e-12).pow(-0.5)
        return (
            out_degree_inv_sqrt[:, None]
            * adjacency
            * in_degree_inv_sqrt[None, :]
        )

    def graph_adjacency(self, features: torch.Tensor) -> torch.Tensor:
        """Despacha a topologia solicitada e, se pedido, substitui seus pesos."""

        if self.graph_type in DGCG_GRAPH_TYPES:
            adjacency = self.dgcg_adjacency(
                features,
                weighted=(
                    self.graph_type == "dgcg-plus"
                    and not self.cosine_rbf_weight
                ),
            )
        else:
            adjacency = self.knn_adjacency(features)
        # Esta etapa ocorre depois da topologia para garantir que o RBF nunca
        # crie nem remova arestas; ele apenas substitui valores nao nulos.
        if self.cosine_rbf_weight:
            adjacency = self.apply_cosine_rbf_weights(features, adjacency)
        return adjacency

    # Cálculos de grafo: as implementações estão em graphs/.
    def grande_rank_correlations(self, features: torch.Tensor) -> torch.Tensor:
        """RBO entre as listas ranqueadas dos vetores usados no grau GRaNDe.

        As listas seguem ``--dgcg-metric`` e os cortes ``--dgcg-list-size`` e
        ``--dgcg-top-k``, mas a correlacao e sempre RBO com ``--grande-rbo-p``,
        independentemente da correlacao escolhida para a topologia. A matriz e
        simetrica, entao serve para os graus de saida e de entrada.
        """
        list_size = min(self.dgcg_list_size, features.shape[0])
        return self.dgcg_correlation_matrix(
            self.ranked_lists(features, list_size),
            correlation="rbo",
            rbo_p=self.grande_rbo_p,
        )

    def grande_degree(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        correlations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calcula o GRaNDe de cada linha da adjacencia."""

        if self.grande_metric == "rbo" and correlations is None:
            correlations = self.grande_rank_correlations(features)
        return grande.grande_degree(
            features, adjacency, correlations,
            metric=self.grande_metric,
            sigma=self.grande_sigma,
        )

    def grande_edge_distances(
        self,
        features: torch.Tensor,
        sources: torch.Tensor,
        targets: torch.Tensor,
        correlations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Distancias das arestas usadas pelo grau GRaNDe."""

        if self.grande_metric == "rbo" and correlations is None:
            correlations = self.grande_rank_correlations(features)
        return grande.grande_edge_distances(
            features, sources, targets, correlations, metric=self.grande_metric,
        )

    def knn_adjacency(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Constroi a adjacencia kNN euclidiana e binaria, sem autolacos."""

        return knn.knn_adjacency(
            features,
            knn=self.knn,
            graph_type=self.graph_type,
        )

    @torch.no_grad()
    def apply_cosine_rbf_weights(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Aplica pesos RBF da distancia cosseno apenas nas arestas existentes."""

        return knn.apply_cosine_rbf_weights(
            features, adjacency, temperature=self.graph_temperature,
        )

    def dgcg_adjacency(
        self,
        features: torch.Tensor,
        weighted: bool = False,
    ) -> torch.Tensor:
        """Constroi a adjacencia do DGCG/DGCG+, sem autolacos."""

        return dgcg.dgcg_adjacency(
            features, weighted,
            metric=self.dgcg_metric,
            correlation=self.dgcg_correlation,
            list_size=self.dgcg_list_size,
            candidate_k=self.dgcg_candidate_k,
            top_k=self.dgcg_top_k,
            threshold=self.dgcg_threshold,
            target_degree=self.dgcg_target_degree,
            target_density=self.dgcg_target_density,
            rbo_p=self.dgcg_rbo_p,
            dtype=self.theta.weight.dtype,
        )

    @torch.no_grad()
    def ranked_lists(self, features: torch.Tensor, list_size: int) -> torch.Tensor:
        """Ordena os vizinhos de cada no segundo ``--dgcg-metric``."""

        return dgcg.build_ranked_lists(features, list_size, metric=self.dgcg_metric)

    def dgcg_correlation_matrix(
        self,
        ranked_lists: torch.Tensor,
        correlation: str | None = None,
        rbo_p: float | None = None,
    ) -> torch.Tensor:
        """Calcula RBO/Jaccard entre todos os pares de listas ranqueadas."""

        return dgcg.dgcg_correlation_matrix(
            ranked_lists,
            correlation=self.dgcg_correlation if correlation is None else correlation,
            rbo_p=self.dgcg_rbo_p if rbo_p is None else rbo_p,
            top_k=self.dgcg_top_k,
            dtype=self.theta.weight.dtype,
        )

    def dgcg_mutual_neighborhood_scores(
        self,
        ranked_lists: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula o escore por no usado como peso no DGCG+."""

        return dgcg.dgcg_mutual_neighborhood_scores(
            ranked_lists, top_k=self.dgcg_top_k, dtype=self.theta.weight.dtype,
        )

    def effective_dgcg_degree_interval(
        self,
        candidate_k: int,
    ) -> tuple[float, float]:
        """Converte o alvo configurado para grau medio no episodio atual."""

        return dgcg.effective_dgcg_degree_interval(
            candidate_k,
            target_degree=self.dgcg_target_degree,
            target_density=self.dgcg_target_density,
        )

    def automatic_dgcg_selection(
        self,
        candidate_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Seleciona o limiar observado mais proximo do intervalo de grau."""

        return dgcg.automatic_dgcg_selection(
            candidate_scores,
            target_degree=self.dgcg_target_degree,
            target_density=self.dgcg_target_density,
        )
