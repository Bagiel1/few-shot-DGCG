"""Trechos reais do projeto para relacionar código, matemática e visualização.

O catálogo é fechado: a API não aceita caminhos de arquivos do navegador.
Os limites são encontrados pela AST, mantendo os números de linha corretos
quando comentários ou outras funções mudam de posição no arquivo.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Excerpt:
    path: str
    symbol: str
    title: str
    first: str | None = None
    last: str | None = None


CATALOG = {
    "sample": Excerpt("proto_sgc/episodes.py", "FeatureEpisodeSampler.sample", "Escolha de classes, suporte e consultas", "selected =", "return Episode("),
    "split": Excerpt("proto_sgc/episodes.py", "split_classes", "Separação das classes de treino, validação e teste"),
    "synthetic": Excerpt("simulador/engine.py", "synthetic_features", "Embeddings sintéticos usados no simulador"),
    "cache": Excerpt("simulador/engine.py", "load_features", "Leitura dos embeddings reais em cache"),
    "concatenate": Excerpt("proto_sgc/model.py", "ProtoSGC.forward", "Suporte e consultas no mesmo episódio", "n_support =", "query_embeddings ="),
    "graph_dispatch": Excerpt("proto_sgc/model.py", "ProtoSGC.graph_adjacency", "Escolha da topologia e aplicação dos pesos"),
    "knn": Excerpt("proto_sgc/graphs/knn.py", "knn_adjacency", "Construção das topologias kNN"),
    "dgcg": Excerpt("proto_sgc/graphs/dgcg.py", "dgcg_adjacency", "Construção do DGCG / DGCG+"),
    "rankings": Excerpt("proto_sgc/graphs/dgcg.py", "build_ranked_lists", "Listas ranqueadas dos vizinhos"),
    "correlation": Excerpt("proto_sgc/graphs/dgcg.py", "dgcg_correlation_matrix", "Correlação RBO / Jaccard"),
    "threshold": Excerpt("proto_sgc/graphs/dgcg.py", "automatic_dgcg_selection", "Escolha automática do limiar"),
    "mutual": Excerpt("proto_sgc/graphs/dgcg.py", "dgcg_mutual_neighborhood_scores", "Pesos de vizinhança mútua do DGCG+"),
    "rbf": Excerpt("proto_sgc/graphs/knn.py", "apply_cosine_rbf_weights", "Substituição dos pesos por RBF de cosseno"),
    "normalize_graph": Excerpt("proto_sgc/model.py", "ProtoSGC.normalized_graph_adjacency", "Autolaços e normalização da adjacência", "n_nodes =", "return ("),
    "grande_degree": Excerpt("proto_sgc/graphs/grande.py", "grande_degree", "Cálculo do grau GRaNDe"),
    "grande_distances": Excerpt("proto_sgc/graphs/grande.py", "grande_edge_distances", "Distâncias usadas pelo GRaNDe"),
    "encode": Excerpt("proto_sgc/model.py", "ProtoSGC.encode_episode", "Projeção, saltos e representação final", 'if self.model_name == "proto-sgc":', "return F.normalize("),
    "normalization": Excerpt("proto_sgc/model.py", "ProtoSGC.encode_episode", "LayerNorm e normalização L2", "embeddings = self.normalization", "return F.normalize("),
    "parameters": Excerpt("proto_sgc/model.py", "ProtoSGC.__init__", "Criação dos parâmetros treináveis", "self.theta =", "self.logit_scale ="),
    "prototypes": Excerpt("proto_sgc/model.py", "ProtoSGC.forward", "Protótipos calculados somente com o suporte", "prototypes =", "prototypes_tensor ="),
    "prediction": Excerpt("proto_sgc/model.py", "ProtoSGC.forward", "Distâncias e logits das consultas", "squared_distances =", "return -scale"),
    "train_step": Excerpt("proto_sgc/training.py", "train_model", "Loss, backward, clipping e atualização", "optimizer.zero_grad", "optimizer.step"),
    "loss": Excerpt("proto_sgc/training.py", "train_model", "Entropia cruzada das consultas", "loss = F.cross_entropy"),
    "optimizer": Excerpt("proto_sgc/training.py", "train_model", "Configuração do AdamW", "optimizer = torch.optim.AdamW"),
    "observed_step": Excerpt("simulador/engine.py", "simulate", "Mesmo passo, com captura dos gradientes no simulador", "optimizer.zero_grad", "optimizer.step"),
    "after_update": Excerpt("simulador/engine.py", "simulate", "Observação depois do passo e referência fixa", "after = inspect_episode", "probe_state = inspect_episode"),
    "evaluate": Excerpt("proto_sgc/training.py", "evaluate", "Avaliação e intervalo de confiança"),
    "validation": Excerpt("proto_sgc/training.py", "train_model", "Seleção do melhor checkpoint pela validação", "if validate_now:"),
    "restore": Excerpt("proto_sgc/training.py", "train_model", "Restauração do melhor estado e teste final", "model.load_state_dict", "test_metrics = evaluate"),
}


def code_catalog() -> dict[str, dict]:
    """Lê os arquivos atuais e extrai os trechos, sem executar seu conteúdo."""
    documents = {}
    result = {}
    for identifier, excerpt in CATALOG.items():
        if excerpt.path not in documents:
            source = (ROOT / excerpt.path).read_text(encoding="utf-8")
            documents[excerpt.path] = (source.splitlines(), ast.parse(source))
        lines, node = documents[excerpt.path]
        for name in excerpt.symbol.split("."):
            node = next(child for child in node.body if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.name == name)
        full_start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
        start, end = full_start, node.end_lineno
        if excerpt.first:
            def statement(prefix):
                matches = [child for child in ast.walk(node) if isinstance(child, ast.stmt)
                           and lines[child.lineno - 1].strip().startswith(prefix)]
                if len(matches) != 1:
                    raise ValueError(f"Trecho {identifier}: âncora {prefix!r} não é única.")
                return matches[0]

            first = statement(excerpt.first)
            last = statement(excerpt.last) if excerpt.last else first
            start, end = first.lineno, last.end_lineno
            if start > end:
                raise ValueError(f"Trecho {identifier}: limites invertidos.")
        result[identifier] = {
            "title": excerpt.title, "path": excerpt.path, "symbol": excerpt.symbol,
            "start_line": start, "end_line": end, "lines": lines[start - 1:end],
        }
    return result
