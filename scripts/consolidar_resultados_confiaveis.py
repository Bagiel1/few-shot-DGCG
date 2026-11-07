"""Consolida resultados few-shot com escopo e comparabilidade explícitos."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resultados" / "consolidacao_confiavel_20261103"
SOURCES = OUT / "fontes"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def unique_check(frame: pd.DataFrame, keys: list[str]) -> int:
    return int(frame.duplicated(keys).sum())


def mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    value = frame[column].mean()
    return None if pd.isna(value) else float(value)


def add_row(rows: list[dict], **kwargs) -> None:
    fields = {
        "id": "",
        "bateria": "",
        "status": "Concluído",
        "escopo_bases": "",
        "versao_base": "Micro",
        "n_way": "",
        "n_shot": "",
        "n_query": "",
        "seeds": "",
        "dispositivo": "",
        "treino": "",
        "dimensao": "",
        "inicializacao": "",
        "preprocessamento": "raw",
        "metodo": "",
        "configuracao": "",
        "regra_agregacao": "",
        "n_observacoes": 0,
        "val_accuracy": None,
        "test_accuracy": None,
        "test_accuracy_std": None,
        "test_macro_f1": None,
        "grupo_comparacao": "",
        "comparabilidade": "",
        "alerta": "",
        "fonte": "",
        "filtro_fonte": "",
    }
    fields.update(kwargs)
    rows.append(fields)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)

    paths = {
        "historico_metricas": ROOT / "resultados/analise_experimentos_20261026/fontes/resultados_metricas.csv",
        "historico_densidades": ROOT / "resultados/analise_experimentos_20261026/fontes/resultados_densidades_protocolos_dgcg.csv",
        "grau_selecionado": ROOT / "resultados/analise_experimentos_20261026/fontes/melhores_graus_por_protocolo_metricas.csv",
        "densidade_selecionada": ROOT / "resultados/analise_experimentos_20261026/fontes/melhores_densidades_por_protocolo_dgcg.csv",
        "claude_e1": ROOT / "resultados/testes_claude_20261102/resultados/res_e1.jsonl",
        "claude_dgcg24": ROOT / "resultados/testes_claude_20261102/resultados/res_dgcg24.jsonl",
        "claude_idinit": ROOT / "resultados/testes_claude_20261102/resultados/res_idinit.jsonl",
        "claude_propagacao": ROOT / "resultados/testes_claude_20261102/resultados/res_proptf.jsonl",
        "claude_topologia_sem_treino": ROOT / "resultados/testes_claude_20261102/resultados/res_topo_tf.jsonl",
        "claude_topologia_com_treino": ROOT / "resultados/testes_claude_20261102/resultados/res_topo_tr.jsonl",
        "claude_adaptativo": ROOT / "resultados/testes_claude_20261102/resultados/res_adapt.jsonl",
        "claude_proxy": ROOT / "resultados/testes_claude_20261102/resultados/res_supproxy.json",
        "escalas_resultados": ROOT / "resultados/consolidado_fewshot_20261102/fontes/resultados_comparacao.csv",
        "escalas_medias": ROOT / "resultados/consolidado_fewshot_20261102/fontes/medias_comparacao.csv",
        "extras_resultados": SOURCES / "buscas_extras.csv",
        "extras_medias": SOURCES / "medias_buscas_extras.csv",
        "extras_status": SOURCES / "status_buscas_extas.csv",
    }
    expected = {
        "historico_metricas": 2430,
        "historico_densidades": 4050,
        "grau_selecionado": 270,
        "densidade_selecionada": 270,
        "claude_e1": 360,
        "claude_dgcg24": 80,
        "claude_idinit": 60,
        "claude_propagacao": 200,
        "claude_topologia_sem_treino": 360,
        "claude_topologia_com_treino": 40,
        "claude_adaptativo": 40,
        "claude_proxy": 20,
        "escalas_resultados": 108,
        "escalas_medias": 36,
        "extras_resultados": 4740,
        "extras_medias": 474,
    }
    key_sets = {
        "historico_metricas": ["dataset", "n_way", "n_shot", "n_query", "seed", "metodo", "alvo"],
        "historico_densidades": ["dataset", "n_way", "n_shot", "n_query", "seed", "metodo", "alvo"],
        "grau_selecionado": ["dataset", "n_way", "n_shot", "n_query", "seed"],
        "densidade_selecionada": ["dataset", "n_way", "n_shot", "n_query", "seed"],
        "claude_e1": ["dataset", "method", "way", "shot", "query", "seed", "feat", "trainfree", "tag"],
        "claude_dgcg24": ["dataset", "method", "way", "shot", "query", "seed", "feat", "trainfree", "tag"],
        "claude_idinit": ["dataset", "method", "way", "shot", "query", "seed", "feat", "trainfree", "tag"],
        "claude_propagacao": ["dataset", "method", "way", "shot", "query", "seed", "feat", "trainfree", "tag"],
        "claude_topologia_sem_treino": ["dataset", "method", "way", "shot", "query", "seed", "feat", "trainfree", "tag"],
        "claude_topologia_com_treino": ["dataset", "method", "way", "shot", "query", "seed", "feat", "trainfree", "tag"],
        "claude_adaptativo": ["dataset", "method", "way", "shot", "query", "seed", "feat", "trainfree", "tag"],
        "claude_proxy": ["dataset", "method", "shot"],
        "escalas_resultados": ["dataset", "n_way", "n_shot", "n_query", "seed", "metodo", "alvo"],
        "escalas_medias": ["dataset", "n_way", "n_shot", "n_query", "metodo", "alvo"],
        "extras_resultados": ["dataset", "n_way", "n_shot", "n_query", "seed", "metodo", "alvo", "extra_mode", "extra_phase", "extras_per_class"],
        "extras_medias": ["n_way", "n_shot", "n_query", "metodo", "alvo", "extra_mode", "extra_phase", "extras_per_class"],
    }

    frames: dict[str, pd.DataFrame] = {}
    inventory: list[dict] = []
    checks: list[dict] = []
    for name, path in paths.items():
        if not path.exists():
            inventory.append({"fonte": name, "arquivo": str(path.relative_to(ROOT)), "linhas": None, "sha256": None, "status": "AUSENTE"})
            checks.append({"checagem": f"Fonte presente: {name}", "status": "FALHA", "detalhe": str(path.relative_to(ROOT))})
            continue
        frame = read(path)
        frames[name] = frame
        inventory.append({
            "fonte": name,
            "arquivo": str(path.relative_to(ROOT)),
            "linhas": len(frame),
            "sha256": digest(path),
            "status": "OK",
        })
        if name in expected:
            ok = len(frame) == expected[name]
            checks.append({"checagem": f"Contagem esperada: {name}", "status": "OK" if ok else "FALHA", "detalhe": f"{len(frame)} / {expected[name]}"})
        if name in key_sets:
            duplicates = unique_check(frame, key_sets[name])
            checks.append({"checagem": f"Chaves únicas: {name}", "status": "OK" if duplicates == 0 else "FALHA", "detalhe": f"duplicadas={duplicates}"})

    rows: list[dict] = []
    hist = frames["historico_metricas"]
    degree_selected = frames["grau_selecionado"]
    density_selected = frames["densidade_selecionada"]
    method_names = {
        "knn-in": "kNN-in",
        "knn-out": "kNN-out",
        "knn-mutual": "kNN-mutual",
        "knn-union": "kNN-union",
        "no-graph": "Sem grafo",
        "dgcg": "DGCG",
        "union": "kNN-union",
        "dgcg24": "DGCG",
        "dgcgplus": "DGCG+",
    }

    def add_historical(frame: pd.DataFrame, scope_id: str, scope: str, ways: str, shots: str, queries: str, group: str) -> None:
        for (method, target), part in frame.groupby(["metodo", "alvo"], dropna=False):
            config = "sem grafo" if method == "no-graph" else ("k=5" if method.startswith("knn") else f"grau {target}")
            add_row(
                rows,
                id=f"{scope_id}-{method}-{target}",
                bateria="Histórico principal",
                escopo_bases=scope,
                n_way=ways,
                n_shot=shots,
                n_query=queries,
                seeds="7",
                dispositivo="GPU VM (informado; não consta no CSV)",
                treino="Padrão, 2.000 episódios",
                dimensao="256",
                inicializacao="Aleatória",
                metodo=method_names[method],
                configuracao=config,
                regra_agregacao="Média igual por base/protocolo",
                n_observacoes=len(part),
                val_accuracy=mean_or_none(part, "val_accuracy"),
                test_accuracy=mean_or_none(part, "test_accuracy"),
                test_macro_f1=mean_or_none(part, "test_macro_f1"),
                grupo_comparacao=group,
                comparabilidade="Comparável apenas a linhas do mesmo grupo",
                alerta="Média de vários protocolos" if ";" in ways else "Recorte de um protocolo",
                fonte=str(paths["historico_metricas"].relative_to(ROOT)),
                filtro_fonte=f"metodo={method}; alvo={target}; {scope_id}",
            )

    add_historical(hist, "HIST-ALL27", "10 bases Micro; 27 protocolos", "2;3;5", "1;5;10", "1;5;10", "HIST_ALL27_TRAINED")
    add_historical(hist[hist.episode_nodes > 6], "HIST-NGT6", "10 bases Micro; 25 protocolos com N>6", "2;3;5", "1;5;10", "1;5;10", "HIST_NGT6_TRAINED")
    hist_protocol = hist[(hist.n_way == 5) & (hist.n_shot == 5) & (hist.n_query == 10)]
    add_historical(hist_protocol, "HIST-5W5S10Q", "10 bases Micro; um protocolo", "5", "5", "10", "HIST_5W5S10Q_TRAINED")

    for label, frame, source_name, target_label in [
        ("DGCG grau selecionado", degree_selected, "grau_selecionado", "grau por validação"),
        ("DGCG densidade selecionada", density_selected, "densidade_selecionada", "densidade por validação"),
    ]:
        for scope_id, part, scope, ways, shots, queries, group in [
            ("ALL27", frame, "10 bases Micro; 27 protocolos", "2;3;5", "1;5;10", "1;5;10", "HIST_ALL27_TRAINED"),
            ("NGT6", frame[frame.episode_nodes > 6], "10 bases Micro; 25 protocolos com N>6", "2;3;5", "1;5;10", "1;5;10", "HIST_NGT6_TRAINED"),
            ("5W5S10Q", frame[(frame.n_way == 5) & (frame.n_shot == 5) & (frame.n_query == 10)], "10 bases Micro; um protocolo", "5", "5", "10", "HIST_5W5S10Q_TRAINED"),
        ]:
            add_row(
                rows,
                id=f"HIST-{scope_id}-{source_name}",
                bateria="Histórico principal",
                escopo_bases=scope,
                n_way=ways,
                n_shot=shots,
                n_query=queries,
                seeds="7",
                dispositivo="GPU VM (informado; não consta no CSV)",
                treino="Padrão, 2.000 episódios",
                dimensao="256",
                inicializacao="Aleatória",
                metodo=label,
                configuracao=target_label,
                regra_agregacao="Média igual por base/protocolo; alvo escolhido por validação",
                n_observacoes=len(part),
                val_accuracy=mean_or_none(part, "val_accuracy"),
                test_accuracy=mean_or_none(part, "test_accuracy"),
                test_macro_f1=mean_or_none(part, "test_macro_f1"),
                grupo_comparacao=group,
                comparabilidade="Comparável apenas a linhas do mesmo grupo",
                alerta="Seleção usa validação, nunca teste",
                fonte=str(paths[source_name].relative_to(ROOT)),
                filtro_fonte=scope_id,
            )

    e1 = frames["claude_e1"]
    for (shot, method, trainfree), part in e1[e1.feat == "raw"].groupby(["shot", "method", "trainfree"]):
        condition = "TF512" if trainfree else "TR256"
        add_row(
            rows,
            id=f"CLAUDE-E1-{shot}S-{method}-{condition}",
            bateria="Claude E1",
            escopo_bases="10 bases Micro; um protocolo",
            n_way="5",
            n_shot=str(shot),
            n_query="10",
            seeds="7",
            dispositivo="CPU local",
            treino="Sem treino" if trainfree else "Padrão, 2.000 episódios",
            dimensao="512" if trainfree else "256",
            inicializacao="Identidade" if trainfree else "Aleatória",
            metodo=method_names[method],
            configuracao="grau 4–6" if method == "dgcg" else ("k=5" if method == "union" else "sem grafo"),
            regra_agregacao="Média igual pelas 10 bases",
            n_observacoes=len(part),
            val_accuracy=mean_or_none(part, "val"),
            test_accuracy=mean_or_none(part, "test"),
            test_macro_f1=mean_or_none(part, "f1"),
            grupo_comparacao=f"CLAUDE_RAW_5W{shot}S10Q_{condition}",
            comparabilidade="Métodos comparáveis dentro do grupo",
            alerta="Comparar TR256 com TF512 confunde treino, dimensão e inicialização",
            fonte=str(paths["claude_e1"].relative_to(ROOT)),
            filtro_fonte=f"feat=raw; shot={shot}; method={method}; trainfree={trainfree}",
        )

    d24 = frames["claude_dgcg24"]
    for (shot, trainfree), part in d24[d24.feat == "raw"].groupby(["shot", "trainfree"]):
        condition = "TF512" if trainfree else "TR256"
        add_row(
            rows,
            id=f"CLAUDE-DGCG24-{shot}S-{condition}",
            bateria="Claude DGCG 2–4",
            escopo_bases="10 bases Micro; um protocolo",
            n_way="5",
            n_shot=str(shot),
            n_query="10",
            seeds="7",
            dispositivo="CPU local",
            treino="Sem treino" if trainfree else "Padrão, 2.000 episódios",
            dimensao="512" if trainfree else "256",
            inicializacao="Identidade" if trainfree else "Aleatória",
            metodo="DGCG",
            configuracao="grau 2–4",
            regra_agregacao="Média igual pelas 10 bases",
            n_observacoes=len(part),
            val_accuracy=mean_or_none(part, "val"),
            test_accuracy=mean_or_none(part, "test"),
            test_macro_f1=mean_or_none(part, "f1"),
            grupo_comparacao=f"CLAUDE_RAW_5W{shot}S10Q_{condition}",
            comparabilidade="Comparável a métodos com mesma condição; faixa do grafo é 2–4",
            alerta="Não confundir com DGCG 4–6",
            fonte=str(paths["claude_dgcg24"].relative_to(ROOT)),
            filtro_fonte=f"feat=raw; shot={shot}; trainfree={trainfree}",
        )

    identity = frames["claude_idinit"]
    for (shot, method), part in identity.groupby(["shot", "method"]):
        add_row(
            rows,
            id=f"CLAUDE-IDINIT-{shot}S-{method}",
            bateria="Claude identidade + treino",
            escopo_bases="10 bases Micro; um protocolo",
            n_way="5",
            n_shot=str(shot),
            n_query="10",
            seeds="7",
            dispositivo="CPU local",
            treino="Padrão, 2.000 episódios",
            dimensao="512",
            inicializacao="Identidade",
            metodo=method_names[method],
            configuracao="grau 2–4" if method == "dgcg24" else ("k=5" if method == "union" else "sem grafo"),
            regra_agregacao="Média igual pelas 10 bases",
            n_observacoes=len(part),
            val_accuracy=mean_or_none(part, "val"),
            test_accuracy=mean_or_none(part, "test"),
            test_macro_f1=mean_or_none(part, "f1"),
            grupo_comparacao=f"CLAUDE_IDINIT512_5W{shot}S10Q",
            comparabilidade="Métodos comparáveis dentro do grupo",
            alerta="DGCG desta bateria usa grau 2–4",
            fonte=str(paths["claude_idinit"].relative_to(ROOT)),
            filtro_fonte=f"shot={shot}; method={method}; tag=idinit",
        )

    prop = frames["claude_propagacao"]
    for (shot, method, tag), part in prop.groupby(["shot", "method", "tag"]):
        add_row(
            rows,
            id=f"CLAUDE-PROP-{shot}S-{method}-{tag}",
            bateria="Claude propagação",
            escopo_bases="10 bases Micro; um protocolo",
            n_way="5",
            n_shot=str(shot),
            n_query="10",
            seeds="7",
            dispositivo="CPU local",
            treino="Sem treino",
            dimensao="512",
            inicializacao="Identidade",
            metodo=method_names[method],
            configuracao=("grau 2–4; " if method == "dgcg24" else "k=5; ") + tag,
            regra_agregacao="Média igual pelas 10 bases",
            n_observacoes=len(part),
            val_accuracy=mean_or_none(part, "val"),
            test_accuracy=mean_or_none(part, "test"),
            test_macro_f1=mean_or_none(part, "f1"),
            grupo_comparacao=f"CLAUDE_RAW_5W{shot}S10Q_TF512",
            comparabilidade="Comparável ao baseline sem treino com mesma faixa de grafo",
            alerta="alpha mistura a adjacência com identidade antes dos saltos",
            fonte=str(paths["claude_propagacao"].relative_to(ROOT)),
            filtro_fonte=f"shot={shot}; method={method}; tag={tag}",
        )

    topo = frames["claude_topologia_sem_treino"]
    for (shot, method, tag), part in topo.groupby(["shot", "method", "tag"]):
        graph = "grau 4–6" if method in {"dgcg", "dgcgplus"} else ("k=5" if method == "union" else "sem grafo")
        add_row(
            rows,
            id=f"CLAUDE-TOPO-TF-{shot}S-{method}-{tag or 'base'}",
            bateria="Claude topologias sem treino",
            escopo_bases="10 bases Micro; um protocolo",
            n_way="5",
            n_shot=str(shot),
            n_query="10",
            seeds="7",
            dispositivo="CPU local",
            treino="Sem treino",
            dimensao="512",
            inicializacao="Identidade",
            metodo=method_names[method],
            configuracao=f"{graph}; {tag or 'base'}",
            regra_agregacao="Média igual pelas 10 bases",
            n_observacoes=len(part),
            val_accuracy=mean_or_none(part, "val"),
            test_accuracy=mean_or_none(part, "test"),
            test_macro_f1=mean_or_none(part, "f1"),
            grupo_comparacao=f"CLAUDE_RAW_5W{shot}S10Q_TF512",
            comparabilidade="Comparável dentro do grupo; usar baseline do próprio método para ganho da variante",
            alerta="Peso de aresta não equivale à intensidade global da propagação",
            fonte=str(paths["claude_topologia_sem_treino"].relative_to(ROOT)),
            filtro_fonte=f"shot={shot}; method={method}; tag={tag!r}",
        )

    topo_tr = frames["claude_topologia_com_treino"]
    for (shot, method, tag), part in topo_tr.groupby(["shot", "method", "tag"]):
        add_row(
            rows,
            id=f"CLAUDE-TOPO-TR-{shot}S-{method}-{tag}",
            bateria="Claude topologias com treino",
            escopo_bases="10 bases Micro; um protocolo",
            n_way="5",
            n_shot=str(shot),
            n_query="10",
            seeds="7",
            dispositivo="CPU local",
            treino="Padrão, 2.000 episódios",
            dimensao="256",
            inicializacao="Aleatória",
            metodo=method_names[method],
            configuracao=("grau 4–6; " if method == "dgcg" else "k=5; ") + tag,
            regra_agregacao="Média igual pelas 10 bases",
            n_observacoes=len(part),
            val_accuracy=mean_or_none(part, "val"),
            test_accuracy=mean_or_none(part, "test"),
            test_macro_f1=mean_or_none(part, "f1"),
            grupo_comparacao=f"CLAUDE_RAW_5W{shot}S10Q_TR256",
            comparabilidade="Comparável ao baseline treinado do mesmo método",
            alerta="Somente cosrbf foi concluído nesta grade",
            fonte=str(paths["claude_topologia_com_treino"].relative_to(ROOT)),
            filtro_fonte=f"shot={shot}; method={method}; tag={tag}",
        )

    adapt = frames["claude_adaptativo"]
    for (method, tag), part in adapt.groupby(["method", "tag"]):
        add_row(
            rows,
            id=f"CLAUDE-ADAPT-5S-{method}-{tag}",
            bateria="Claude adaptativo",
            escopo_bases="10 bases Micro; um protocolo",
            n_way="5",
            n_shot="5",
            n_query="10",
            seeds="7",
            dispositivo="CPU local",
            treino="Sem treino",
            dimensao="512",
            inicializacao="Identidade",
            metodo=method_names[method],
            configuracao=f"{'grau 4–6' if method == 'dgcg' else 'k=5'}; {tag}",
            regra_agregacao="Média igual pelas 10 bases",
            n_observacoes=len(part),
            val_accuracy=mean_or_none(part, "val"),
            test_accuracy=mean_or_none(part, "test"),
            test_macro_f1=mean_or_none(part, "f1"),
            grupo_comparacao="CLAUDE_RAW_5W5S10Q_TF512",
            comparabilidade="Comparável ao sem grafo e aos grafos sem treino do mesmo grupo",
            alerta="Dois limiares testados; não escolher o melhor pelo teste",
            fonte=str(paths["claude_adaptativo"].relative_to(ROOT)),
            filtro_fonte=f"method={method}; tag={tag}",
        )

    scales = frames["escalas_medias"]
    for row in scales.itertuples(index=False):
        add_row(
            rows,
            id=f"ESCALAS-{row.dataset}-{row.n_shot}S-{row.metodo}-{row.alvo}",
            bateria="Flowers e Cars em escalas",
            escopo_bases=row.dataset,
            versao_base=row.dataset.rsplit("_", 1)[-1].title(),
            n_way=str(row.n_way),
            n_shot=str(row.n_shot),
            n_query=str(row.n_query),
            seeds=str(row.seeds),
            dispositivo="GPU VM (informado; não consta no CSV)",
            treino="Padrão, 2.000 episódios",
            dimensao="256",
            inicializacao="Aleatória",
            metodo=method_names[row.metodo],
            configuracao=(f"densidade {row.alvo}" if row.metodo == "dgcg" else ("k=5" if row.metodo == "knn-union" else "sem grafo")),
            regra_agregacao="Média entre três seeds; desvio amostral entre seeds",
            n_observacoes=int(row.concluidos),
            val_accuracy=float(row.val_accuracy_mean),
            test_accuracy=float(row.test_accuracy_mean),
            test_accuracy_std=float(row.test_accuracy_std_seed),
            test_macro_f1=float(row.test_macro_f1_mean),
            grupo_comparacao=f"ESCALAS_{row.dataset}_{row.n_way}W{row.n_shot}S{row.n_query}Q",
            comparabilidade="Comparável aos outros métodos da mesma base, versão e shot",
            alerta="Micro, Mini e Extended mudam as classes; episódio mantém 55 ou 75 nós",
            fonte=str(paths["escalas_medias"].relative_to(ROOT)),
            filtro_fonte=f"dataset={row.dataset}; shot={row.n_shot}; metodo={row.metodo}; alvo={row.alvo}",
        )

    catalog = pd.DataFrame(rows)
    catalog.to_csv(OUT / "catalogo_resultados.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    extras = frames["extras_medias"].copy()
    extras.insert(0, "grupo_comparacao", extras.apply(lambda r: f"EXTRAS_{r.n_way}W{r.n_shot}S{r.n_query}Q_{r.metodo}_{r.alvo}", axis=1))
    extras.insert(1, "escopo", "Média pelas 10 bases Micro; seed 7")
    extras.insert(2, "dispositivo", "GPU VM (informado; não consta no CSV)")
    extras.insert(3, "treino", "Padrão, 2.000 episódios")
    extras.insert(4, "alerta", "Comparar somente com o controle sem extras da mesma bateria e protocolo")
    extras.to_csv(OUT / "extras_474_medias.csv", index=False)

    # Recalcula as 36 médias das escalas a partir das 108 linhas brutas.
    scale_raw = frames["escalas_resultados"]
    scale_recalc = scale_raw.groupby(["dataset", "n_way", "n_shot", "n_query", "metodo", "alvo"], dropna=False).agg(
        test_accuracy_mean=("test_accuracy", "mean"),
        val_accuracy_mean=("val_accuracy", "mean"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        concluidos=("seed", "size"),
    ).reset_index()
    merged = scales.merge(scale_recalc, on=["dataset", "n_way", "n_shot", "n_query", "metodo", "alvo"], suffixes=("_csv", "_recalc"), validate="one_to_one")
    max_scale_diff = float((merged.test_accuracy_mean_csv - merged.test_accuracy_mean_recalc).abs().max())
    checks.append({"checagem": "Médias de escalas recalculadas", "status": "OK" if max_scale_diff < 1e-9 else "FALHA", "detalhe": f"diferença máxima={max_scale_diff:.12g}"})

    # Compara a reprodução Claude CPU com o recorte histórico GPU correspondente.
    hist_slice = hist_protocol.rename(columns={"metodo": "method", "test_accuracy": "historico_test"})
    hist_slice["method"] = hist_slice.method.replace({"knn-union": "union"})
    hist_slice = hist_slice[(hist_slice.method.isin(["no-graph", "union", "dgcg"])) & ((hist_slice.method != "dgcg") | (hist_slice.alvo == "4:6"))]
    cpu = e1[(e1.feat == "raw") & (~e1.trainfree) & (e1.shot == 5)][["dataset", "method", "test"]]
    pairs = cpu.merge(hist_slice[["dataset", "method", "historico_test"]], on=["dataset", "method"], validate="one_to_one")
    pairs["delta_cpu_gpu_pp"] = pairs.test - pairs.historico_test
    reproduction = pairs.groupby("method").agg(
        pares=("dataset", "size"),
        cpu_mean=("test", "mean"),
        gpu_mean=("historico_test", "mean"),
        delta_mean=("delta_cpu_gpu_pp", "mean"),
        delta_abs_max=("delta_cpu_gpu_pp", lambda x: x.abs().max()),
    ).reset_index()
    reproduction.to_csv(OUT / "reproducao_cpu_gpu_5w5s10q.csv", index=False)
    checks.append({"checagem": "Pares CPU/GPU do protocolo 5w5s10q", "status": "OK" if len(pairs) == 30 else "FALHA", "detalhe": f"pares={len(pairs)}"})

    headline_ids = [
        "HIST-ALL27-dgcg-4:6",
        "HIST-5W5S10Q-dgcg-4:6",
        "CLAUDE-E1-5S-dgcg-TR256",
        "CLAUDE-E1-5S-dgcg-TF512",
        "HIST-ALL27-no-graph--",
        "HIST-5W5S10Q-no-graph--",
        "CLAUDE-E1-5S-no-graph-TF512",
        "CLAUDE-IDINIT-5S-no-graph",
    ]
    headline = catalog[catalog.id.isin(headline_ids)].copy()
    missing_headlines = sorted(set(headline_ids) - set(headline.id))
    checks.append({"checagem": "Resultados que causaram confusão estão presentes", "status": "OK" if not missing_headlines else "FALHA", "detalhe": ", ".join(missing_headlines) or "8/8"})

    failures = [item for item in checks if item["status"] != "OK"]
    audit = {
        "status": "OK" if not failures else "FALHA",
        "generated_at": "2026-11-03",
        "canonical_rows": len(catalog),
        "extras_rows": len(extras),
        "checks": checks,
        "failures": failures,
    }
    (OUT / "auditoria.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    pd.DataFrame(inventory).to_csv(OUT / "inventario_fontes.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "checagens.csv", index=False)
    payload = {
        "catalog": json.loads(catalog.to_json(orient="records", force_ascii=False)),
        "extras": json.loads(extras.to_json(orient="records", force_ascii=False)),
        "inventory": inventory,
        "checks": checks,
        "reproduction": json.loads(reproduction.to_json(orient="records", force_ascii=False)),
        "headline": json.loads(headline.to_json(orient="records", force_ascii=False)),
        "audit": audit,
    }
    (OUT / "workbook_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"status": audit["status"], "canonical_rows": len(catalog), "extras_rows": len(extras), "checks": len(checks), "failures": len(failures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
