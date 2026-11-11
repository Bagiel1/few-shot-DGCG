# Few-shot com grafos

Classificação few-shot com ProtoNet e propagação em grafo (SGC) sobre vetores de um extrator congelado. O projeto compara o classificador sem grafo com grafos kNN e DGCG para entender quando a propagação ajuda ou atrapalha.

## Por onde começar

- Relatório: [resultados/relatorios/experimentos_fewshot_relatorio_final.pdf](resultados/relatorios/experimentos_fewshot_relatorio_final.pdf) (baterias 1 a 10).
- Números: [planilha principal](resultados/consolidacao_confiavel_20261103/resultados_fewshot_fonte_unica.xlsx) e [planilha por bateria](resultados/planilha_por_bateria/resultados_fewshot_por_bateria.xlsx).
- Guia das pastas de resultados: [resultados/README.md](resultados/README.md).

## Pastas

| Pasta | Conteúdo |
|---|---|
| `proto_sgc/` | Implementação: dados, features, episódios, grafos, modelo, treino e métricas. |
| `experimento*.py` (raiz) | Um módulo por bateria de experimentos. |
| `scripts/` | Comandos shell que executam cada bateria. |
| `tests/` | Testes automatizados. |
| `docs/` | Documentação técnica e protocolo de cada bateria. |
| `resultados/` | Planilhas, fontes CSV, gráficos e relatórios. |
| `data/`, `venv/`, `resultados/execucoes/` | Dados, ambiente Python e saídas de execução. Ficam só na máquina local. |

Os atalhos `runs`, `outputs`, `output` e `tmp` apontam para pastas dentro de `resultados/`. Eles existem porque o código e os scripts de geração dos relatórios usam esses caminhos.

## Onde ler o código

Comece por `proto_sgc/experiment.py`, que organiza o fluxo dados → features → episódios → modelo → treino ou avaliação.

| Arquivo | Responsabilidade |
|---|---|
| `proto_sgc/config.py` | Opções da linha de comando e validação dos hiperparâmetros. |
| `proto_sgc/features.py` | ResNet-18 e ViT-B/16 congelados, transformação das imagens e cache dos vetores. |
| `proto_sgc/model.py` e `proto_sgc/graphs/` | Projeção, construção dos grafos (kNN, DGCG, DGCG+, GRaNDe), propagação e classificação por protótipos. |
| `proto_sgc/training.py` | Treino, avaliação, escolha do checkpoint pela validação e gravação das métricas. |
| `proto_sgc/episodes.py` e `proto_sgc/reference_bank.py` | Sorteio dos episódios, banco de imagens extras e contexto de queries. |
| `proto_sgc/homophily.py` e `proto_sgc/training_curve.py` | Homofilia das arestas e curva pareada durante o treino. |
| `proto_sgc/result_inventory.py` | Conferência da planilha principal contra as fontes antes de uma bateria. |

## Baterias

| Bateria | Módulo | Script | Documentação |
|---|---|---|---|
| Métodos e faixas de grau | `experimentos_metricas_graus.py` | `scripts/buscar_metricas_graus.sh` | `docs/documentacao.md` |
| Densidades do DGCG | `experimentos_densidades_protocolos.py` | `scripts/buscar_densidades_protocolos.sh` | `docs/documentacao.md` |
| Imagens extras | `experimentos_extras.py`, `experimentos_extras_protocolos.py` | `scripts/buscar_extras.sh`, `scripts/buscar_extras_protocolos.sh` | `docs/EXPERIMENTO_EXTRAS_PROTOCOLOS.md` |
| Escalas Micro, Mini e Extended | `experimentos_escalas.py` | `scripts/buscar_escalas.sh` | `docs/EXPERIMENTO_ESCALAS.md` |
| Contexto de queries | `experimentos_extras.py` | `scripts/buscar_contexto_queries.sh` | `docs/experimento_contexto_queries.md` |
| ResNet-18 × ViT-B/16 | `experimentos_backbones.py` | `scripts/buscar_backbones.sh` | `docs/experimento_backbones.md` |
| Homofilia com treino | `experimento_homofilia_treinado.py` | `scripts/buscar_homofilia_treinada.sh` | `docs/homofilia_treinada.md` |

## Executar

Na raiz do projeto, com as dependências de `requirements.txt` instaladas:

```bash
python protosgc_fewshot.py --quick
bash scripts/buscar_escalas.sh --dry-run
bash scripts/buscar_escalas.sh --device cuda
```

Os scripts usam `venv/bin/python` quando existe, ou o Python do ambiente conda ativo.

## Testes

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

## Antes de uma bateria nova

Consulte a planilha principal e as fontes para não repetir condições já concluídas. Só reaproveite um resultado quando base, protocolo, extrator, dimensão, treino, seed, dispositivo e configuração do grafo forem os mesmos. Não substitua o resultado de uma base pela média de várias bases ou protocolos.
