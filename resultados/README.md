# Resultados

## Leia primeiro

- [Relatório atual](relatorios/experimentos_fewshot_relatorio_final.pdf): 21 páginas, baterias 1 a 10.
- [Planilha principal](consolidacao_confiavel_20261103/resultados_fewshot_fonte_unica.xlsx): resultados com protocolo, treino, dimensão, inicialização, dispositivo, agregação, fonte, grupo de comparação e margens de IC95.
- [Planilha por bateria](planilha_por_bateria/resultados_fewshot_por_bateria.xlsx): aba geral com 13.322 registros individuais e 16 abas, uma por bateria.
- [Guia da consolidação](consolidacao_confiavel_20261103/README.md): regra de uso e conferência da planilha principal.

Para números, use as planilhas. Os PDFs explicam e preservam a leitura da data em que foram gerados. Não compare um número de um PDF com outro sem conferir o grupo de comparação na planilha.

## Onde está cada bateria

| Bateria | Fontes |
|---|---|
| Métodos, faixas de grau e densidades (27 protocolos, 10 bases) | [analise_experimentos_20261026/fontes/](analise_experimentos_20261026/fontes/) e [tabelas/](analise_experimentos_20261026/tabelas/) |
| Primeiras buscas de outubro | [analise_experimentos_20261012/](analise_experimentos_20261012/) |
| Imagens extras (4.740 execuções) | [consolidacao_confiavel_20261103/fontes/](consolidacao_confiavel_20261103/fontes/) |
| Flowers e Cars em Micro, Mini e Extended (108 execuções) | [consolidado_fewshot_20261102/fontes/](consolidado_fewshot_20261102/fontes/) |
| Sem treino, inicialização, saltos, ponderação de arestas e propagação adaptativa, em CPU | [testes_claude_20261102/resultados/](testes_claude_20261102/resultados/), com descrição em [testes_claude_20261102/README.md](testes_claude_20261102/README.md) e médias em [explicacao_fewshot_20261103/](explicacao_fewshot_20261103/) |
| CPU local contra histórico em GPU (80 pares) | [conferencia_claude_historico/pares.csv](conferencia_claude_historico/pares.csv) |
| Contexto de queries (54 execuções) | [consolidacao_confiavel_20261103/fontes/contexto_queries_20261107/](consolidacao_confiavel_20261103/fontes/contexto_queries_20261107/) |
| ResNet-18 × ViT-B/16 (240 condições) | [planilha_por_bateria/fontes/features_20261108/](planilha_por_bateria/fontes/features_20261108/) |
| Homofilia durante o treino (60 treinos) | [homofilia_treino_20261108/](homofilia_treino_20261108/) |

A bateria de homofilia durante o treino ainda não tem aba nas planilhas. Os números dela estão nos CSVs da pasta e nas páginas 14 a 17 do relatório atual.

## Outras pastas

- `relatorios/`: PDFs. Versões anteriores em `relatorios/anteriores/pdf/`.
- `reproducao/`: scripts e figuras usados para gerar os relatórios.
- `graficos_para_aprovacao/`: dois gráficos de grau do DGCG por protocolo.
- `execucoes/`: saídas originais das execuções (checkpoints, logs, features em cache). Fica só na máquina local. As pastas `smoke_*` são testes de funcionamento e não entram em médias.

As cópias em `fontes/` são os arquivos usados nos relatórios e nas planilhas, preservados para rastreabilidade. Não são execuções novas.

## Convenção para próximas baterias

Uma pasta por bateria dentro de `resultados/`, com os CSVs originais em `fontes/`, um arquivo de hashes e um README curto com protocolo e comando. PDFs finais em `relatorios/`, scripts de geração em `reproducao/` e saídas pesadas em `execucoes/`.
