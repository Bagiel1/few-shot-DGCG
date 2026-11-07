# Central dos resultados

## Leia primeiro

- [Fonte única numérica](consolidacao_confiavel_20261103/resultados_fewshot_fonte_unica.xlsx): resultados com protocolo, treino, dimensão, inicialização, dispositivo, agregação, fonte e grupo de comparação..
- [Guia da consolidação](consolidacao_confiavel_20261103/README.md): regra de uso, conteúdo e reprodução da auditoria..
- [Guia explicativo — 23 páginas](relatorios/entendendo_resultados_fewshot.pdf): narrativa sobre treino, inicialização, arestas, adaptativo, hipóteses e próximos controles..
- [Relatório principal — histórico consolidado](relatorios/relatorio_fewshot.pdf): narrativa dos experimentos anteriores, extras e escalas..
- [Relatórios anteriores](relatorios/anteriores/pdf/): versões históricas e seus geradores..

Para números, use a planilha da fonte única.. Os PDFs servem para explicação e preservam a leitura feita na data em que foram gerados.. Um número de PDF não deve ser comparado a outro sem conferir o grupo de comparação na planilha..

## Encontre os números

| Estudo | Fontes e tabelas |
|---|---|
| Consolidação auditada | [Catálogo canônico, médias de extras, inventário e checagens](consolidacao_confiavel_20261103/).. |
| Testes recentes do Claude | [JSON/JSONL originais](testes_claude_20261102/resultados/) e [descrição das baterias](testes_claude_20261102/README.md).. |
| Médias e interpretação dos testes recentes | [Médias por configuração, fontes congeladas e hashes](explicacao_fewshot_20261103/).. |
| CPU local versus histórico GPU | [80 pares correspondentes](conferencia_claude_historico/pares.csv).. |
| Flowers/Cars Micro, Mini e Extended | [Resultados, médias e status de 108 execuções](consolidado_fewshot_20261102/fontes/).. |
| Histórico de graus e densidades | [Fontes completas](analise_experimentos_20261026/fontes/) e [tabelas](analise_experimentos_20261026/tabelas/).. |
| Primeiras buscas | [Análise histórica de outubro](analise_experimentos_20261012/).. |
| Execuções e checkpoints locais | [Diretórios originais por bateria](execucoes/).. |
| Geração dos relatórios | [Scripts, figuras e análises auxiliares](reproducao/).. |

As cópias em `fontes/` são fotografias usadas nos relatórios, não novas execuções.. Foram preservadas para manter a rastreabilidade.. As pastas `smoke_*` em execuções são testes técnicos, não evidência científica para agregar às médias..

## Convenção para próximos trabalhos

Use uma pasta por bateria dentro de `resultados/`, com `fontes/` ou `resultados/` para dados brutos e tabelas nomeadas claramente.. Guarde PDFs finais em `relatorios/`, scripts de geração em `reproducao/` e saídas pesadas em `execucoes/`.. O código antigo ainda acessa esses mesmos locais pelos atalhos da raiz..
