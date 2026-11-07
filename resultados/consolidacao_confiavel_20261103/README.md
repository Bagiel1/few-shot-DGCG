# Fonte única dos resultados few-shot

Comece por `resultados_fewshot_fonte_unica.xlsx`.. A primeira aba resolve a diferença entre 76,05%, 75,14%, 73,85% e 76,10% e indica quais comparações são válidas..

## Regra de uso

Nenhum resultado deve ser citado sem protocolo, escopo de bases, seed, dispositivo, condição de treino, dimensão, inicialização, método, configuração do grafo e regra de agregação..

O campo `grupo_comparacao` identifica condições compatíveis.. Comparações entre grupos diferentes exigem uma justificativa explícita e normalmente são apenas descritivas..

## Conteúdo

- `resultados_fewshot_fonte_unica.xlsx`: visão principal, 155 resultados canônicos, 474 médias de extras, fontes, auditoria e glossário..
- `catalogo_resultados.csv`: versão tabular dos resultados canônicos..
- `extras_474_medias.csv`: médias da bateria de 4.740 execuções com extras..
- `inventario_fontes.csv`: arquivo, quantidade de linhas e SHA-256 de cada fonte..
- `checagens.csv` e `auditoria.json`: resultado das validações automáticas..
- `reproducao_cpu_gpu_5w5s10q.csv`: reprodução do protocolo 5-way, 5-shot e 10-query..
- `fontes/`: cópia dos três arquivos completos da bateria de extras que estavam em Downloads..

Os JSONL do Claude, as fontes históricas e as fontes das escalas continuam nas pastas originais.. O inventário registra seus caminhos e hashes, evitando novas cópias desnecessárias..

## Auditoria executada

Foram verificadas contagens esperadas, chaves duplicadas, médias das escalas recalculadas a partir das 108 execuções, presença dos oito resultados que causaram confusão e os 30 pares CPU/GPU do protocolo 5w5s10q..

O estado atual é `OK`, com 35 checagens aprovadas e nenhuma falha..

## Reproduzir

Na raiz do projeto, execute o consolidado numérico e depois gere a planilha..

```bash
/home/bagiel/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/consolidar_resultados_confiaveis.py
/home/bagiel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node resultados/consolidacao_confiavel_20261103/build_workbook.mjs
```

Os arquivos científicos não são alterados por esses comandos.. A consolidação lê as fontes e recria as tabelas derivadas..
