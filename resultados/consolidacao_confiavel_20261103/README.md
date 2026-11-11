# Fonte única dos resultados few-shot

Comece por `resultados_fewshot_fonte_unica.xlsx`.. A primeira aba resolve a diferença entre 76,05%, 75,14%, 73,85% e 76,10% e indica quais comparações são válidas..

## Regra de uso

Nenhum resultado deve ser citado sem protocolo, escopo de bases, seed, dispositivo, condição de treino, dimensão, inicialização, método, configuração do grafo e regra de agregação..

O campo `grupo_comparacao` identifica condições compatíveis.. Comparações entre grupos diferentes exigem uma justificativa explícita e normalmente são apenas descritivas..

## Conteúdo

- `resultados_fewshot_fonte_unica.xlsx`: visão principal, 155 resultados canônicos, 474 médias de extras, 18 médias e 54 execuções do contexto de queries, margens de IC95, fontes, auditoria e glossário.
- `catalogo_resultados.csv`: versão tabular dos 155 resultados canônicos.
- `extras_474_medias.csv`: médias da bateria de 4.740 execuções com extras.
- `inventario_fontes.csv`: arquivo, quantidade de linhas e SHA-256 de cada fonte.
- `checagens.csv` e `auditoria.json`: resultado das validações automáticas de 03/11/2026.
- `reproducao_cpu_gpu_5w5s10q.csv`: reprodução do protocolo 5-way, 5-shot e 10-query.
- `fontes/`: arquivos completos da bateria de extras e, em `fontes/contexto_queries_20261107/`, os da bateria de contexto de queries.

As demais fontes continuam nas pastas originais. O inventário registra seus caminhos e hashes.

## Conferência

A consolidação de 03/11/2026 verificou contagens esperadas, chaves duplicadas, médias das escalas recalculadas a partir das 108 execuções e os 30 pares CPU/GPU do protocolo 5w5s10q: 35 checagens aprovadas, nenhuma falha.

Para conferir a planilha atual contra as fontes sem alterar nada:

```bash
python -c "from pathlib import Path; from proto_sgc.result_inventory import audit_results; audit_results(Path('.')); print('OK')"
```

## Atualização da planilha

A planilha recebeu abas e colunas depois de 03/11/2026 (contexto de queries em 07/11 e margens de IC95 em 09/11), acrescentadas diretamente no arquivo.

**Não execute `build_workbook.mjs` como está.** Ele recria a versão de 03/11 a partir de `workbook_data.json` e apagaria as abas e colunas novas. Antes de usá-lo de novo é preciso adaptá-lo ao conteúdo atual. `scripts/consolidar_resultados_confiaveis.py` regrava apenas os CSVs e JSON derivados desta pasta.
