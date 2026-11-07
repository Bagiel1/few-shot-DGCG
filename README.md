# Few-shot

Comece por [resultados/README.md](resultados/README.md).. A fonte numérica atual é [a consolidação auditada](resultados/consolidacao_confiavel_20261103/resultados_fewshot_fonte_unica.xlsx).. Os PDFs, métricas e experimentos estão concentrados em `resultados/`..

| Pasta | Conteúdo |
|---|---|
| `resultados/relatorios/` | Relatório principal e guia explicativo; versões anteriores em `anteriores/pdf/`.. |
| `resultados/testes_claude_20261102/` | Resultados JSON/JSONL e scripts dos testes recentes.. |
| `resultados/execucoes/` | Saídas originais dos experimentos, checkpoints, logs e features em cache.. |
| `resultados/analise_*/` e demais baterias | Fontes CSV, gráficos, tabelas e auditorias por estudo.. |
| `resultados/reproducao/` | Scripts e figuras usados para gerar os relatórios.. |
| `scripts/` | Comandos shell para executar as baterias.. |
| `docs/` | Documentação técnica, protocolos e registro da organização.. |
| `proto_sgc/`, `tests/` | Implementação e testes.. |
| `data/`, `venv/` | Dados locais e ambiente Python, preservados.. |

## Executar

Na raiz do projeto, com o ambiente apropriado..

```bash
bash scripts/buscar_escalas.sh --dry-run
bash scripts/buscar_escalas.sh --device cuda
```

Os módulos Python de experimentos permanecem na raiz para manter imports, hashes de configuração e comandos de reprodução existentes.. Os atalhos `runs`, `outputs`, `output` e `tmp` apontam para `resultados/`: não são cópias duplicadas, e preservam os caminhos usados pelo código e pelos relatórios anteriores..

A limpeza removeu apenas caches regeneráveis e prévias de páginas dos PDFs.. Dados científicos, checkpoints, caches de features, código e versões anteriores dos relatórios foram preservados.. Detalhes em [docs/organizacao_20261103.json](docs/organizacao_20261103.json)..
