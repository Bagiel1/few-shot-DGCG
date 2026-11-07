# Guia explicativo dos resultados few-shot — 03/11/2026

PDF na raiz: `entendendo_resultados_fewshot.pdf` — 23 páginas, 17 figuras, revisão visual de todas as páginas concluída..

Fontes congeladas em `fontes/`; hashes em `sha256.txt`.. Nenhum treino foi executado, nenhum código experimental foi alterado e o relatório principal anterior foi preservado..

Reprodução a partir da raiz do projeto:

```bash
venv/bin/python tmp/pdfs/explicacao_20261103/charts.py
/home/bagiel/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 tmp/pdfs/explicacao_20261103/build.py
```

Os sete JSONL têm 1.240 linhas; o diagnóstico de proxy tem 20 registros.. Não há chaves duplicadas dentro de cada JSONL, mas controles aparecem em mais de um arquivo.. Médias por configuração foram exportadas para CSV.. As médias têm peso igual por base; não foi calculada significância estatística..

Principais distinções documentadas: treino padrão confunde dimensão/inicialização com aprendizado; identidade sem treino mantém LayerNorm/L2; o otimizador atualiza Θ, LayerNorm e escala de logits; estado inicial não concorre por validação; homofilia entre bases não implica predição por episódio; adaptativo empata na média com no-graph; diferenças CPU/GPU não foram causalmente isoladas..
