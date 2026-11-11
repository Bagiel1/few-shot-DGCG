# Homofilia durante o treino — 08/11/2026

Bateria executada na VM em CUDA com
`bash scripts/buscar_homofilia_treinada.sh --training-curve --device cuda`
(commit `8716eee`). Os arquivos em `fontes/` são os CSVs originais gerados
pela execução, sem alteração. `hashes.sha256` permite conferir a integridade:

```bash
cd resultados/homofilia_treino_20261108 && sha256sum -c hashes.sha256
```

## Protocolo

- Dez bases Micro, 5-way, 1 e 5-shot, 10 queries por classe, seed 7.
- ResNet-18 ImageNet congelada em 128 px, projeção treinada 512 → 256.
- 60 treinos de 2.000 episódios: sem grafo, kNN-union e DGCG 4–6, pareados
  pela sequência de episódios em cada base/shot.
- Validação em 200 episódios a cada 200 passos; teste em 1.000 episódios.
- Homofilia das queries: fração do peso das arestas que chegam às queries
  cujos extremos são da mesma classe, sem autolaços. Os rótulos das queries
  entram só nessa medição, depois de o grafo estar montado.

## Arquivos

| Arquivo | Linhas | Conteúdo |
|---|---:|---|
| `resultados_homo.csv` | 60 | Teste final de cada treino e homofilia medida em 200 episódios de teste |
| `status_homo.csv` | 60 | Estado de cada condição |
| `curva_episodios.csv` | 80.000 | Um registro por episódio de treino com grafo |
| `curva_blocos.csv` | 400 | Blocos de 200 episódios: homofilia, ganho no treino e no teste |
| `primeira_vantagem.csv` | 40 | Primeiro bloco com vantagem do grafo e homofilia dos 2.000 episódios de treino |
| `pontos_homofilia.csv` | 40 | Homofilia de teste e ganho final contra o sem grafo |
| `medias_homo.csv` | 4 | Médias das dez bases por shot e grafo |
| `correlacoes.csv` | 2 | Pearson homofilia de teste × ganho final, por shot |
| `correlacoes_entre_bases.csv` | 6 | Pearson homofilia de treino × ganho final, por shot e grafo |

São três medidas diferentes: homofilia nos 2.000 episódios de treino,
homofilia em 200 episódios de teste e acurácia nos 1.000 episódios de teste.

Os caminhos de checkpoint dentro dos CSVs são os da VM. Os checkpoints não
foram copiados para este repositório.

Relatório: `resultados/relatorios/experimentos_fewshot_relatorio_final.pdf`,
páginas 14 a 17. Código: `experimento_homofilia_treinado.py`,
`proto_sgc/homophily.py` e `proto_sgc/training_curve.py`. Documentação:
`docs/homofilia_treinada.md`.
