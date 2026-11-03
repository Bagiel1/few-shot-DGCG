# Testes Claude — 02/11/2026 (few-shot, área mais promissora)

Feito por Claude (Opus 5.5) a pedido de Gabriel. **Nenhum arquivo de código do projeto foi alterado**; backup do estado: branch `backup/antes-testes-2026-11-02` (commit `7ee4820`). Tudo roda por um harness (`scripts/fs_lab.py`) que importa `proto_sgc` e aplica variantes por wrapper em memória.

## Setup comum
ResNet-18 ImageNet 128px congelada (cache existente de `runs/busca_extras_auto/_work`), 10 bases Micro, split 10/5/5, **5-way, 1 e 5 shots, 10 queries, seed 7**, CPU. Treino: 2000 ep., val 200 a cada 200, teste 1000 (defaults). "Sem treino" = Theta identidade (512-d), só avaliação. Resultados: uma linha JSON por execução em `resultados/*.jsonl` (val, test, IC, F1, grau, homofilia). Uma seed só: nada aqui tem significância estatística.

Checagem de fidelidade: DGCG 2-4 treinado reproduz o histórico (`resultados_metricas.csv`) quase base a base (ex.: Bacteria 66,2=66,2; Flowers 82,0=82,0).

## Resultados concluídos (médias de teste nas 10 bases)

### 1) Meta-treino vs sem treino (`res_e1.jsonl`, 360 execuções)
| | no-graph | union | DGCG 4-6 |
|---|---|---|---|
| 1-shot treinado | 60,33 | 66,82 | 64,87 |
| 1-shot sem treino | 61,08 | 65,70 | 64,15 |
| 5-shot treinado | 73,65 | 74,84 | 75,14 |
| 5-shot sem treino | **76,10** | 73,45 | 73,84 |
- Meta-treino **piora o no-graph** em 8/10 bases (−4 a −7 pp em Birds, Flowers, RESISC, Sports, Plant Village). Ajuda muito só em Textures (+12/+15 pp) e Cars 1-shot.
- 5-shot: no-graph sem treino supera todos os treinados; validação concorda (81,0 vs 79,7–80,0). O baseline "no-graph" histórico estava prejudicado pelo treino.
- Nos grafos, o treino muda só ±1 pp: o ganho vem da propagação.
- Pré-processamento CL2N / PT+CL2N: +0,5 a +1 pp (union); pouco relevante.

### 2) DGCG 2-4 vs 4-6 (`res_dgcg24.jsonl`)
Nestes protocolos (55/75 nós) 4-6 > 2-4: 1-shot 64,87 vs 63,61; 5-shot 75,14 vs 74,12. Consistente com o histórico de Gabriel: 2-4 é melhor na média dos 27 protocolos por causa dos episódios pequenos (≤15 nós: +2,7 a +11 pp); com >40 nós 4-6 ≥ 2-4 (76,98 vs 76,69). Grau ideal cresce com N.

### 3) Força da propagação, sem treino (`res_proptf.jsonl`, DGCG 2-4 e union)
Saltos 1–4 e mistura com identidade mudam ≤1 pp no 1-shot. No 5-shot menos propagação é melhor (union: 2 saltos 73,45 → mistura α=0,5 75,24), mas ainda < no-graph 76,10.

### 4) Theta iniciada na identidade + treino (`res_idinit.jsonl`, feature-dim 512)
no-graph: 1-shot 62,85 (vs 60,33 treinado padrão / 61,08 sem treino); 5-shot 76,29. Mantém ganho em Textures (46,0 / 57,4) e perde menos nas outras. Grafos +0,1 a +1,1 pp.

### 5) Topologia/qualidade de aresta, sem treino (`res_topo_tf.jsonl`, 360 execuções)
Variantes (wrappers em `scripts/topo.py`): peso RBO / RBO², SNN, DGCG+, RBF cosseno, poda por RBO < mediana, simetrização, só mútuas, arestas de suporte por rótulo, GRaNDe.
- Ganhos pequenos (≤1,8 pp). 1-shot: melhor union+RBF cosseno 65,92 (union 65,70; DGCG 64,15). 5-shot: pesar arestas do DGCG ajuda **consistentemente** (RBO² e RBF cosseno: +1 pp, 10/10 bases; DGCG+ +1,6 pp, 9/10), mas no-graph sem treino (76,10) segue melhor.
- GRaNDe, poda e só-mútuas não ajudaram.

### 6) Homofilia (fração das arestas ligando mesma classe; rótulos de query só para medir)
- Correlação entre bases homofilia × ganho do grafo sobre no-graph: **0,65 (1-shot) e 0,86 (5-shot)**. Cars (32–39%) e Textures (26–32%) têm homofilia quase de acaso (~19%) → grafo atrapalha.
- Paradoxo: DGCG tem homofilia 12–18 pp maior que union, mas union ganha no 1-shot (grau ~8 vs 5, simétrico). Homofilia não é tudo; alcance dos suportes parece importar.
- **Proxy sem rótulo de query** (`res_supproxy.json`, 5-shot): homofilia das arestas ENTRE SUPORTES vs das queries, correlação entre bases **0,998** (union) / **0,999** (DGCG). Por episódio é ruidosa (~0).

## Em andamento quando os créditos acabaram (parciais, NÃO conclusivos)
- `res_topo_tr.jsonl`: confirmação COM treino de union|cosrbf, dgcg|cosrbf, dgcg|topo:wrbo2, dgcgplus. Parcial pareado: cosrbf ≈ base (±0,4 pp). Rodar: `python scripts/runner.py "topo:tr:union|cosrbf,dgcg|cosrbf,dgcg|topo:wrbo2,dgcgplus|" res_topo_tr.jsonl 2` (retoma).
- `res_adapt.jsonl`: **propagação auto-calibrada** 5-shot sem treino: w = clip((h_sup−lo)/(hi−lo),0,1), A_eff = w·Â + (1−w)·I, h_sup = homofilia entre suportes do episódio. Parcial em 5 bases: DGCG adapt 79,97–80,21 vs DGCG 79,55 e no-graph 79,94; union adapt 79,85–80,09 vs union 78,93. Ou seja, acompanha o melhor dos dois regimes. Rodar `scripts/run_adapt.py`.

## Área mais promissora (opinião do Claude)
**Qualidade/confiabilidade das arestas como decisão de quanto propagar**, não mais ajuste de grau. A homofilia explica quando o grafo ajuda; ela é estimável sem rótulos de query a partir das arestas entre suportes (5-shot). Próximos passos: (1) terminar adapt e testar com treino e seeds 17/27; (2) proxy para 1-shot (sem pares de suporte da mesma classe), p.ex. consistência de predição com/sem propagação ou concordância RBO; (3) calibrar lo/hi na validação, nunca no teste; (4) reportar o no-graph sem treino/idinit como baseline honesto.

Execução: `cd outputs/testes_claude_20261102/scripts && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 ../../../venv/bin/python runner.py <grade> <saida.jsonl> 2`. Grades em `runner.py`. Os scripts esperam a pasta `feats/`/`work/` ao lado (criadas automaticamente; `work/` é descartável). PC fraco: usar no máximo 2 processos.

## Atualização final (concluídos após a seção "Em andamento")
- **Com treino, RBF cosseno (`res_topo_tr.jsonl`, 10 bases):** DGCG 1-shot 65,27 vs 64,87 (4/10 bases melhores); 5-shot 75,81 vs 75,14 (7/10). Union 1-shot 66,91 vs 66,82 (6/10); 5-shot 75,78 vs 74,84 (7/10). Ganho pequeno, mais no 5-shot. As variantes `dgcg|topo:wrbo2` e `dgcgplus|` NÃO rodaram (filtro da grade não casou); repetir se interessar.
- **Propagação auto-calibrada 5-shot, sem treino (`res_adapt.jsonl`, 10 bases):** DGCG 73,84 → 75,72 (lo/hi 0,35/0,75) e 76,00 (0,45/0,85); union 73,45 → 75,84 e 76,10. No-graph 76,10; oráculo max(grafo, no-graph) por base 76,27 (DGCG). Conclusão: a regra elimina o prejuízo do grafo (+2,2 a +2,6 pp sobre o grafo), mas **não supera o no-graph** no 5-shot; lo/hi foram escolhidos a priori, não pela validação. O valor está em evitar dano onde a homofilia é baixa; para superar o no-graph é preciso aresta melhor, não só dosagem.
