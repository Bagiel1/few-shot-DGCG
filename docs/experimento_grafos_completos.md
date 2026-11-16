# Grafos completos com imagens extras

## Grade

- Dez familias do Meta-Album, nas versoes Micro e Mini.
- Protocolos separados: 2-way/5-shot/5-query, 2-way/10-shot/10-query,
  5-way/5-shot/5-query e 5-way/10-shot/10-query.
- Extras totais por episodio: Micro 0 a 100; Mini 0 a 200; passo 10.
- Dois pesos em todos os pares: RBO e RBF euclidiano.
- ResNet-18 ImageNet congelada, entrada 128 px, projecao 512 para 256,
  LayerNorm e escala dos logits treinadas. Seed 7.
- 2.000 episodios de treino, 200 de validacao a cada 200 passos,
  1.000 de teste. Checkpoint escolhido pela acuracia de validacao.

Sao 2.560 condicoes com grafo (880 Micro e 1.680 Mini) e 80 controles
sem grafo, totalizando 2.640 condicoes. Cada controle e treinado uma unica
vez por base, escala e protocolo, com seed 7 e zero extras. A projecao,
LayerNorm e escala dos logits sao treinadas com o mesmo orcamento e os
mesmos suportes/queries do respectivo protocolo com grafo.

Sao 16 graficos, cada um com uma curva por base e a referencia sem grafo
como linha horizontal tracejada na mesma cor. Os protocolos e escalas
permanecem separados. O CSV inclui `gain_vs_no_graph_pp`.

## Grafo e pesos

O grafo contem suporte, queries e extras. A matriz e construida nos vetores
congelados, antes de Theta. Nao ha selecao kNN nem limiar DGCG.

RBF: `exp(-||xi-xj||² / (2 * 32²))`. A distancia euclidiana usa os vetores
brutos. Os pesos sao os valores diretos do kernel, sem softmax nem reescala.
Sigma 32 e fixo em todas as bases, escalas, protocolos e quantidades de extras.

RBO: rankings euclidianos com o proprio no na primeira posicao, como no
DGCG existente. Usa p=0,9, lista limitada a 200 e profundidade limitada a
40 e ao numero de nos. O peso e a soma truncada ja implementada no projeto.
Um par com RBO zero tem peso zero mesmo sendo avaliado no grafo completo.

Nos dois casos, a diagonal e retirada antes da contagem de topologia.
Depois sao adicionados autolacos de peso 1 e aplicada a normalizacao
SGC por graus `D^(-1/2) (A + I) D^(-1/2)`, com duas propagacoes.
LayerNorm e L2 depois da projecao seguem o pipeline do projeto.

## Sorteio e particionamento

Extras entram em treino, validacao e teste. Cada fase usa somente suas
proprias classes. O sorteio prioriza imagens das classes representadas
pelos suportes do episodio. Quando essas imagens se esgotam, completa
com imagens de outras classes da mesma particao. Dentro de cada grupo,
o sorteio e uniforme entre imagens, sem reposicao e sem cotas por classe.
O avaliador usa as categorias do dataset para formar os dois grupos.
O modelo recebe apenas os vetores das imagens extras.

Suporte e queries sao amostrados primeiro. Seus identificadores ficam
excluidos do pool de extras. Um RNG separado, definido pela seed da fase
e pelas identidades do episodio, permuta o restante. Essa ordem e dividida
em imagens das classes do episodio primeiro e imagens de outras classes
depois. Quantidades maiores usam prefixos da mesma ordem. Assim, suporte e queries sao iguais
entre os dois grafos e entre quantidades de extras.

Exemplo Micro com 40 imagens por classe, 2-way/10-shot/10-query:
sobram 40 imagens das duas classes do episodio. Ate 40 extras, todas sao
dessas classes. Com 50 extras, sao 40 dessas classes e 10 de outras.

Micro usa split de classes 10/5/5. Mini usa metade das classes para treino,
um quarto para validacao e o restante para teste, com arredondamento para
baixo nos dois primeiros. Flowers Mini fica 51/25/26 e Cars Mini 98/49/49.
Os splits exatos e as classes ficam nas metricas de cada condicao.

Os rotulos das extras ficam fora do modelo. Prototipos usam suporte rotulado.
Somente queries supervisionam a loss e entram nas metricas. Sao registrados
hashes dos episodios por fase para conferir o pareamento.

## Executar na VM

Use o ambiente PyTorch/CUDA ja instalado. Os graficos requerem matplotlib,
que pode ser instalado com `python -m pip install matplotlib` se necessario.

Auditar e conferir o plano sem baixar imagens ou rodar treino:

```bash
bash scripts/buscar_grafos_completos.sh --dry-run
```

Executar a bateria completa:

```bash
bash scripts/buscar_grafos_completos.sh --device cuda
```

O mesmo comando retoma as condicoes pendentes. Cada condicao concluida tem
`result.json`, checkpoint e `run.log`. O manifesto impede misturar codigo
ou configuracoes diferentes na mesma pasta. A retomada e por condicao,
reiniciando uma condicao que tenha sido interrompida antes de concluir.
Um lock impede duas execucoes simultaneas na mesma pasta de resultados.

Para verificar uma condicao completa antes de continuar, acrescente `--limit 1`.
A opcao limita apenas novas condicoes, preservando o plano e a retomada.

Verificacao curta, em pasta separada:

```bash
bash scripts/buscar_grafos_completos.sh --device cuda \
  --datasets flowers --versions micro --protocols 2w5s5q \
  --graphs complete-rbf --limit 1 \
  --train-episodes 2 --val-every 1 --val-episodes 2 --test-episodes 3 \
  --output-root runs/smoke_grafos_completos
```

Recriar CSV e graficos com os resultados salvos:

```bash
bash scripts/buscar_grafos_completos.sh --summarize-only
```

A saida padrao e `runs/grafos_completos_extras_seed7/`, com
`resultados.csv`, `progress.json` e os 16 PNGs em `graficos/`.
Graficos de execucoes parciais mostram somente pontos concluidos.
Caches de features ficam em `_features/BASE_ESCALA/`, compartilhados
por todos os protocolos e quantidades. Caches existentes em `runs/`
sao copiados quando fingerprint, extrator e dimensao conferem.
`--feature-root` aceita outra raiz
com caches na mesma organizacao. A validade dos caches e conferida pelo
fingerprint e configuracao do extrator.

Antes do plano, o executor audita a planilha oficial contra as fontes.
Os resultados antigos de kNN/DGCG nao substituem os grafos completos
com extras de toda a particao. Todas as novas metricas mantem base,
protocolo, escala, seed, dispositivo e configuracao explicitos.
