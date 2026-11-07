# Experimento de contexto com queries não rotuladas

## Pergunta respondida

O experimento verifica se o kNN-union melhora quando recebe mais imagens não rotuladas no mesmo episódio, mantendo fixos o suporte e as queries usadas para calcular a acurácia.

## Formação de cada episódio

No protocolo padrão deste arquivo, o sorteio mestre contém, por classe:

- 1 imagem de suporte;
- 1 query avaliada;
- 9 queries de contexto.

Em 5-way, o grafo com contexto tem 55 nós: 5 suportes, 5 queries avaliadas e 45 queries não rotuladas.

Somente as 5 queries avaliadas entram na entropia cruzada e nas métricas. As 45 queries de contexto entram em `extra_x`, participam da construção do grafo e das duas propagações, mas seus rótulos não são passados ao modelo.

## Pareamento entre as condições

O programa sempre sorteia o episódio mestre completo com 10 queries por classe, inclusive quando o contexto ativo é zero. Por isso, uma execução com contexto zero e outra com contexto nove usam os mesmos suportes e as mesmas queries avaliadas quando recebem a mesma seed.

Os níveis intermediários são aninhados:

- `0` por classe: `--extra-images 0`;
- `5` por classe: `--extra-images 25`;
- `9` por classe: `--extra-images 45`.

O nível menor sempre é subconjunto do nível maior. Assim, a comparação não troca imagens para completar quantidades diferentes.

## Execuções

Controle pareado, treinado e avaliado sem contexto:

```bash
python experimento_contexto_queries.py \
  --extra-images 0 \
  --graph-type knn-union \
  --device cuda \
  --seed 7 \
  --work-dir runs/contexto_queries
```

Experimento 45/45, com 45 imagens de contexto no treino, na validação e no teste:

```bash
python experimento_contexto_queries.py \
  --extra-images 45 \
  --graph-type knn-union \
  --device cuda \
  --seed 7 \
  --work-dir runs/contexto_queries
```

Diagnóstico com o mesmo treinamento do controle e 45 imagens adicionadas somente no teste:

```bash
python experimento_contexto_queries.py \
  --extra-images 45 \
  --extra-phases test \
  --graph-type knn-union \
  --device cuda \
  --seed 7 \
  --work-dir runs/contexto_queries_teste
```

Para esse diagnóstico, a execução de controle correspondente usa os mesmos argumentos e troca apenas `--extra-images 45` por `0`.

## Bateria completa

O script `scripts/buscar_contexto_queries.sh` executa Flowers Micro e Cars Micro, com seeds 7, 17 e 27. Para cada par base/seed, ele cria estas nove execuções:

- um controle com zero contexto;
- 25 contextos somente no treino;
- 45 contextos somente no treino;
- 25 contextos no treino e na validação;
- 45 contextos no treino e na validação;
- 25 contextos somente no teste;
- 45 contextos somente no teste;
- 25 contextos no treino, na validação e no teste;
- 45 contextos no treino, na validação e no teste.

O controle com zero contexto é único porque as quatro escolhas de fase produzem exatamente a mesma execução quando nenhuma imagem extra está ativa. O plano padrão contém 54 execuções: 2 bases × 3 seeds × 9 condições.

Para conferir o plano sem executar modelos:

```bash
bash scripts/buscar_contexto_queries.sh --device cuda --dry-run
```

Para fazer um teste curto com Flowers e uma seed:

```bash
bash scripts/buscar_contexto_queries.sh \
  --datasets flowers_micro \
  --seeds 7 \
  --device cuda \
  --quick
```

Para executar a bateria completa:

```bash
bash scripts/buscar_contexto_queries.sh --device cuda
```

Os resultados consolidados ficam em `runs/contexto_queries_bateria/resultados.csv`, `status.csv` e `medias.csv`. O script preserva as execuções concluídas e as ignora quando o mesmo comando é retomado.

## Arquivos envolvidos

- `experimento_contexto_queries.py` fixa os padrões 5-way, 1-shot, 1-query e nove contextos por classe.
- `proto_sgc/reference_bank.py` contém `QueryContextBank`, responsável por sortear o episódio mestre e separar avaliação de contexto.
- `proto_sgc/config.py` acrescenta apenas `--extra-source queries`; quantidade, modo, fases e pool reutilizam as opções que já existiam.
- `proto_sgc/training.py` conecta o banco ao treino, à validação e ao teste sempre em modo de propagação.
- `proto_sgc/experiment.py` registra a configuração nos nomes e nos logs.
- `tests/test_query_context.py` verifica pareamento, aninhamento, ausência de rótulos e uma execução curta do pipeline.
- `scripts/buscar_contexto_queries.sh` define a matriz 0/25/45 e as quatro condições de fase.

O arquivo `.metrics.json` registra `source: queries_from_same_master_episode`, a quantidade total solicitada e a configuração de cada fase.
