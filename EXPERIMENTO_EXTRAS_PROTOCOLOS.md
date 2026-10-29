# Extras por protocolo

Execute na raiz do projeto, com o ambiente Python ativado:

```bash
bash buscar_extras_protocolos.sh --device cuda
```

Para escolher automaticamente GPU/CPU, use `--device auto`. A disponibilidade
do dispositivo é verificada antes dos treinos. A grade para no primeiro erro.

## Grade padrão

- Ways: 2 e 5.
- Shots por classe: 1, 5 e 10.
- Queries por classe: 10.
- Extras por classe: 5, 10 e 20. Controle sem extras incluído uma vez por método/protocolo/base/seed.
- Fases: só treino, treino + validação, treino + validação + teste.
- DGCG 2-4 e 4-6: ranked lists e propagação.
- kNN-in/out/mutual/union: propagação. No graph: controle.
- Dez bases Micro; seed 7; defaults de treino preservados.

São 6 protocolos, 79 condições por protocolo/base e **4.740 execuções**.
O protocolo anterior 2-way/10-shot/10-query está incluído. O 5-way testa tarefas
com mais classes; 1/5 shots testam menor supervisão. Não se trata de escolher
o melhor way pela acurácia: o número de classes altera a dificuldade da tarefa.

O banco é fixo em 20 imagens/classe, inclusive nos controles. Cada fase usa
somente seu próprio split; extras, suporte e queries não se sobrepõem.
Varia a quantidade utilizada do banco, não a reserva. Micro permite no máximo
40 imagens por classe: shots + queries + banco deve ser <= 40.
O limite de 20 extras preserva 10-shot/10-query com imagens distintas.

## Conferir ou ajustar

```bash
bash buscar_extras_protocolos.sh --dry-run
bash buscar_extras_protocolos.sh --device cuda --ways 5 --shots 1 5 --extras-per-class 5 10 20 --output-root runs/extras_5way
```

`--queries`, `--datasets`, `--methods`, `--degrees`, `--seeds` e `--modes`
também podem ser configurados. Use outra pasta ao mudar a grade. Para repetir
com três seeds, por exemplo, `--seeds 7 17 27` gera 14.220 execuções no padrão.

## Resultados e retomada

Destino padrão: `runs/busca_extras_protocolos/`.

Cada subpasta de protocolo contém `resultados.csv`, `medias.csv`, `status.csv`
e os logs/JSONs individuais, atualizados a cada execução. Na raiz, os mesmos
CSVs são consolidados ao terminar/interromper um protocolo. As médias da raiz
mantêm way/shot/query separados e os ganhos usam o controle do mesmo protocolo.
Os embeddings são compartilhados entre os protocolos na pasta `_work`.

Repetir o mesmo comando retoma a grade e pula as execuções já concluídas,
desde que configuração e versão de código sejam as mesmas. Para consolidar
os resultados já salvos sem treinar:

```bash
bash buscar_extras_protocolos.sh --summarize-only
```

Os resultados anteriores em `busca_extras_auto` são preservados. A grade nova
tem seus próprios controles e não importa automaticamente execuções antigas.
