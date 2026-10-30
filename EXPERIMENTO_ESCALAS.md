# Flowers e Cars: transferência do ajuste Micro

Grade preparada, ainda sem resultados científicos desta bateria..

- Sem extras, sem banco reservado e rankings restritos ao episódio..
- DGCG ajustado no Micro, kNN-union (k=5) e ProtoNet sem grafo..
- Padrão: 5-way, shots 1/5, query10, seeds 7/17/27; 108 execuções..
- ResNet-18 ImageNet congelada, 128px, projeção256, dois saltos SGC, 2000 episódios de treino, validação200 a cada200 e teste1000, conforme defaults atuais..
- Checkpoint selecionado pela validação em cada execução; nenhum ajuste usa teste..

## Bases e divisões de classes

| Base | OpenML Micro / Mini / Extended | Split Micro | Split Mini/Extended |
|---|---|---|---|
| Flowers | 44239 / 44283 / 44318 | 10/5/5 | 51/25/26 |
| Cars | 44245 / 44289 / 44323 | 10/5/5 | 98/49/49 |

Fontes oficiais: [Flowers](https://meta-album.github.io/datasets/FLW.html), [Cars](https://meta-album.github.io/datasets/CRS.html) e [versões Meta-Album](https://meta-album.github.io/)..

Micro contém20 classes e40 imagens/classe; Mini mantém40 imagens/classe e amplia as classes; Extended usa as classes elegíveis com todas as imagens disponíveis, potencialmente desbalanceadas.. Splits acima são decisões desta bateria, aproximadamente50%/25%/25%, não partições oficiais do Meta-Album.. O pipeline valida a quantidade real de classes ao carregar..

## O que significa melhor no Micro

Por base e protocolo, escolher a maior acurácia de validação seed7 entre os quatro intervalos de grau e quinze densidades dos CSVs históricos sem extras, em outputs/analise_experimentos_20261026/fontes/.. Não escolher o maior teste nem o way com maior acurácia.. Não usar os controles da bateria de extras, que reservam imagens..

| Base | Shot | Densidade escolhida | Validação Micro (%) |
|---|---:|---:|---:|
| Flowers | 1 | 0,125 | 89,7600 |
| Flowers | 5 | 0,050 | 95,4000 |
| Cars | 1 | 0,075 | 28,3200 |
| Cars | 5 | 0,025 | 36,9600 |

Empates: prefere grau4:6, depois graus crescentes e densidades crescentes.. Todos os empates são registrados no manifesto.. Os alvos são fixos entre versões e seeds; não há nova busca em Mini/Extended.. Cada modelo é treinado na própria versão, não são transferidos pesos do Micro..

A escolha depende da seed7 histórica: a repetição Micro/seed7 não é confirmação independente da seleção.. As seeds17/27 fornecem repetições adicionais.. A mesma seed controla divisão de classes, inicialização e episódios; o desvio entre seeds inclui todas essas fontes de variação, não apenas treinamento.. Métodos são pareados dentro da mesma base/versão/protocolo/seed..

Mudar Micro para Mini altera também as classes de avaliação; não interpretar diferenças como efeito causal puro do volume de treino.. Mesmo Mini/Extended deve ter as listas de classes salvas conferidas antes de assumir identidade das partições.. O orçamento de episódios permanece fixo; isso não iguala a cobertura das imagens entre versões..

## Executar na VM

Na raiz do projeto, usando o ambiente com PyTorch/GPU e dependências instaladas..

```bash
conda activate ./.conda-env
bash buscar_escalas.sh --dry-run
bash buscar_escalas.sh --device cuda
```

O script prefere o conda ativo, depois venv, depois python3; FSC_PYTHON permite indicar outro executável.. Downloads das versões ausentes são feitos pelo loader OpenML existente; embeddings ficam compartilhados por base/versão em runs/comparacao_escalas/_work/..

```bash
# Cinco seeds: 180 execuções
bash buscar_escalas.sh --device cuda --seeds 7 17 27 37 47 --output-root runs/comparacao_escalas_5seeds

# Os seis protocolos anteriores: 324 execuções
bash buscar_escalas.sh --device cuda --ways 2 5 --shots 1 5 10 --output-root runs/comparacao_escalas_6protocolos

# Consolidar sem treinar
bash buscar_escalas.sh --summarize-only
```

Usar novamente o mesmo comando retoma resultados completos compatíveis.. Manifesto fixa configuração, código e hashes dos CSVs de seleção; alteração exige outra pasta.. Para no primeiro erro, preserva resultados e atualiza os CSVs após cada execução/interrupção.. Não executar duas instâncias na mesma pasta..

Saídas: resultados.csv, status.csv, medias.csv, manifesto e, por execução, comando, log, checkpoint e métricas JSON com classes de cada split.. Médias mantêm base/versão/protocolo/método separados e incluem desvio amostral entre seeds; campos concluidos/planejados sinalizam grupos parciais.. Uma seed isolada deixa desvio vazio.. Não confundir esse desvio com IC episódico..

--quick é apenas diagnóstico e exige pasta separada; não usar seus resultados como bateria científica..

## Verificação local

Quatro testes automatizados passaram, incluindo seleção independente do teste, argumentos reais do modelo, dry-run sem escrita e médias/desvio por versão.. Três execuções reais reduzidas em Flowers Micro, seed17, CPU, concluíram para DGCG, kNN-union e sem grafo em runs/smoke_escalas_cache_20261030/.. São testes de funcionamento; Mini/Extended e GPU ainda não foram executados nesta verificação..
