# ProtoNet + SGC Few-Shot

O script executa classificação few-shot episódica com ProtoNet e SGC. Cada grafo é construído com os exemplos de suporte e consulta do episódio, portanto o método é transdutivo. Os rótulos das consultas são usados apenas no cálculo da loss e na avaliação.

## Organização do código

O programa foi separado por responsabilidade. `protosgc_fewshot.py` continua sendo o arquivo que você executa; o código do experimento fica no pacote `proto_sgc/`.

```text
few_shot/
├── protosgc_fewshot.py       # Entrada do programa e compatibilidade de imports
├── proto_sgc/
│   ├── __init__.py          # Identifica a pasta como um pacote Python
│   ├── experiment.py        # Conecta as etapas do experimento
│   ├── config.py            # Argumentos, valores padrão e validações
│   ├── data.py              # Imagens e rótulos: pasta local ou OpenML
│   ├── features.py          # ResNet-18 congelada e cache de embeddings
│   ├── episodes.py          # Divisão de classes e amostragem dos episódios
│   ├── model.py             # ProtoSGC: projeção, propagação e protótipos
│   ├── graphs/
│   │   ├── __init__.py      # Identifica o subpacote de algoritmos de grafo
│   │   ├── knn.py           # kNN euclidiano/cosseno binário e RBF opcional
│   │   ├── dgcg.py          # Rankings, correlações, DGCG e DGCG+
│   │   └── grande.py        # Distâncias e cálculo do grau GRaNDe
│   ├── training.py          # Treino, avaliação, métricas e checkpoints
│   └── runtime.py           # Sementes, dispositivo e leitura de tensores
├── rodar_base.sh            # Executa comparações e gera a tabela LaTeX
├── requirements.txt
└── documentacao.md
```

### Ordem sugerida de leitura

1. Comece em [`experiment.py`](proto_sgc/experiment.py), na função `main()`. Ela apresenta o fluxo completo: configura a execução, carrega dados, extrai embeddings, divide classes, cria o amostrador e treina cada variante.
2. Leia [`episodes.py`](proto_sgc/episodes.py). `split_classes()` separa as classes de treino, validação e teste; `FeatureEpisodeSampler.sample()` escolhe o suporte e as consultas de um episódio. A estrutura `Episode` reúne esses quatro tensores.
3. Em [`model.py`](proto_sgc/model.py), leia `ProtoSGC.forward()` e depois `encode_episode()`. O primeiro transforma o suporte em protótipos e produz os logits das consultas; o segundo calcula as representações usadas nessa classificação.
4. Leia `train_model()` e `evaluate()` em [`training.py`](proto_sgc/training.py). Aqui os rótulos das consultas entram na loss e na acurácia, os parâmetros são atualizados e o melhor estado é escolhido pela validação.
5. Aprofunde a construção do grafo em [`knn.py`](proto_sgc/graphs/knn.py), [`dgcg.py`](proto_sgc/graphs/dgcg.py) e [`grande.py`](proto_sgc/graphs/grande.py), conforme a variante que estiver estudando.
6. Consulte [`data.py`](proto_sgc/data.py) e [`features.py`](proto_sgc/features.py) para entender a preparação das imagens. Use [`config.py`](proto_sgc/config.py) como referência das opções do experimento.

### O que acontece com os dados

```text
Imagens + labels.csv (ou OpenML)
    → ResNet-18 congelada
    → embeddings agrupados por classe
    → classes de treino / validação / teste
    → episódio: suporte + consultas
    → representações calculadas pelo ProtoNet ou Proto-SGC
    → protótipos calculados a partir do suporte
    → logits das consultas
    → loss e acurácia usando os rótulos das consultas
```

Um **embedding** é o vetor de atributos de uma imagem. A ResNet-18 produz 512 valores por imagem e fica congelada: o meta-treino trabalha com esses vetores já extraídos. Os embeddings de todas as classes podem ser extraídos juntos; a separação por classes acontece antes da amostragem dos episódios de treino, validação e teste.

Um **episódio** é uma tarefa pequena de classificação. No padrão 5-way/1-shot/5-query, ele contém cinco classes, cinco exemplos de suporte no total e 25 consultas. Assim, `support_x` tem formato `[5, 512]`, `query_x` tem `[25, 512]`, e o grafo do Proto-SGC possui 30 nós. Os rótulos são remapeados para `0, 1, 2, 3, 4` dentro desse episódio, e suporte e consultas são amostrados sem reposição.

Um **protótipo** é a média das representações dos exemplos de suporte de uma classe. `forward()` compara cada consulta com os protótipos e devolve uma matriz de logits com formato `[25, 5]` nesse exemplo. Esse método recebe `support_y`, mas não recebe `query_y`.

### Como o modelo usa os arquivos de grafo

`ProtoSGC` guarda as opções da execução e os parâmetros treináveis. Seus métodos `knn_adjacency()`, `dgcg_adjacency()` e `grande_degree()` encaminham os cálculos aos arquivos de `graphs/`. Isso mantém a interface anterior do modelo e permite ler cada algoritmo separadamente. As funções de `graphs/` recebem seus parâmetros explicitamente.

Dentro do modelo, `graph_adjacency()` escolhe a topologia e aplica os pesos opcionais. Depois, `normalized_graph_adjacency()` adiciona autolaços e normaliza a matriz. `encode_episode()` usa essa matriz na propagação.

Notação usada nos comentários:

| Símbolo | Significado |
| --- | --- |
| `X` | Embeddings de suporte e consultas concatenados. |
| `A` | Adjacência do episódio, antes dos autolaços e da normalização. |
| `A_hat` | Adjacência com autolaços e normalizada. |
| `Theta` | Projeção linear treinável, chamada `theta` no código. |
| `K` | Número de propagações, definido por `--sgc-hops`. |

Sem GRaNDe, o Proto-SGC propaga `X` com `A_hat` por `K` saltos e depois aplica `Theta`. Com GRaNDe, primeiro calcula `X Theta`, usa esses vetores projetados para calcular os graus e então propaga. A topologia continua sendo construída a partir dos embeddings de entrada. Em ambos os casos, aplica LayerNorm e normalização L2 antes de formar os protótipos. No baseline ProtoNet, cada vetor passa pela projeção e pelas normalizações sem propagação em grafo.

### Execução e uso em notebooks

Todos os comandos deste documento continuam usando `python protosgc_fewshot.py`. O `rodar_base.sh` continua chamando esse mesmo arquivo. Ao copiar o projeto para outra pasta ou máquina, leve também a pasta `proto_sgc/`, pois o arquivo de entrada agora importa os módulos dela. Execute os comandos a partir da raiz do projeto.

Em novos notebooks, você pode importar diretamente de cada módulo:

```python
from proto_sgc.model import ProtoSGC
from proto_sgc.episodes import FeatureEpisodeSampler, split_classes
from proto_sgc.training import evaluate
```

O acesso anterior, como `from protosgc_fewshot import ProtoSGC`, também continua disponível. Os argumentos da linha de comando, as sementes, o cache de embeddings, o formato dos checkpoints salvos pelo programa e as mensagens de resultados foram preservados.

O intervalo padrão de grau do DGCG é `(4.0, 6.0)` tanto pela linha de comando
quanto ao criar `ProtoSGC` diretamente.

## Instalação

Para explorar o código passo a passo no navegador, consulte o [simulador interativo](simulador/README.md). Ele permite alterar parâmetros, acompanhar cada episódio antes/depois da atualização e abrir uma aba com a matemática e os valores reais daquele passo.

```bash
pip install torch torchvision numpy pandas pillow tqdm openml
```

Consulte todas as opções com:

```bash
python protosgc_fewshot.py --help
```

## Execução rápida

Teste completo do pipeline, usando a topologia `knn-union`:

```bash
python protosgc_fewshot.py --quick
```

O modo `--quick` executa 200 episódios de treino, 50 de validação e 200 de teste.

Sem `--data-root`, o script baixa automaticamente o Flowers Micro pelo OpenML, usando o ID `44239`.

Para utilizar um Meta-Album já baixado:

```bash
python protosgc_fewshot.py \
  --data-root /caminho/para/o/dataset \
  --quick
```

A pasta deve conter `labels.csv` e as imagens, normalmente dentro de `images/`.

## Topologias disponíveis

| Topologia        | Descrição                                                                              |
| ---------------- | -------------------------------------------------------------------------------------- |
| `knn-out`        | Cada nó agrega os seus `k` vizinhos pela métrica escolhida (padrão: euclidiana).        |
| `knn-in`         | Inverte as arestas do kNN; cada nó agrega os nós que o escolheram como vizinho.        |
| `knn-union`      | União entre `knn-out` e `knn-in`, produzindo um grafo simétrico. É a opção padrão.     |
| `knn-reciprocal` | Mantém apenas relações kNN mútuas.                                                     |
| `dgcg`           | Conecta nós de acordo com a correlação entre suas listas ranqueadas de vizinhos.       |
| `dgcg-plus`      | Usa a mesma topologia do DGCG, mas pondera as arestas pelo escore de vizinhança mútua. |
| `all`            | Executa todas as topologias anteriores, sequencialmente e com os mesmos episódios.     |

Exemplo com uma topologia específica:

```bash
python protosgc_fewshot.py \
  --graph-type knn-reciprocal \
  --knn 5 \
  --quick
```

Comparação de todas as topologias:

```bash
python protosgc_fewshot.py --graph-type all --quick
```

## DGCG e DGCG+

DGCG com os rankings euclidianos padrão e correlação RBO:

```bash
python protosgc_fewshot.py \
  --graph-type dgcg \
  --dgcg-metric euclidean \
  --dgcg-correlation rbo \
  --quick
```

DGCG+:

```bash
python protosgc_fewshot.py \
  --graph-type dgcg-plus \
  --dgcg-metric euclidean \
  --dgcg-correlation rbo \
  --quick
```

Por padrão, o limiar do DGCG é escolhido automaticamente para aproximar o grau médio do intervalo configurado em `--dgcg-target-degree`:

```bash
--dgcg-target-degree 4 6
```

Também é possível definir um limiar manual:

```bash
--dgcg-threshold 0.3
```

Correlações disponíveis:

```text
rbo
jaccardk
jaccard-median
jaccard-max
```

### Métrica do kNN

Use `--knn-metric cosine` para selecionar vizinhos por cosseno nas variantes
`knn-out`, `knn-in`, `knn-union` e `knn-reciprocal`. O padrão permanece
`--knn-metric euclidean`. Esta opção é independente de `--dgcg-metric` e de
`--grande-metric`. As arestas continuam binárias por padrão; selecionar cosseno
não restaura a ponderação exponencial do antigo script dos Downloads.
No simulador, use o seletor **Métrica dos vizinhos kNN**. Checkpoints kNN por
cosseno recebem o sufixo `cosine` para distinguir as duas configurações.

## Referencias extras reservadas, sem aumentar o grafo

Experimento opcional para controlar quantas imagens nao rotuladas das mesmas
classes entram somente nos rankings durante validacao e teste:

```bash
python protosgc_fewshot.py \
  --graph-type dgcg \
  --n-way 5 --n-shot 1 --n-query 5 \
  --dgcg-extra-references 50 \
  --work-dir runs/dgcg_extras
```

`--dgcg-extra-references N` define o TOTAL de extras por episodio, nao por classe.
Com 5-way e N=50, sao 10 referencias por classe selecionada. Quando N nao e
divisivel por n-way, a distribuicao difere em no maximo uma imagem por classe.

O banco e fixo por particao, determinado pela seed: por padrao reserva 20 imagens
por classe, configuravel por `--dgcg-extra-bank-per-class`. Essas imagens nunca
sao amostradas como suporte ou consultas em qualquer episodio dessa particao.
Validacao e teste tem bancos independentes de suas proprias classes. O treino
nao usa esses bancos. A flag antiga `--dgcg-train-global-rankings` pode ser
combinada para manter contexto global somente no treino, se desejado.

Nas bases Micro com 40 imagens por classe, o banco de 20 deixa outras 20 para
suporte/consultas e permite ate 100 extras em 5-way. Banco+shot+query precisa
caber em cada classe; uma configuracao impossivel falha antes do treino.

As referencias usadas num episodio sao exclusivamente das suas classes, com
prefixos do banco reservado. Os rotulos reais servem para organizar esse
protocolo de amostragem; nao sao fornecidos a funcao de rankings/correlacoes.
Isto assume um banco nao rotulado das classes-alvo disponivel na avaliacao.

Rankings usam `extras + suporte + consultas`, limitados por L; correlacoes ainda
usam top-k. Grafo, candidatos, pesos e propagacao continuam contendo apenas
suporte+consultas. Nao ha propagacao nas imagens extras. O modo nao combina com
`--dgcg-reference-train-rankings`, pois sao contextos de avaliacao distintos.

Para um controle comparavel use `--dgcg-extra-references 0`. Ele reserva o MESMO
banco, mas nao acrescenta extras aos rankings. Use as mesmas seeds, tamanho de
banco e protocolo em N=0/10/25/50/100: os episodios elegiveis permanecem iguais.
Omitir a flag preserva o modo original sem reserva; portanto esse resultado nao
e o mesmo controle N=0 do banco reservado.

Cada chamada executa treino, selecao por validacao e teste; mudar N pode mudar
o checkpoint selecionado. Nao e uma avaliacao de varios N em um checkpoint fixo.
Os checkpoints guardam IDs da reserva e do pool episodico e usam sufixos
`extraN-bankM`, evitando misturar com resultados anteriores. O resumo registra
tambem esse modo. Sem a flag, o comportamento anterior permanece.

## Rankings com treino como referencia em todas as fases

Para usar todas as imagens de treino como contexto dos rankings, inclusive na
avaliacao, execute:

```bash
./venv/bin/python protosgc_fewshot.py \
  --graph-type dgcg \
  --dgcg-reference-train-rankings \
  --dgcg-list-size 200 \
  --work-dir runs/dgcg_referencia_treino
```

O novo modo e opcional; sem essa flag o DGCG permanece episodico.

- **Treino:** cada imagem tem sua lista ordenada contra todas as imagens de
  `splits["train"]`, incluindo a propria imagem e os demais exemplos do episodio.
  Exemplos episodicos nao sao duplicados. Sao guardados no maximo 200 itens por
  ranking quando `--dgcg-list-size 200` e usado.
- **Validacao/teste:** o universo contem apenas todas as imagens de treino e o
  suporte+consultas do episodio atual. Sao calculados rankings para os nos do
  episodio contra esse universo; depois suas listas sao comparadas por RBO ou
  Jaccard. Imagens de outros episodios/classes de avaliacao nao sao referencias.
- **Grafo:** candidatos, limiar, propagacao e pesos DGCG+ continuam restritos aos
  nos do episodio. As imagens extras servem somente como IDs nos rankings.
- **Rotulos:** nao entram nos rankings/correlacoes. As consultas do episodio
  participam como embeddings nao rotulados, seguindo avaliacao transdutiva.
  Rotulos de teste sao usados somente para medir desempenho.

`L` e a quantidade de itens armazenados em cada lista, nao uma amostra de 200
imagens de treino: a busca compara todas as referencias e guarda os primeiros
`L`. `--dgcg-top-k` continua limitando a profundidade da correlacao (default 40).
A referencia e congelada nos embeddings da ResNet; a avaliacao nao atualiza
parametros. Checkpoints e resumos recebem `reference-train-rankings` para
distinguir esse modo do experimento antigo que usava contexto global so no treino.

## Rankings globais somente no treinamento

Para comparar o DGCG episodico com uma variante que calcula suas correlacoes
usando rankings de todas as imagens da particao de treino:

```bash
./venv/bin/python protosgc_fewshot.py \
  --graph-type dgcg \
  --dgcg-metric euclidean \
  --dgcg-train-global-rankings \
  --work-dir runs/dgcg_rankings_treino \
  --quick
```

Para a referencia episodica, execute com os mesmos argumentos e semente,
removendo `--dgcg-train-global-rankings` e usando outra pasta em `--work-dir`.
A opcao e desativada por padrao. Tambem aceita `dgcg-plus` e `all`; em `all`,
somente as variantes DGCG mudam. ProtoNet e uma topologia kNN isolada nao
aceitam essa flag.

O fluxo da variante e:

1. Concatena somente embeddings de `splits["train"]`, mantendo um ID por imagem.
2. Compara todas essas imagens para gerar rankings de tamanho
   `min(--dgcg-list-size, numero de imagens de treino)`, incluindo o proprio item.
3. Em cada episodio, seleciona os rankings das imagens de suporte e consulta
   pelos IDs e compara seus prefixos com RBO/Jaccard. Itens externos ao episodio
   continuam nos rankings e contribuem para as intersecoes.
4. Usa essa matriz de correlacoes para selecionar as arestas entre os nos do
   episodio. A lista de candidatos e o procedimento de limiar continuam locais.
5. Validacao e teste usam exclusivamente rankings dos respectivos episodios.

Assim, o top-k da correlacao no treino pode superar o numero de nos do episodio,
mas o grafo continua contendo somente suporte e consultas. Os pesos nativos do
DGCG+ continuam episodicos: esta primeira ablacao muda apenas as correlacoes
usadas na selecao. O grau GRaNDe, inclusive seu modo RBO em `X Theta`, e o RBF
opcional tambem continuam sendo calculados nos vetores do episodio.

Os rankings sao calculados uma vez por treinamento de uma topologia, em CPU,
e mantidos na memoria. A busca e exata, em blocos: tem custo de comparar todos
os pares de treino, mas nao guarda uma matriz global de correlacoes N x N.
Guarda apenas os rankings N x L e calcula correlacoes para os nos de cada
episodio. Em empates de distancia, o ranking global usa a ordem estavel dos IDs,
sempre colocando o proprio item primeiro.

O identificador `(classe, indice da imagem)` serve apenas para localizar o
ranking, sem usar igualdade de rotulos na correlacao. Nao procura imagens por
igualdade dos embeddings, evitando confundir exemplos com vetores duplicados.
Os arquivos de checkpoint recebem o sufixo `train-global-rankings` e a flag e
registrada em `args`. Nao ha carregamento automatico desses rankings do disco.

Essa variante usa contextos diferentes no treino e na avaliacao. Usar rankings
globais tambem na validacao/teste seria outro protocolo: cada episodio teria
acesso indireto a outras imagens da sua particao. Nao esta habilitado aqui.

A implementacao de pre-calculo esta em
[`training_rankings.py`](proto_sgc/graphs/training_rankings.py); o ponto que
fornece as correlacoes ao modelo esta em [`training.py`](proto_sgc/training.py).

## Ablação opcional com pesos RBF

Os grafos `knn-*` e `dgcg` são binários por padrão. `dgcg-plus` usa seus
pesos próprios de vizinhança mútua. A opção abaixo serve apenas para uma
ablação: preserva as arestas escolhidas e substitui seus valores pelo RBF da
distância de cosseno.

```bash
python protosgc_fewshot.py \
  --graph-type dgcg \
  --cosine-rbf-weight \
  --graph-temperature 0.2 \
  --quick
```

Ela também pode ser aplicada às topologias kNN, ao DGCG+ ou a todas as topologias.

## GRaNDe

GRaNDe modifica a normalização da adjacência, não a topologia. Pode ser combinado com qualquer tipo de grafo:

```bash
python protosgc_fewshot.py \
  --graph-type all \
  --grande \
  --grande-metric euclidean \
  --quick
```

Métricas disponíveis:

```text
euclidean
cosine
rbo
```

Exemplo com cosseno:

```bash
python protosgc_fewshot.py \
  --graph-type dgcg-plus \
  --grande \
  --grande-metric cosine \
  --quick
```

## Baseline sem grafo

Para executar apenas o ProtoNet:

```bash
python protosgc_fewshot.py \
  --model protonet \
  --quick
```

O baseline `protonet` não aceita `--graph-type all`, `--grande` ou `--cosine-rbf-weight`.

## Configuração dos episódios

Exemplo de experimento 5-way, 5-shot, com 10 consultas por classe:

```bash
python protosgc_fewshot.py \
  --n-way 5 \
  --n-shot 5 \
  --n-query 10 \
  --graph-type knn-union
```

Para outro dataset OpenML, informe seu ID e a divisão das classes:

```bash
python protosgc_fewshot.py \
  --openml-id ID_DO_DATASET \
  --class-split CLASSES_TREINO CLASSES_VALIDACAO CLASSES_TESTE \
  --graph-type all
```

## Resultados

Por padrão, os embeddings, checkpoints e resultados são armazenados em:

```text
runs/proto_sgc_meta_album/
```

Outra pasta pode ser definida com:

```bash
--work-dir runs/meu_experimento
```

Ao final, o script informa a acurácia média, o intervalo de confiança de 95%, a loss e o caminho do melhor checkpoint de cada topologia.

## Nova bateria: métricas e grau versus protocolo

`buscar_metricas_graus.sh` é uma bateria separada. Não sobrescreve os resultados
nem o PDF de outubro. Não usa referências extras nem rankings globais:
mantém o protocolo episódico original, ResNet-18 congelada, k=5, euclidiano,
arestas binárias, dois saltos SGC e GRaNDe desligado.

Métodos: kNN-in, kNN-out, kNN-mutual, kNN-union, no-graph e DGCG.
**Mutual é o mesmo método que reciprocal**, não uma sétima baseline.
`--graph-type knn-mutual` é aceito como alias; o nome antigo continua funcionando.
No-graph usa `--model protonet`: mesma projeção e classificação por protótipos,
mas sem propagação entre imagens.

```bash
# Confira os comandos e a quantidade antes de rodar (não baixa nem treina).
bash buscar_metricas_graus.sh --dry-run

# Bateria completa na GPU; retoma execuções concluídas automaticamente.
bash buscar_metricas_graus.sh --device cuda

# Teste menor, em uma base e um protocolo (ainda são nove treinamentos).
bash buscar_metricas_graus.sh --datasets flowers_micro --ways 5 --shots 1 --queries 5 --quick --output-root runs/metricas_smoke

# Escolha de intervalos mais detalhada ou repetição com mais seeds.
bash buscar_metricas_graus.sh --degrees 1:2 2:4 4:6 6:8 8:12 --seeds 7 17 27 --output-root runs/metricas_repeticoes

# Refaz somente as tabelas/gráficos dos resultados já existentes.
bash buscar_metricas_graus.sh --summarize-only
```

O shell respeita o Python do ambiente Conda ativo. Para outro interpretador,
use `FMG_PYTHON=/caminho/python bash buscar_metricas_graus.sh ...`.

Grade padrão: dez bases Micro, split fixo em tamanho 10/5/5, way 2/3/5,
shot 1/5/10, query 1/5/10 e seed 7. DGCG testa os intervalos 2–4, 4–6,
6–8 e 8–12 **em cada protocolo**, incluindo o default 4–6 como controle.
São **2.430 treinamentos**: dez bases × 27 protocolos × (cinco baselines + quatro
intervalos DGCG). Outra seed multiplica essa quantidade. `--quick` usa
200/50/200 episódios em vez de 2.000/200/1.000; não confundir os resultados.
Seeds diferentes também alteram a divisão das classes e os episódios,
não apenas os pesos iniciais.

### Métricas registradas

Em cada episódio de validação/teste, calcula-se acurácia, F1 macro e
acurácia balanceada (recall médio das classes). O resumo é a média dos
episódios, com IC95 normal separado para cada métrica. Não se mistura a
numeração local das classes de episódios diferentes numa única matriz de confusão.
F1 usa zero quando uma divisão é indefinida. Nos episódios atuais, todas as
classes têm o mesmo número de consultas: **acurácia balanceada coincide com
acurácia comum**, salvo arredondamento.

Cada execução, inclusive quando chamada diretamente por `protosgc_fewshot.py`,
gera um `.metrics.json` ao lado do checkpoint. Contém métricas de validação
do checkpoint selecionado e de teste desse mesmo checkpoint. Valores no JSON
estão em [0,1]; nos CSVs da nova bateria estão em porcentagem, exceto loss e
estatísticas de grafo. O critério de seleção continua sendo acurácia na
validação, nunca teste. As linhas antigas de acurácia/loss no log continuam
compatíveis com os extratores dos scripts anteriores.

### Como verificar se o intervalo muda

Saídas em `runs/busca_metricas_graus/`:

- `resultados.csv`: todos os métodos/intervalos, métricas e graus realizados.
- `melhores_graus_por_protocolo.csv`: melhor intervalo na validação por
  base/seed/way/shot/query; inclui empates e ganho de teste contra 4–6.
- `variacao_intervalos.csv`: varia um eixo por vez, mantendo base, seed e
  demais eixos fixos; mostra intervalos escolhidos, quantidade de intervalos
  distintos e amplitude/desvio dos centros e dos limites.
- `graficos/`: métricas descritivas por base, mapas de intervalo escolhido
  (shot × query em painéis de way) e grau realizado versus número de nós.
- `manifest.json`: configuração da bateria, usada para impedir mistura de
  execuções incompatíveis. Use outra pasta ao mudar a configuração.
- Uma pasta por execução, com `run.log`, `checkpoint.pt` e `metrics.json`.
  Os embeddings são compartilhados por base em `_work/`, não reextraídos
  em cada configuração.

Só se escolhe o melhor intervalo quando **todos os alvos planejados daquele
protocolo concluíram**. Em empate de validação, prefere-se 4–6 se estiver
empatado, depois o menor intervalo; o teste não desempata. Intervalos empatados
ficam explícitos: mudanças na escolha não demonstram sozinhas diferença
estatística ou um grau ideal. Gráficos gerais de métricas são descritivos e,
se a bateria estiver parcial, podem conter configurações diferentes por método.

O grau observado é a quantidade de vizinhos não nulos por linha da adjacência
usada em `A @ X`, **antes dos autolaços e da normalização**, não a soma dos
pesos nem o grau GRaNDe. Registra média, desvio, mínimo/máximo, p05/p95,
densidade dirigida `arestas / (N*(N-1))`, isolamento e variação das médias
entre episódios. Em grafos dirigidos, registra também graus por coluna.
Para intervalos-alvo DGCG, registra os limites solicitados e efetivos,
limitação pelo número de candidatos e fração de episódios cuja média ficou
fora do alvo. Empates de correlação e fallback anti-isolamento podem fazer o
grau realizado sair do intervalo. `N = way * (shot + query)`; um intervalo
4–6 não pode ser realizado literalmente quando há apenas quatro nós.

**Resultados antigos:** não é possível deduzir F1 macro apenas de acurácia
e loss agregadas. É necessário reavaliar checkpoints ou gerar novas execuções.
O PDF anterior não recebeu valores inventados; será atualizado com os novos
resultados quando a bateria for executada.

## Bateria complementar: densidades x way/shot/query

`buscar_densidades_protocolos.sh` cruza as densidades do DGCG com os mesmos
27 protocolos (way 2/3/5, shot 1/5/10 e query 1/5/10), nas dez bases Micro,
split de tamanho fixo 10/5/5 e seed 7. Não muda a bateria de graus nem usa extras.

Densidades padrão: `0.025 0.05 0.075 0.10 0.125 0.15 0.175 0.20 0.25 0.30
0.40 0.50 0.60 0.75 0.90`. São **4.050 treinamentos DGCG por seed**
(dez bases × 27 protocolos × 15 densidades), sem repetir kNNs/no-graph.
Demais hiperparâmetros e contagens de episódios mantêm os defaults anteriores.

```bash
# Conferência sem baixar/treinar:
bash buscar_densidades_protocolos.sh --dry-run

# Execução na GPU (preferencialmente depois de terminar a bateria de graus):
bash buscar_densidades_protocolos.sh --device cuda

# Exemplo menor, com densidades e protocolos personalizados:
bash buscar_densidades_protocolos.sh --datasets flowers_micro --ways 3 5 --shots 1 5 --queries 1 5 --densities 0.05 0.125 0.25 --quick --output-root runs/densidades_smoke

# Regerar somente CSVs/gráficos e reler a comparação com grau 4-6:
bash buscar_densidades_protocolos.sh --summarize-only
```

Saída padrão em `runs/busca_densidades_protocolos/`, com retomada automática,
manifesto de configuração e log/checkpoint/JSON por execução. `--fail-fast`
para no primeiro erro; em qualquer falha imprime também o fim do log.
O shell respeita Conda ativo; outro Python pode ser escolhido com `FDP_PYTHON`.

O cache OpenML é reutilizado. Para evitar reextrair embeddings, o script copia
os arquivos `features_resnet18_*.pt` de
`runs/busca_metricas_graus/_work/BASE` para sua própria `_work/BASE` quando
disponíveis; o extrator continua verificando o fingerprint. Não copia ou
compartilha os checkpoints/JSONs mutáveis da outra bateria. Altere a origem
com `--feature-cache-root`. É uma cópia de cache existente, não uma nova
extração da ResNet. Deixe a extração de origem terminar antes de copiar e
evite duas baterias simultâneas na mesma GPU.

Métricas: acurácia, F1 macro, acurácia balanceada, seus IC95, loss e estatísticas
reais de grau/densidade na validação/teste. A densidade fornecida é um
**alvo**, não garantia de densidade realizada: escores empatados e fallback
anti-isolamento podem desviá-la, sobretudo em episódios pequenos.
Com os defaults desta grade, todos os outros nós são candidatos e
`grau-alvo = rho * (way * (shot + query) - 1)`; o CSV também registra o número
efetivo de candidatos, o grau equivalente e os desvios da densidade observada.

Saídas principais:

- `resultados_densidades_protocolos.csv`: todas as execuções e métricas.
- `melhores_densidades_por_protocolo.csv`: escolha de densidade pela maior
  acurácia de validação em cada base/seed/way/shot/query, somente depois de
  terminar todos os alvos planejados daquele grupo. Empates são listados;
  a menor densidade desempata, nunca o teste.
- `variacao_densidades.csv`: mudanças da escolha ao variar cada eixo,
  mantendo os demais fixos; amplitude/desvio das densidades escolhidas.
- `graficos/`: curvas validação/teste por way, mapas shot × query da densidade
  escolhida por seed e densidade realizada versus tamanho do episódio.
  Curvas médias usam somente grupos com grade completa.

A comparação com DGCG grau 4–6 lê os `metrics.json` já concluídos em
`runs/busca_metricas_graus/` (altere com `--baseline-root`). Não treina essa
baseline outra vez e não grava nada na sua pasta. Se estiver ausente ou tiver
parâmetros/partições incompatíveis, a comparação fica sinalizada, sem ganho
calculado. Usa o mesmo protocolo, seed e orçamento de episódios; não mistura
rodadas `--quick` com completas. A pasta de saída deve ser diferente da pasta
da bateria de graus. O PDF anterior não é alterado automaticamente.
