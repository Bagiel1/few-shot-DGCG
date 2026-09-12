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
│   │   ├── knn.py           # kNN euclidiano binário e RBF opcional
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
| `knn-out`        | Cada nó agrega os seus `k` vizinhos por distância euclidiana.                          |
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
