# Homofilia e ganho dos grafos com o classificador treinado

O objetivo e verificar se a associacao positiva entre homofilia e ganho do
grafo, mostrada na pagina 12 do PDF das baterias 1 a 7, permanece com treino.

## Protocolo

- Dez bases Micro, os mesmos conjuntos do PDF.
- ResNet-18 ImageNet congelada, imagens 128x128, embeddings de 512 valores.
- 5-way, 1 ou 5-shot, 10 queries por classe; seed 7.
- Sem grafo, kNN-union k=5 euclidiano e DGCG grau medio 4-6/RBO.
- Dois saltos; sem extras, GRaNDe ou ponderacoes adicionais.
- Treino de Theta 512->256, LayerNorm e escala; AdamW e defaults existentes.
- Treino2000; validacao200 a cada200; teste1000; checkpoint pela validacao.
- Homofilia em 200 episodios, seed+909; teste em seed+303.

Para cada base e grafo, o eixo X e a homofilia das linhas de query: soma dos
pesos de arestas que ligam imagens de mesma classe dividida pela soma de
todos os pesos nessas linhas, sem autolacos. Os pesos sao somados entre os
200 episodios antes da divisao. O eixo Y e a acuracia do grafo menos a do
sem grafo treinado, na mesma base/shot/seed, em pontos percentuais.

Cada painel completo tem 20 pontos: dez bases vezes dois grafos. Pearson
e calculado em cada shot e seed separadamente. Nao se calcula p-valor
tratando esses 20 pontos como independentes: os dois grafos compartilham
a mesma base e o mesmo controle sem grafo. Correlacao nao demonstra causa.

## Por que o padrao pode ser testado sem repetir treinamentos

O modelo constroi A a partir de X congelado, antes de Theta. Assim,
para o mesmo episodio e as mesmas features, A e sua homofilia sao identicos
antes e depois do treino da projecao. Nao estamos medindo homofilia de
X Theta, nem construindo uma topologia nova no espaco projetado.

Ja existem as acuracias treinadas das dez bases e a homofilia das arestas
correspondentes em CPU, seed7. O comportamento padrao do script cruza essas
fontes auditadas. Isso testa a relacao homofilia x ganho COM treino sem
repetir os treinos nem a classificacao. A referencia sem treino do PDF e
recalculada das fontes usando h_query: r=0.6472814894 e r=0.8612571005.

```bash
bash scripts/buscar_homofilia_treinada.sh --dry-run
bash scripts/buscar_homofilia_treinada.sh
```

Esse modo nao usa GPU, mesmo que a opcao de dispositivo da nova avaliacao
tenha default cuda: os resultados reutilizados sao explicitamente CPU.
Usa apenas a seed7 completa. Os rotulos das queries foram usados para
diagnosticar arestas ja construidas; nunca para construir o grafo/prever.

As fontes sao `res_e1.jsonl` (raw, trainfree=false, tag vazia) e
`res_topo_tf.jsonl` (raw, tag vazia, union/DGCG4-6). Os hashes e as medias
sao verificados contra o inventario e a planilha principal antes da analise.

## Opcional: reavaliar os modelos e medir as arestas novamente

```bash
bash scripts/buscar_homofilia_treinada.sh --evaluate-models --device cuda \
  --output-root runs/homofilia_treinada_resnet_nova_avaliacao --dry-run
```

O script procura checkpoints compativeis nos diretorios de execucao e na
pasta dos testes anteriores. Pode indicar outros caminhos com
`--checkpoint-roots PASTA1 PASTA2`. Rejeita ViT, outros tamanhos/dimensoes,
normalizacoes extras, identidade, outros protocolos, extras, densidades,
outros graus e dispositivos de treino diferentes. Carrega checkpoints
locais confiaveis do projeto e confere tambem as classes do split.

Se faltar checkpoint, o comportamento e parar antes de extrair ou treinar.
Um CSV de acuracias sozinho nao permite reconstruir os pesos treinados.
Para autorizar somente os treinamentos faltantes:

```bash
bash scripts/buscar_homofilia_treinada.sh --evaluate-models --device cuda \
  --train-missing --output-root runs/homofilia_treinada_resnet_nova_avaliacao
```

O maximo e 60 modelos: dez bases x dois shots x tres metodos. Sao 40 grafos
diagnosticados e 20 controles sem grafo. Classes 10/5/5; apenas rotulos do
suporte formam os prototipos. As queries supervisionam a loss no treino e
as metricas no teste. O diagnostico usa apenas classes de teste.

Para verificar funcionamento com menos episodios, usar pasta separada:

```bash
bash scripts/buscar_homofilia_treinada.sh --evaluate-models --train-missing \
  --device cpu --quick --datasets flowers_micro --shots 1 \
  --output-root runs/smoke_homofilia_treinada
```

Esse modo usa treino200/val50/test200/hom20 e nao e comparavel ao resultado
final. Para novas seeds, apenas no modo --evaluate-models, usar --seeds.
Uma reavaliacao CUDA nao e uma reproducao exata do ambiente CPU do PDF.

## Arquivos de saida

Pasta padrao: `runs/homofilia_treinada_resnet/`.

- `resultados.csv`: 60 resultados de classificacao e origem das informacoes.
- `pontos_homofilia.csv`: 40 pontos com homofilia, teste e ganho vs controle.
- `correlacoes.csv`: Pearson por shot/seed e a referencia sem treino do PDF.
- `medias.csv`: medias por metodo/shot nas bases concluidas.
- `status.csv`: condicoes concluidas/pendentes.
- `homofilia_treinada_seed7.png`: os dois paineis do grafico.
- `manifest.json`: configuracao, definicao, hashes e fontes.

Cada condicao tem `diagnostic.json`. No modo padrao, os campos
classification_reused e homophily_reused indicam o reaproveitamento.
Na nova avaliacao, checkpoint_reused e o hash identificam o modelo usado.
Repetir o comando retoma condicoes concluidas na nova avaliacao. Mudancas
de codigo/protocolo requerem outra pasta para nao misturar resultados.

Os CSVs e o grafico sao gerados com percentuais em escala 0-100; diferencas
de acuracia sao pontos percentuais. Homofilia alta pode estar associada a
um ganho maior sem que o ganho medio da bateria seja positivo.

## Resultado conferido com os dados existentes

| Protocolo | Pearson sem treino (referencia) | Pearson com treino |
|---|---:|---:|
| 1-shot | 0.6472814894 | 0.6930406192 |
| 5-shot | 0.8612571005 | 0.7627671435 |

Cada valor usa 20 pontos, duas topologias nas mesmas dez bases, seed7/CPU.
A associacao positiva permanece. A homofilia e a mesma porque A nao depende
de Theta; mudam as acuracias e os ganhos no eixo Y. Isso nao e uma medicao
nova de um grafo aprendido. A comparacao com/sem treino tambem altera a
dimensao final de 512 para 256, conforme o protocolo original do PDF.

88 testes automatizados passaram. Uma execucao real reduzida em Flowers
Micro/CPU/1-shot verificou os tres metodos no modo --evaluate-models --quick,
e a retomada foi conferida sem repetir treinos. Esses testes de funcionamento
nao fornecem os valores da tabela acima, que vem das fontes historicas.

## Modo `--training-curve`: homofilia durante o treino

```bash
bash scripts/buscar_homofilia_treinada.sh --training-curve --device cuda
```

Treina do inicio as 60 condicoes (10 bases Micro, 1/5-shot, sem grafo,
kNN-union e DGCG 4-6, seed 7) e ignora checkpoints existentes. A pasta padrao
passa a ser `runs/homofilia_curva_treino/`. Os tres metodos de cada base/shot
recebem a mesma inicializacao e a mesma sequencia de episodios de treino; o
pareamento e conferido pelo hash dos exemplos de cada episodio.

O que e registrado, alem dos arquivos do modo padrao:

- `curva_episodios.csv`: um registro por episodio de treino com grafo
  (2.000 por condicao), com acuracia do episodio, acuracia do sem grafo no
  mesmo episodio e as somas `same_query`/`weight_query` da homofilia.
- `curva_blocos.csv`: um registro por bloco entre validacoes (200 episodios),
  com homofilia do bloco, ganho medio no treino e ganho no teste. O teste de
  cada bloco usa os 1.000 episodios fixos de teste com o estado do modelo
  naquele passo.
- `primeira_vantagem.csv`: por condicao, primeiro bloco em que o grafo fica a
  frente do sem grafo no treino e no teste, correlacoes de Pearson e a
  homofilia agregada dos 2.000 episodios de treino.
- `correlacoes_entre_bases.csv`: Pearson entre a homofilia de treino e o
  ganho final de teste, por shot e por grafo.

A avaliacao no teste a cada bloco serve apenas ao diagnostico. O checkpoint
final continua sendo escolhido pela validacao. A homofilia usa os rotulos
das queries somente depois de o grafo estar montado.

Os resultados da execucao de 08/11/2026 na VM estao em
`resultados/homofilia_treino_20261108/`.
