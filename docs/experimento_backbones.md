# Comparacao de features: ResNet-18 e ViT-B/16

## O que foi implementado

`--backbone resnet18` preserva o extrator historico: ResNet-18 congelada,
entrada 128x128 e embedding de 512 valores. `--backbone vit-b-16` seleciona
ViT-B/16 congelado, entrada 224x224 e embedding de 768 valores. Nos dois,
`--weights imagenet` seleciona explicitamente `IMAGENET1K_V1`.

Nenhum dos dois extratores e treinado nos episodios. Os embeddings sao
extraidos uma vez e salvos em cache. Trocar o tamanho de entrada exige
reextracao, nao treinamento do backbone.

Inspecao local em 08/11/2026: Flowers Micro (OpenML 44239) e Cars Micro
(44245) possuem, cada um, 800 arquivos de imagem, todos com 128x128 pixels.
Isso foi medido nos arquivos locais, nao assumido a partir do modelo.
O tamanho pode ser consultado com Pillow: `Image.open(caminho).size`
devolve `(largura, altura)`.

## Transformacao das imagens

- ResNet: redimensionamento direto para um quadrado de `--image-size`
  (padrao 128), conversao em tensor e normalizacao ImageNet historica.
- ViT: transformacao oficial de `ViT_B_16_Weights.IMAGENET1K_V1`:
  redimensiona o lado menor para 256, faz recorte central de 224x224,
  converte em tensor e normaliza. `--image-size` diferente de 224 e rejeitado.

As transformacoes acontecem em memoria. Nao alteram os arquivos originais.
Ampliar uma imagem originalmente 128x128 nao recupera detalhes perdidos.
Comparar ResNet/128 com ViT/224 avalia os extratores com esses pipelines;
nao isola arquitetura de resolucao, transformacao ou receita de pre-treino.
ResNet com `--image-size 224` esta disponivel, mas continua usando resize
quadrado, sem o recorte do ViT: igualar tamanho nao iguala toda a transformacao.

## Com treino e sem treino adicional

Sem `--train-free`, o comportamento treinado continua: Theta projeta 512 ou
768 valores para `--feature-dim` (padrao 256). Theta, parametros afins da
LayerNorm e escala dos logits sao treinados por AdamW. O backbone fica
congelado. O checkpoint e escolhido exclusivamente pela acuracia de validacao.

Com `--train-free`, Theta e identidade quadrada (512x512 ou 768x768).
LayerNorm permanece com seus valores iniciais (peso 1, vies 0), a escala
permanece em seu valor inicial, e todos os parametros sao congelados.
Continuam existindo LayerNorm, normalizacao L2 e, nos metodos com grafo,
duas propagacoes. Isso reproduz o tratamento sem treino do harness historico;
nao significa usar vetores crus sem normalizacao.
Nao ha otimizador nem selecao de checkpoint. Validacao e teste sao medidos
com o mesmo estado fixo; o snapshot salvo chama-se `fixed_*_train-free.pt`.
`--feature-dim` e substituido pela dimensao de entrada nesse modo, com aviso.

Os rotulos de suporte formam os prototipos. As queries de treino fornecem
a loss no modo treinado; queries de validacao/teste fornecem as metricas.
O modelo nao recebe os rotulos das queries para construir arestas/prever.
O modo sem treino novo aceita somente episodios padrao, sem os modos de
extras ou rankings globais; combinacoes sao rejeitadas antes da extracao.

## Comandos de uma execucao (nao uma bateria completa)

Executar na raiz do projeto. Flowers Micro, 5-way/1-shot/10-query, seed 7,
kNN-union, CPU e orcamento historico 2000/200/1000 episodios:

```bash
venv/bin/python protosgc_fewshot.py \
  --openml-id 44239 --backbone vit-b-16 --weights imagenet \
  --n-way 5 --n-shot 1 --n-query 10 --seed 7 \
  --model proto-sgc --graph-type knn-union --device cpu \
  --work-dir runs/comparacao_backbones/flowers_micro
```

O mesmo comando com `--train-free` mede os vetores novos sem otimizacao.
Nesse modo os episodios de treino sao ignorados; validacao/teste continuam
com 200/1000 episodios. Para GPU, usar `--device cuda`.

Variacoes explicitas para a futura comparacao:

- ResNet historica: substituir por `--backbone resnet18` (128 por padrao).
- Cars Micro: substituir o ID por `44245` e o work-dir por
  `runs/comparacao_backbones/cars_micro`.
- Sem grafo: `--model protonet`.
- DGCG: `--model proto-sgc --graph-type dgcg`; especificar a densidade
  decidida no protocolo, pois omitir usa grau 4--6, nao densidade historica.
- 5-shot: `--n-shot 5`; seeds adicionais: `--seed 17` e `--seed 27`.

O primeiro uso do ViT com ImageNet baixa os pesos oficiais se ausentes.
`--weights random` serve somente para verificacao de funcionamento offline;
nao representa o experimento cientifico com features pre-treinadas.

## Cache, saidas e pareamento

O cache fica diretamente no work-dir informado. O nome identifica o
backbone, pesos, tamanho e, no ViT, a receita resize/crop. O conteudo registra
`feature_spec`, dimensao e fingerprint dos arquivos. Cache aleatorio inclui
tambem a seed no nome/metadados. O cache ImageNet pode ser compartilhado
entre seeds e entre modos com/sem treino, pois o backbone e fixo.
Caches ImageNet historicos da ResNet continuam sendo aceitos quando o
fingerprint confere. Nenhum resultado antigo precisa ser reconstruido.

As novas condicoes salvam snapshots e `.metrics.json` sob:

```text
<work-dir>/<backbone>_<pesos>_<tamanho>px_<preprocessamento>/
  <trained ou train-free>/<way>w<shot>s<query>q_seed<seed>/
```

Compatibilidade: ResNet/128 com treino continua salvando diretamente no
work-dir, pois as baterias antigas procuram os JSONs nesse local.
Use uma pasta propria por base e configuracao de hiperparametros.
Os JSONs e snapshots registram backbone, transformacao, modo e demais args.

As classes e episodios usam as seeds historicas: split por `seed`, treino
por `seed+101`, validacao por `seed+202`, teste por `seed+303`. Com os mesmos
registros ordenados, os indices sorteados independem da dimensao do vetor.
Cada condicao treinada aprende seu proprio estado; sem treino nao seleciona
estado por validacao. Hiperparametros nao devem ser escolhidos pelo teste.

## Verificacao de implementacao

```bash
venv/bin/python -m unittest discover -s tests -v
```

Os testes verificam saida real do ViT (768 valores), congelamento,
entrada 224, invalidacao/reuso de cache, identidade sem otimizador e
pareamento de exemplos entre dimensoes. O teste de arquitetura usa pesos
aleatorios para nao depender de rede; isso nao e um resultado experimental.

## Bateria automatica: 240 condicoes, 44 reaproveitadas e 196 novas

Entrada: `bash scripts/buscar_backbones.sh --device cuda`.
O planejador e executor ficam em `experimentos_backbones.py`.

| Eixo | Condicoes |
|---|---|
| Bases | Flowers Micro (44239), Cars Micro (44245) |
| Extratores congelados | ResNet-18 / ImageNet-1K V1 / 128px; ViT-B/16 / ImageNet-1K V1 / 224px |
| Treinamento adicional | Com treino; sem treino |
| Episodio | 5-way, 1 ou 5-shot, 10 queries avaliadas por classe |
| Metodos | Sem grafo, kNN-union, DGCG densidade historica, DGCG grau 4-6, DGCG grau 2-4 |
| Seeds | 7, 17, 27 |
| Classes | 10 de treino, 5 de validacao, 5 de teste, disjuntas |
| Extras / GRaNDe / pesos RBF | Desativados |
| kNN | k=5, distancia euclidiana, arestas binarias |
| Propagacao | Dois saltos |
| Saida de Theta com treino | 256 |
| Sem treino | Identidade 512x512 ou 768x768; LayerNorm, L2 e escala fixas |

Total da grade: 2 bases x 2 extratores x 2 modos x 2 shots x 5 metodos x 3 seeds =
240 condicoes, sendo 120 com treino e 120 sem treino.

Antes de montar o plano, o runner le a aba Resultados da planilha principal,
verifica os hashes das fontes no inventario e recalcula as medias historicas.
Na configuracao padrao CUDA/ImageNet/completa, 44 resultados da ResNet com
treino sao reaproveitados: 36 da comparacao anterior e oito dos graus 4-6 e
2-4 (duas bases, dois shots, seed7). Sao executadas 196 condicoes novas:
120 do ViT, 60 da ResNet sem treino em GPU e 16 da ResNet treinada com graus
4-6/2-4 nas seeds 17/27. Reaproveitamento nao depende de
copiar checkpoints ou de os diretorios de execucao antigos estarem na VM.
As metricas sao lidas do CSV original, com sua procedencia preservada.

Os resultados do Claude em CPU/seed7 ficam como referencias separadas no
relatorio de auditoria. Eles nao substituem resultados CUDA; o DGCG grau4-6
desses testes corresponde a uma das variantes agora incluidas, mas o dispositivo e diferente. Nao ha importacao automatica
de resultados quick, pesos aleatorios, outros dispositivos ou alvos de grafo.
A GPU historica e informada na planilha; o CSV nao registra seu modelo nem
as versoes do ambiente. Isso permanece explicito, sem inventar equivalencia
exata de hardware ou checkpoints antigos.
Micro/Mini/Extended nao sao misturados nesta bateria: somente Micro.

Com treino: 2000 episodios de treino; validacao a cada 200 episodios,
com 200 episodios por validacao; teste final com 1000 episodios, usando o
checkpoint de maior acuracia de validacao. Sao dez avaliacoes de validacao.
Sem treino: zero episodios de treino, uma avaliacao de validacao com 200
e teste com 1000, sem selecao de estado. O campo train_episodes=2000 nos
argumentos e ignorado nesse modo; o resultado registra zero executados.

Mantemos os defaults do otimizador: AdamW, learning rate 0.001,
weight decay 0.0001 e clipping 5. Batch de extracao 32; uma thread de CPU
por subprocesso, configuravel com `--cpu-threads`. Nao existe otimizacao
mista/AMP nesta bateria. As execucoes sao sequenciais na GPU.

DGCG: metrica euclidiana, RBO p=0.9, lista/candidatos maximos 200 e
profundidade 40, limitados pelo tamanho do episodio. Densidades fixas:

| Base | 1-shot | 5-shot |
|---|---:|---:|
| Flowers Micro | 0.125 | 0.05 |
| Cars Micro | 0.075 | 0.025 |

Esses valores foram conferidos contra `select_micro()` em
`experimentos_escalas.py`, que seleciona exclusivamente por validacao
historica da ResNet/128 treinada, seed 7. A bateria transfere essas mesmas
densidades para o ViT e o modo sem treino. Nao procura o melhor DGCG de
cada representacao. A densidade realizada pode diferir do alvo.

A selecao historica foi feita COM treino da projecao, na ResNet; nao foi
repetida sem treino. A variante por densidade transfere esse alvo para os
demais casos e fica identificada como tal. Para comparar configuracoes
fixas sem o custo de buscar varias opcoes pela validacao, Gabriel pediu
em 08/11/2026 mais duas variantes, nos dois extratores e modos:

| Identificador na bateria | Comando enviado ao modelo |
|---|---|
| `dgcg` | `--dgcg-target-density RHO RHO` (os valores da tabela acima) |
| `dgcg-4-6` | `--dgcg-target-degree 4 6`, sem densidade |
| `dgcg-2-4` | `--dgcg-target-degree 2 4`, sem densidade |

4-6 e 2-4 sao intervalos de GRAU MEDIO do grafo, nao obrigam cada no a ter
esse numero de vizinhos. A selecao automatica do limiar por episodio segue
igual, com empates e tratamento de nos isolados. Nao ha uma nova busca de
graus/densidades por validacao na bateria. A validacao ainda seleciona o
checkpoint de Theta no modo treinado, como em todos os outros metodos.
Os nomes, diretorios e medias das tres variantes DGCG sao separados.

### O que os resultados respondem

Comparamos o ganho do grafo sobre o sem grafo dentro de cada
base/extrator/modo/shot/seed. Tambem observamos a mudanca de acuracia com a
troca de extrator e com o treino adicional. As medias ficam separadas por
condicao, acompanhadas do desvio amostral entre seeds.

A comparacao principal usa ResNet/128 e ViT/224 com suas transformacoes
registradas. Isso avalia a troca do pipeline de features, sem isolar
exclusivamente arquitetura. A dimensao final no modo sem treino tambem
muda (512 versus 768), enquanto no treinado permanece 256.

### Rodar, medir tempo e retomar

Conferir as 240 condicoes sem executar ou criar pastas (44 linhas de reaproveitamento e 196 comandos novos):

```bash
bash scripts/buscar_backbones.sh --device cuda --dry-run
```

Rodar as primeiras 24 condicoes completas para medir o custo real na VM:

```bash
bash scripts/buscar_backbones.sh --device cuda --max-runs 24
```

Essas 24 condicoes pendentes ja fazem parte da bateria final. As 44 condicoes
historicas da ResNet treinada sao puladas e nao consomem esse limite.
Com cinco metodos, as primeiras 30 novas cobrem as condicoes pendentes em
Flowers/seed7. O limite 24 e apenas um primeiro lote, nao garante cobertura
completa dos grupos de tempo. Depois, continuar:

```bash
bash scripts/buscar_backbones.sh --device cuda
```

A retomada pula os concluidos e os resultados historicos auditados. `--max-runs` limita quantas execucoes ainda
pendentes serao feitas nesta chamada; nao limita o total do manifesto.
Mudanca de codigo, configuracao, versoes do ambiente ou GPU exige outra
pasta, para evitar misturar resultados.

Para apenas verificar funcionamento, com menos episodios e pasta separada:

```bash
bash scripts/buscar_backbones.sh --device cuda --quick \
  --datasets flowers_micro --shots 1 --seeds 7 \
  --output-root runs/smoke_backbones_gpu
```

`--quick` usa 200 episodios de treino, validacao a cada 50 com 50 episodios,
e teste com 200. Sem treino, usa apenas val50/test200. Esses resultados
nao entram na bateria completa. `--weights random` e somente para testes
offline e tambem exige pasta separada.

A RTX 5080 foi mencionada por Gabriel como possibilidade, ainda sem
confirmacao. Verificar na VM com `nvidia-smi`. O runner registra nome da
GPU, versoes PyTorch/torchvision/CUDA e executa um pequeno kernel antes
de iniciar a bateria; detectar CUDA, sozinho, nao garante compatibilidade.
O suporte Blackwell foi introduzido no PyTorch 2.7 com builds CUDA 12.8,
conforme as [notas oficiais](https://pytorch.org/blog/pytorch-2-7/). Isso nao significa que se deva
substituir um ambiente mais recente que ja funcione.

Ainda nao ha medicao desta bateria na GPU da VM. O tempo tambem depende
de CPU, armazenamento, downloads e ocupacao da placa. Cada job registra
segundos reais, incluindo inicializacao do Python e eventual extracao.
`tempo_estimado.json` extrapola medianas por extrator/modo/shot/metodo,
informando quantas condicoes pendentes possuem referencia medida. A
extrapolacao inicial transfere tempos de Flowers para Cars e inclui custos
iniciais de cache/download; e aproximada, nao uma garantia.

Validacao local do executor: 74 testes automatizados passaram. Uma bateria
reduzida real em Flowers Micro/ResNet ImageNet/CPU, seed 7, cobriu ambos os
shots, modos e os tres metodos: 12/12 execucoes, cerca de 80 segundos somados,
com cache ja disponivel e orcamento quick. Isso nao mede o custo da bateria
completa, do ViT ImageNet ou da GPU. O forward real do ViT foi verificado
offline com pesos aleatorios nos testes de implementacao.

### Saidas

Tudo fica em `runs/comparacao_backbones/`, que aponta para a area de
execucoes do projeto e e ignorada pelo Git:

- `manifest.json`: grade, configuracao, hashes do codigo, planilha/fontes e hardware.
- `auditoria_reaproveitamento.json`: 44 registros GPU auditados, localizacao na
  planilha, linhas de origem e referencias CPU mantidas separadas.
- `resultados.csv`: resultados individuais novos e reaproveitados, com origem,
  dispositivo, fonte e linha. Ganho sobre o sem grafo em pontos percentuais;
  duracao somente para as novas execucoes (vazio para historicos).
- `medias.csv`: medias e desvios entre seeds, separados por condicao;
  contagens mostram grupos ainda incompletos.
- `status.csv`: estado de todas as 240 condicoes, incluindo `reused`.
- `tempo_estimado.json`: tempos medidos e estimativa parcial do restante.
- `_work/`: caches de features e saidas intermediarias compartilhadas.
- Cada condicao nova executada: `command.json`, `run.log`, `status.json`, `metrics.json`
  e `checkpoint.pt` (snapshot fixo no modo sem treino).

As acuracias nos CSVs estao em porcentagem; os JSONs usam valores em [0,1].
O IC95 de cada execucao mede variacao entre episodios; o desvio entre
seeds no CSV de medias e uma medida diferente.

Recriar os CSVs a partir do manifesto, sem executar experimentos:

```bash
bash scripts/buscar_backbones.sh --summarize-only
```

### Enviar o codigo ao Git e executar na VM

No computador local, na raiz do repositorio:

```bash
git status --short
git add AGENTS.md README.md protosgc_fewshot.py \
  proto_sgc/config.py proto_sgc/experiment.py proto_sgc/features.py \
  proto_sgc/training.py proto_sgc/reporting.py proto_sgc/result_inventory.py \
  experimentos_backbones.py scripts/buscar_backbones.sh \
  tests/test_backbones.py tests/test_backbone_sweep.py \
  docs/experimento_backbones.md
git diff --cached --stat
git commit -m "Adiciona bateria ResNet e ViT com e sem treino"
git push origin experimentos_query
```

A lista explicita inclui a implementacao e a organizacao feitas neste chat.
Nao inclui a planilha, backups, resultados de contexto nem o handoff local.

Na VM, usando o ambiente Python/GPU que ja funciona para o projeto:

```bash
cd /mnt/windows/data/gmaia/fewshot/few-shot-DGCG
git status --short
git fetch origin
git switch experimentos_query
git pull --ff-only origin experimentos_query
nvidia-smi
bash scripts/buscar_backbones.sh --device cuda --dry-run
bash scripts/buscar_backbones.sh --device cuda --max-runs 24
bash scripts/buscar_backbones.sh --device cuda
```

Se a VM tiver alteracoes locais que impedem o Git, preserve-as e resolva
antes do pull; nao use reset --hard. O wrapper prefere o Python do Conda
ativo; depois venv local; depois python3. Para escolher explicitamente:
`FSC_PYTHON=/caminho/do/ambiente/bin/python bash scripts/buscar_backbones.sh --device cuda`.
Para sobreviver a uma desconexao SSH, executar dentro de uma sessao tmux,
se disponivel na VM.


### Verificacao do reaproveitamento

78 testes automatizados passaram apos esta atualizacao. A chamada real com
somente ResNet treinada consolidou 36/36 resultados historicos sem GPU,
sem executar treinamento e sem criar checkpoints. As 24 linhas relevantes
da planilha (12 medias Micro e 12 grupos Claude E1) tambem foram conferidas
na versao ja presente no Git HEAD.

O `--dry-run` tambem consulta a planilha e as fontes, antes de mostrar o
plano. Uma fonte adulterada, uma media divergente ou uma planilha ausente
interrompe o planejamento; nao dispara treinos para substituir dados que
nao puderam ser verificados. O modo `--summarize-only` verifica se a
planilha/fontes continuam correspondendo ao manifesto antes de consolidar.
Nao e necessario enviar a alteracao local da planilha de contexto para
este reaproveitamento: a tabela historica ja existe na versao do repositorio.

Se desejar executar somente o experimento do ViT (120 novas condicoes):

```bash
bash scripts/buscar_backbones.sh --device cuda --backbones vit-b-16 \
  --output-root runs/comparacao_backbones_somente_vit
```

A mudanca do codigo de auditoria exige uma pasta nova se a bateria antiga
ja tiver sido iniciada; resultados existentes nao sao apagados.


Verificacao das variantes por grau: os oito resultados reaproveitados sao
lidos individualmente de `resultados_metricas.csv`. As duas medias ALL27
na planilha servem para conferir a fonte inteira, e nao sao usadas como
acuracia individual de Flowers/Cars. As escolhas e seus identificadores
ficam em `DEGREES`/`METHODS` em `experimentos_backbones.py`.

Apos incluir as variantes, passaram 80 testes automatizados e quatro
execucoes reais reduzidas em Flowers Micro/ResNet/CPU/seed7/1-shot:
grau4-6 e grau2-4, cada um com e sem treino adicional. Essas execucoes
verificam funcionamento e nao sao resultados da bateria final em GPU.
