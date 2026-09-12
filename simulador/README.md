# Laboratório interativo do ProtoSGC

Na raiz do projeto, com as dependências de `requirements.txt` instaladas:

```bash
python -m simulador
```

Abra **http://127.0.0.1:8765** no navegador. Mantenha o terminal aberto enquanto usa o simulador; `Ctrl+C` encerra o servidor. Neste projeto também é possível usar diretamente `venv/bin/python -m simulador`.

Se a porta estiver ocupada:

```bash
python -m simulador --port 8766
```

## O que explorar

1. Selecione topologia, GRaNDe, protocolo few-shot, learning rate e demais parâmetros. Clique em **Executar experimento** para recalcular. Os parâmetros são fixos durante uma execução; alterá-los inicia outra execução, sem misturar trajetórias de treino.
2. Use o controle de passos, as setas ou **Reproduzir** para navegar pelos estados registrados.
3. Percorra **Episódio → Grafo → Propagação → Representação → Protótipos → Predição → Atualização**. Na propagação, selecione o salto; no grafo, selecione um nó para ver os pesos de agregação da sua linha de Â.
4. Compare **Antes da atualização** e **Depois da atualização**. Os exemplos são os mesmos nos dois estados.
5. Selecione **Episódio de referência fixo** para observar os mesmos exemplos ao longo de todo o treino. Sua loss não é usada para atualizar os pesos.
6. Abra **Matemática do passo**. As fórmulas têm os valores do episódio que gerou aquela atualização, incluindo gradiente, clipping, momentos do AdamW e um coeficiente de Θ antes/depois.
7. Expanda **Tensores da etapa selecionada** para examinar as matrizes numéricas.
8. Em **Código que faz esta etapa**, veja os trechos reais com arquivo, função e números de linha. O seletor acompanha a topologia e a normalização da execução. Na aba matemática, cada operação também tem **Ver o código desta operação**. Os trechos são lidos dos arquivos `.py` ao carregar a página; não são pseudocódigo.

## Relação com o experimento original

- Cada passo é **um episódio e uma atualização do otimizador**, como em `training.py`. Não corresponde a uma época que percorre todo o dataset.
- `ObservedProtoSGC` herda o modelo do projeto e observa saídas com hooks. Usa o `forward` original, os mesmos construtores de grafos, `FeatureEpisodeSampler`, `split_classes` e `evaluate`.
- O meta-treino usa AdamW, clipping global, loss de entropia cruzada e sementes `seed + 101/202/303` para treino/validação/teste. Valida a cada 5 passos e no último, em 10 episódios; testa o melhor estado em 20 episódios de classes disjuntas. Empates na validação preservam o primeiro melhor estado.
- A referência fixa usa classes de treino e `seed + 404`. Sua avaliação e a repetição do episódio após a atualização são observações adicionais sem gradientes; não alteram o aprendizado.
- Por padrão, a única substituição na origem dos dados é um conjunto sintético de embeddings fixos de 8 dimensões. Não é uma execução da ResNet nem uma medição de desempenho no Flowers Micro. São geradas `4 × N-way` classes com 16 exemplos por classe; treino recebe `2 × N-way`, validação e teste recebem `N-way` cada.
- As figuras mostram somente as duas primeiras coordenadas. Os cálculos usam todas as dimensões; a distância aparente no plano não determina a distância usada pelo modelo.
- Os parâmetros didáticos padrão usam projeção em 4 dimensões e learning rate 0,01 para facilitar a inspeção. O experimento de pesquisa conserva seus próprios valores padrão.
- Theta é mostrada na convenção `X @ Theta`; o peso do `nn.Linear` é armazenado transposto. Os estados históricos não são substituídos pelo melhor checkpoint ao terminar a execução.
- Nada é escrito na pasta `runs/`. Resultados e estados de otimização existem apenas em memória. A interface não baixa recursos externos e o servidor escuta apenas em `127.0.0.1`.

## Usar embeddings reais já extraídos

Passe um cache criado por `features.py`, contendo `features_by_class`:

```bash
python -m simulador \
  --features-cache runs/proto_sgc_meta_album/features_resnet18_imagenet_128px.pt
```

O simulador preserva os vetores do cache. A base deve conter classes suficientes para as três partições e exemplos suficientes para suporte e consultas. A opção de dispersão sintética é desativada nesse modo.

Para limitar o tamanho da resposta ao navegador, X, os saltos e as linhas de Θ/gradientes são enviados com no máximo 8 coordenadas de entrada. A projeção e todas as distâncias continuam usando a dimensão completa do cache. Quando embeddings duplicados tornam a identificação de uma imagem ambígua, a interface mostra o rótulo local do episódio em vez de atribuir uma identidade incerta.

## Arquivos

| Arquivo | Responsabilidade |
| --- | --- |
| `engine.py` | Dados sintéticos ou cache, execução do PyTorch e registro dos tensores. |
| `server.py` | Servidor local e validação das requisições. |
| `source_code.py` | Catálogo de trechos extraídos dos arquivos Python com suas linhas de origem. |
| `static/index.html` | Estrutura da interface e controles. |
| `static/app.js` | Navegação dos estados, desenhos e explicações matemáticas. |
| `static/style.css` | Apresentação e adaptação a telas menores. |

## Verificação

```bash
python -m unittest discover -s tests -v
node tests/check_simulator_ui.cjs
```

Os testes comparam logits e gradientes com o modelo original em todas as topologias, opções de GRaNDe e pesos RBF; conferem o treino, a seleção por validação, o teste final, a reprodução das sementes e a fórmula numérica do AdamW. Não fazem downloads.

O teste opcional em Node exercita o JavaScript com snapshots reais e um DOM mínimo de teste. Verifica controles, navegação, fórmulas e tratamento de erros; não substitui inspeção visual em um navegador. Use `SIMULATOR_PYTHON=/caminho/do/python node tests/check_simulator_ui.cjs` para escolher outro ambiente Python.
