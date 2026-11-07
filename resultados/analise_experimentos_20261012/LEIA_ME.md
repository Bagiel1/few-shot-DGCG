# Análise dos experimentos few-shot DGCG

Abra `relatorio.html` para o relatório completo, com explicações, tabelas e filtros. Ele funciona offline; todos os gráficos estão embutidos.

Foram analisadas 1332 observações válidas após remover 270 duplicações. Há 28 ajustes incompletos em Plant Village.

```text
        metodo   n  media_val  media_teste  media_loss  media_gap
        knn-in 270     77.630       74.192       0.637      3.438
       knn-out 270     78.443       75.230       0.613      3.213
knn-reciprocal 270     79.162       75.623       0.594      3.539
          dgcg 270     79.274       76.047       0.596      3.227
```

Melhor alvo comum por validação: densidade 0.125, validação 67.74%, teste 62.64%. Baseline 4–6 teste 62.35%.
Ganho médio de ajuste por base selecionado pela validação: -0.02 pp. Ganho em transferência leave-one-dataset-out: -0.02 pp.

## Limites das conclusões
Uma rodada por configuração. IC original mede episódios, não seeds. Densidade realizada não foi registrada. Comparações entre n-way mudam a tarefa. Sem predições individuais não é possível avaliar calibração ou confusão de classes. Grau/densidade foram ajustados só em 5-way/1-shot/5-query.

## Gráficos

### Médias da grade e sensibilidade aos episódios pequenos
![Médias da grade e sensibilidade aos episódios pequenos](graficos/01_medias_metodos.png)
As barras são médias descritivas da grade, sem IC por treinamento. Não são resultados de um único protocolo.

### Onde cada método funciona melhor
![Onde cada método funciona melhor](graficos/02_metodos_por_base.png)
Compare os métodos dentro da mesma linha. Bases diferentes têm classes e dificuldades diferentes.

### Ganhos e perdas em relação ao kNN-in
![Ganhos e perdas em relação ao kNN-in](graficos/03_ganho_knnin.png)
Valores positivos favorecem o método da coluna. Cada célula usa a mesma base e os mesmos 27 protocolos.

### Comparação no protocolo default
![Comparação no protocolo default](graficos/04_protocolo_default.png)
Esta comparação elimina a mistura entre protocolos. Cada valor é o teste de um checkpoint escolhido pela validação.

### Resultado depois da escolha de protocolo por validação
![Resultado depois da escolha de protocolo por validação](graficos/05_selecao_validacao.png)
Cada método/base/n-way escolhe o shot/query de maior validação; desempate por menor loss de validação. O teste é consultado depois da escolha.

### Todas as combinações shot/query em 2-way
![Todas as combinações shot/query em 2-way](graficos/06_heatmap_way2.png)
Cada célula é a média de dez bases. Maior suporte/query usa mais informação; não representa ganho gratuito com o mesmo orçamento.

### Todas as combinações shot/query em 3-way
![Todas as combinações shot/query em 3-way](graficos/06_heatmap_way3.png)
Cada célula é a média de dez bases. Maior suporte/query usa mais informação; não representa ganho gratuito com o mesmo orçamento.

### Todas as combinações shot/query em 5-way
![Todas as combinações shot/query em 5-way](graficos/06_heatmap_way5.png)
Cada célula é a média de dez bases. Maior suporte/query usa mais informação; não representa ganho gratuito com o mesmo orçamento.

### Efeito de n_shot
![Efeito de n_shot](graficos/07_efeito_n_shot.png)
Médias balanceadas sobre os outros parâmetros e datasets; o treinamento também muda com o protocolo. Query tem efeito transdutivo porque participa do grafo.

### Efeito de n_query
![Efeito de n_query](graficos/07_efeito_n_query.png)
Médias balanceadas sobre os outros parâmetros e datasets; o treinamento também muda com o protocolo. Query tem efeito transdutivo porque participa do grafo.

### Diferenças entre classes de validação e de teste
![Diferenças entre classes de validação e de teste](graficos/08_gap_val_teste.png)
Gap positivo significa validação mais fácil ou mais otimista que o teste. As classes são disjuntas: este gap, sozinho, não prova overfitting.

### Loss, acurácia e diferenças entre splits
![Loss, acurácia e diferenças entre splits](graficos/09_loss_validacao.png)
Loss depende das probabilidades/confiança, além dos acertos. Estes CSVs não contêm probabilidades suficientes para avaliar calibração diretamente.

### Resposta média do DGCG ao grau e à densidade
![Resposta média do DGCG ao grau e à densidade](graficos/10_curvas_grau_densidade.png)
Compare validação para escolher o alvo. Alvos grandes podem aumentar mistura entre classes; alvos pequenos podem reduzir troca de informação.

### Sensibilidade de grau em cada dataset
![Sensibilidade de grau em cada dataset](graficos/11_dgcg_grau_bases.png)
Plant Village está ausente porque todas as suas 28 execuções de ajuste estão incompletas.

### Sensibilidade de densidade em cada dataset
![Sensibilidade de densidade em cada dataset](graficos/11_dgcg_densidade_bases.png)
Plant Village está ausente porque todas as suas 28 execuções de ajuste estão incompletas.

### Quanto o ajuste realmente acrescentou no teste
![Quanto o ajuste realmente acrescentou no teste](graficos/12_ganhos_ajuste.png)
Todos os ganhos usam o baseline da mesma rodada e o mesmo protocolo. Nenhum alvo foi escolhido pelo teste.

### DGCG ajustado e as baselines no mesmo protocolo
![DGCG ajustado e as baselines no mesmo protocolo](graficos/13_ajuste_vs_baselines.png)
O DGCG ajustado vem de outra rodada: pequenas diferenças de execução existem. Esta comparação é descritiva; ganho de ajuste deve ser medido contra 4–6 da própria rodada.

### Dossiê Bacteria: todos os protocolos
![Dossiê Bacteria: todos os protocolos](graficos/base_bacteria_micro.png)
s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).

### Dossiê Birds: todos os protocolos
![Dossiê Birds: todos os protocolos](graficos/base_birds_micro.png)
s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).

### Dossiê Cars: todos os protocolos
![Dossiê Cars: todos os protocolos](graficos/base_cars_micro.png)
s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).

### Dossiê Flowers: todos os protocolos
![Dossiê Flowers: todos os protocolos](graficos/base_flowers_micro.png)
s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).

### Dossiê Manufacturing: todos os protocolos
![Dossiê Manufacturing: todos os protocolos](graficos/base_manufacturing_micro.png)
s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).

### Dossiê Plankton: todos os protocolos
![Dossiê Plankton: todos os protocolos](graficos/base_plankton_micro.png)
s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).

### Dossiê Plant Village: todos os protocolos
![Dossiê Plant Village: todos os protocolos](graficos/base_plant_village_micro.png)
s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).

### Dossiê RESISC: todos os protocolos
![Dossiê RESISC: todos os protocolos](graficos/base_resisc_micro.png)
s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).

### Dossiê Sports: todos os protocolos
![Dossiê Sports: todos os protocolos](graficos/base_sports_micro.png)
s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).

### Dossiê Textures: todos os protocolos
![Dossiê Textures: todos os protocolos](graficos/base_textures_micro.png)
s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).

### Comparar desempenho com o orçamento de imagens
![Comparar desempenho com o orçamento de imagens](graficos/14_orcamento.png)
Protocolos com mais imagens tendem a resolver tarefas mais informadas. Número de nós não substitui tempo medido: custo da ordenação/propagação depende do método e da implementação.
