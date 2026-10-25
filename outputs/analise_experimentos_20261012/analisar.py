from __future__ import annotations

import base64
import hashlib
import html
import itertools
import json
import math
import shutil
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/bagiel/Downloads')
FIG = ROOT / 'graficos'
TABLE = ROOT / 'tabelas'
FIG.mkdir(exist_ok=True)
TABLE.mkdir(exist_ok=True)
FILES = ['melhor_baseline_medio.csv','melhor_baseline_por_dataset.csv','melhores_medios.csv',
 'melhores_medios_por_metodo.csv','melhores_por_dataset.csv','melhores_por_dataset_e_metodo.csv',
 'melhores_por_nway.csv','melhores_por_nway_protocolos.csv','ranking_medio.csv',
 'ranking_medio_protocolos.csv','resultados.csv','resultados_protocolos.csv',
 'melhores_medios_dgcg_protocolo.csv','melhores_por_dataset_dgcg_protocolo.csv',
 'melhores_por_dataset_e_modo_dgcg_protocolo.csv','ranking_medio_dgcg_protocolo.csv',
 'resultados_dgcg_protocolo.csv']
D = {name:pd.read_csv(SOURCE/name) for name in FILES}
KEY = ['metodo','dataset','n_way','n_shot','n_query']
METRICS = ['val_accuracy','val_ci95','val_loss','test_accuracy','test_ci95','test_loss']
METHODS = ['knn-in','knn-out','knn-reciprocal','dgcg']
LABEL = {'knn-in':'kNN-in','knn-out':'kNN-out','knn-reciprocal':'kNN recíproco','dgcg':'DGCG'}
COLORS = dict(zip(METHODS,['#2878b5','#dc8325','#32a180','#8056b3']))
NAMES = {'bacteria_micro':'Bacteria','plankton_micro':'Plankton','flowers_micro':'Flowers',
 'sports_micro':'Sports','birds_micro':'Birds','plant_village_micro':'Plant Village',
 'textures_micro':'Textures','manufacturing_micro':'Manufacturing','cars_micro':'Cars','resisc_micro':'RESISC'}
A,B,C = D['resultados.csv'],D['resultados_protocolos.csv'],D['resultados_dgcg_protocolo.csv']
ALL = pd.concat([A,B[B.metodo=='dgcg']],ignore_index=True)
ALL['nos'] = ALL.n_way*(ALL.n_shot+ALL.n_query)
ALL['acaso'] = 100/ALL.n_way
ALL['acima_acaso'] = (ALL.test_accuracy-ALL.acaso)/(100-ALL.acaso)
ALL['gap_val_teste'] = ALL.val_accuracy-ALL.test_accuracy
ALL['grafo_completo_knn'] = (ALL.metodo!='dgcg') & (ALL.nos<=6)
C['alvo'] = C.apply(lambda r: f'{r.alvo_min:g}' if r.alvo_min==r.alvo_max else f'{r.alvo_min:g}–{r.alvo_max:g}',axis=1)
CV = C[C.status=='ok'].copy()
datasets = sorted(ALL.dataset.unique())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':180,'axes.titlepad':14})
SECTIONS=[]
FIGURES=[]
TABLES=[]

def csv(name,df):
    df.to_csv(TABLE/f'{name}.csv',index=False)
    TABLES.append(name)
    return df

def tab(df,n=40):
    return df.head(n).to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}',escape=True,classes='data')

def section(title,text,df=None):
    body=''.join(f'<p>{p}</p>' for p in text)
    if df is not None: body+='<div class="scroll">'+tab(df)+'</div>'
    SECTIONS.append((title,body))

def fig(name,title,caption):
    plt.gcf().tight_layout(pad=1.8,w_pad=3.0)
    plt.savefig(FIG/f'{name}.png',bbox_inches='tight',facecolor='white')
    plt.savefig(FIG/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close()
    FIGURES.append((name,title,caption))
    image64=base64.b64encode((FIG/f'{name}.png').read_bytes()).decode()
    SECTIONS.append((title,f'<figure><img src="data:image/png;base64,{image64}" alt="{html.escape(title)}"><figcaption>{caption}</figcaption></figure>'))

def heat(ax,frame,title,cmap='viridis',vmin=None,vmax=None,fmt='.1f'):
    im=ax.imshow(frame.to_numpy(dtype=float),aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(frame.columns)),[str(x) for x in frame.columns])
    ax.set_yticks(range(len(frame.index)),[NAMES.get(x,LABEL.get(x,str(x))) for x in frame.index])
    ax.set_title(title)
    for i in range(len(frame.index)):
        for j in range(len(frame.columns)):
            val=frame.iloc[i,j]
            if pd.notna(val):
                ax.text(j,i,format(val,fmt),ha='center',va='center',fontsize=8,
                 color='white' if (val-(vmin or np.nanmin(frame.to_numpy()))) / max(1e-9,(vmax or np.nanmax(frame.to_numpy()))-(vmin or np.nanmin(frame.to_numpy())))<.65 else '#15222f')
    return im

def select(df,groups,score='val_accuracy'):
    return df.sort_values([score,'val_loss'],ascending=[False,True],kind='stable').groupby(groups,sort=True).head(1).copy()

def paired_summary(matrix):
    rows=[]
    rng=np.random.default_rng(20261012)
    for m1,m2 in itertools.combinations(matrix.columns,2):
        diff=(matrix[m1]-matrix[m2]).dropna().to_numpy()
        boot=diff[rng.integers(0,len(diff),size=(20000,len(diff)))].mean(axis=1)
        p=float(stats.wilcoxon(diff,zero_method='wilcox').pvalue) if np.any(diff) else 1.
        rows.append(dict(metodo_1=m1,metodo_2=m2,n_bases=len(diff),ganho_medio_pp=diff.mean(),
         bootstrap_p025=np.quantile(boot,.025),bootstrap_p975=np.quantile(boot,.975),
         vitorias_1=int((diff>1e-8).sum()),empates=int((abs(diff)<1e-8).sum()),derrotas_1=int((diff< -1e-8).sum()),p_wilcoxon=p))
    out=pd.DataFrame(rows)
    order=np.argsort(out.p_wilcoxon.to_numpy())
    adj=np.maximum.accumulate(out.p_wilcoxon.to_numpy()[order]*(len(out)-np.arange(len(out))))
    out['p_holm']=0.
    out.loc[order,'p_holm']=np.minimum(adj,1.)
    return out

# Every supplied file is read and recorded; derived files are checked against sources.
audit=[]
for name,df in D.items():
    numeric=df.select_dtypes(include=np.number)
    audit.append(dict(arquivo=name,linhas=len(df),colunas=len(df.columns),
      status_ok=int((df.status=='ok').sum()) if 'status' in df else None,
      status_incompleto=int((df.status!='ok').sum()) if 'status' in df else None,
      celulas_numericas_ausentes=int(numeric.isna().sum().sum()),
      sha256=hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()))
csv('inventario_17_arquivos',pd.DataFrame(audit))
assert len(ALL)==1080 and not ALL.duplicated(KEY).any()
rep=A.merge(B,on=KEY,suffixes=('_a','_b'))
assert len(rep)==270
assert all(np.allclose(rep[f'{m}_a'],rep[f'{m}_b']) for m in METRICS)
csv('repeticoes_knn_in',rep)
assert ALL.status.eq('ok').all()
assert ALL[METRICS].notna().all().all()
assert ALL[['val_accuracy','test_accuracy']].ge(0).all().all() and ALL[['val_accuracy','test_accuracy']].le(100).all().all()
assert ALL[['val_ci95','test_ci95','val_loss','test_loss']].ge(0).all().all()
assert np.allclose(ALL.val_acima_acaso,(ALL.val_accuracy-ALL.acaso)/(100-ALL.acaso))
assert CV[METRICS].notna().all().all()
csv('execucoes_incompletas_dgcg',C[C.status!='ok'])
csv('resultados_unificados_1080',ALL)
csv('varredura_dgcg_252_validos',CV)

reconc=[]
for name,df in D.items():
    if name.startswith('resultados'):continue
    if 'modo' in df:
        raw=CV
        keys=['modo','dataset','alvo_min','alvo_max'] if 'dataset' in df else ['modo','alvo_min','alvo_max']
    else:
        raw=B if ('protocolos' in name or name in ['melhores_medios.csv','melhores_por_dataset.csv']) else A
        keys=KEY if 'dataset' in df else ['metodo','n_way','n_shot','n_query']
    if 'dataset' in df:
        merge=df.merge(raw,on=keys,suffixes=('_s','_r'),how='left',indicator=True)
        cols=[m for m in METRICS if m in df]
        delta=max([float((merge[m+'_s']-merge[m+'_r']).abs().max()) for m in cols],default=0)
        unmatched=int((merge._merge!='both').sum())
    else:
        agg=raw.groupby(keys).agg(datasets_concluidos=('dataset','size'),media_val_accuracy=('val_accuracy','mean'),media_test_accuracy=('test_accuracy','mean')).reset_index()
        if 'val_acima_acaso' in raw: agg=agg.merge(raw.groupby(keys).val_acima_acaso.mean().rename('media_val_acima_acaso').reset_index(),on=keys)
        merge=df.merge(agg,on=keys,suffixes=('_s','_r'),how='left',indicator=True)
        cols=[x for x in ['datasets_concluidos','media_val_accuracy','media_test_accuracy','media_val_acima_acaso'] if x in df]
        delta=max([float((merge[m+'_s']-merge[m+'_r']).abs().max()) for m in cols],default=0)
        unmatched=int((merge._merge!='both').sum())
    reconc.append(dict(arquivo=name,linhas=len(df),nao_localizadas=unmatched,maior_diferenca=delta,coerente=unmatched==0 and delta<1e-8))
csv('auditoria_resumos',pd.DataFrame(reconc))

section('Como ler os resultados',[
 '<strong>Acurácia</strong> é a porcentagem de consultas acertadas; maior é melhor. <strong>Loss</strong> é a entropia cruzada; menor é melhor. <strong>pp</strong> significa pontos percentuais: 70% contra 65% é +5 pp.',
 'Os melhores hiperparâmetros são escolhidos pela validação. O desempenho reportado depois é o teste da configuração escolhida. Uma curva com todos os testes é descritiva; não use seu pico para escolher parâmetros.',
 'Comparar 2-way com 5-way mede tarefas diferentes. O acaso é 50%, 33,33% e 20% para 2, 3 e 5 classes. O escore (acurácia − acaso)/(100 − acaso) ajuda a contextualizar, mas não torna a dificuldade das tarefas equivalente.',
 'Os arquivos não contêm predições individuais, acurácias por episódio, seeds, tempos, curvas de treinamento ou densidades realizadas. Por isso, não permitem matrizes de confusão, teste pareado por episódio, medir tempo/GPU, provar calibração ou confirmar a densidade efetiva de cada grafo.' ])
section('Auditoria de todos os arquivos',[
 f'Foram lidos os 17 CSVs. Há 810 execuções de baselines, 540 de protocolos e 280 de grau/densidade: 1.630 linhas brutas. As 270 linhas kNN-in repetidas nos dois primeiros experimentos têm todas as seis métricas idênticas. Após remover essa repetição, ficam 1.080 combinações método/base/protocolo únicas e 252 ajustes DGCG válidos: <strong>1.332 observações utilizáveis</strong>. As cópias não foram contadas como replicações independentes.',
 'A busca de grau/densidade tem <strong>28 execuções incompletas em Plant Village</strong>, uma por alvo. Seus números ausentes foram preservados; essa base está excluída apenas da análise de ajuste DGCG. As comparações dessa busca têm nove bases.',
 'Os resumos fornecidos foram reconciliados com os arquivos de resultados correspondentes. Essa checagem valida valores e médias publicados, e não constitui uma replicação dos treinos.' ],pd.DataFrame(reconc))

# Overall methods with / without structurally saturated small episodes.
overview=ALL.groupby('metodo').agg(n=('dataset','size'),media_val=('val_accuracy','mean'),media_teste=('test_accuracy','mean'),media_loss=('test_loss','mean'),media_gap=('gap_val_teste','mean')).reindex(METHODS).reset_index()
csv('medias_metodos_grade_completa',overview)
clean=ALL[ALL.nos>6]
clean_over=clean.groupby('metodo').agg(media_teste=('test_accuracy','mean'),media_loss=('test_loss','mean')).reindex(METHODS).reset_index()
csv('medias_metodos_sem_episodios_ate6nos',clean_over)
section('Comparação geral dos quatro métodos',[
 'A grade está balanceada: cada método tem 270 resultados, os mesmos dez datasets e 27 protocolos. A média abaixo dá peso igual às bases e protocolos; ela resume esta grade específica e não determina o melhor método para toda aplicação.',
 'A comparação adicional remove os dois protocolos com até seis nós para mostrar quanto o grafo completo de kNN interfere no ranking. Essa exclusão é uma análise de sensibilidade, não apaga os resultados ruins.' ],overview)
fig_,axs=plt.subplots(1,2,figsize=(12,4.5))
for ax,df,title in [(axs[0],overview,'Todos os 27 protocolos'),(axs[1],clean_over,'25 protocolos com mais de 6 nós')]:
 ax.bar([LABEL[m] for m in df.metodo],df.media_teste,color=[COLORS[m] for m in df.metodo])
 for i,v in enumerate(df.media_teste):ax.text(i,v+.5,f'{v:.2f}%',ha='center')
 ax.set_ylim(0,100);ax.set_ylabel('Acurácia média no teste (%)');ax.set_title(title);ax.tick_params(axis='x',labelrotation=20)
fig('01_medias_metodos','Médias da grade e sensibilidade aos episódios pequenos','As barras são médias descritivas da grade, sem IC por treinamento. Não são resultados de um único protocolo.')

means_ds=ALL.groupby(['dataset','metodo']).test_accuracy.mean().unstack().reindex(columns=METHODS)
csv('metodos_por_dataset_media_grade',means_ds.reset_index())
f,ax=plt.subplots(figsize=(9,6));im=heat(ax,means_ds,'Média dos 27 protocolos por base e método',vmin=0,vmax=100);f.colorbar(im,ax=ax,label='Acurácia no teste (%)')
fig('02_metodos_por_base','Onde cada método funciona melhor','Compare os métodos dentro da mesma linha. Bases diferentes têm classes e dificuldades diferentes.')
delta=means_ds.subtract(means_ds['knn-in'],axis=0).drop(columns='knn-in')
f,ax=plt.subplots(figsize=(8,6));im=heat(ax,delta,'Diferença para kNN-in',cmap='RdBu',vmin=-8,vmax=8);f.colorbar(im,ax=ax,label='Ganho no teste (pp)')
fig('03_ganho_knnin','Ganhos e perdas em relação ao kNN-in','Valores positivos favorecem o método da coluna. Cada célula usa a mesma base e os mesmos 27 protocolos.')

# Pairwise rank and tests, dataset is statistical unit (not 270 independent runs).
tests=csv('comparacoes_pareadas_por_base',paired_summary(means_ds))
tests_clean=csv('comparacoes_pareadas_sem_episodios_pequenos',paired_summary(clean.groupby(['dataset','metodo']).test_accuracy.mean().unstack().reindex(columns=METHODS)))
friedman=stats.friedmanchisquare(*[means_ds[m] for m in METHODS])
ranks=means_ds.rank(axis=1,ascending=False).mean().sort_values().rename('rank_medio').reset_index()
csv('ranking_por_base',ranks)
section('Comparações estatísticas exploratórias',[
 f'Usei cada dataset como bloco: dez diferenças pareadas entre métodos após tirar a média dos mesmos protocolos. Isso evita tratar 270 configurações correlacionadas como 270 amostras independentes. Friedman: estatística {friedman.statistic:.3f}, p={friedman.pvalue:.4g}.',
 'A tabela apresenta Wilcoxon pareado, correção de Holm para seis comparações e IC bootstrap de diferenças médias reamostrando datasets (20.000 reamostragens). São evidências exploratórias sobre estes dez datasets; as bases foram escolhidas previamente e não são uma amostra aleatória de todo domínio de imagens.',
 'Os ICs do CSV original representam variação entre episódios para um modelo treinado. Não capturam variação entre seeds/treinamentos. Sobreposição de IC não decide significância e acurácias médias dos CSVs não permitem reconstruir testes pareados por episódio.' ],tests)

# Fixed protocol, validation selected per way; LODO common configuration.
fixed=ALL[(ALL.n_way==5)&(ALL.n_shot==1)&(ALL.n_query==5)].copy()
csv('comparacao_fixa_5way_1shot_5query',fixed)
fixed_p=fixed.pivot(index='dataset',columns='metodo',values='test_accuracy').reindex(columns=METHODS)
f,ax=plt.subplots(figsize=(9,6));im=heat(ax,fixed_p,'Protocolo fixo 5-way / 1-shot / 5-query',vmin=0,vmax=100);f.colorbar(im,ax=ax,label='Teste (%)')
fig('04_protocolo_default','Comparação no protocolo default','Esta comparação elimina a mistura entre protocolos. Cada valor é o teste de um checkpoint escolhido pela validação.')
csv('estatistica_protocolo_fixo',paired_summary(fixed_p))
selected=select(ALL,['metodo','dataset','n_way'])
csv('protocolos_selecionados_por_validacao_nway',selected)
selmeans=selected.groupby(['n_way','metodo']).test_accuracy.mean().unstack().reindex(columns=METHODS)
f,ax=plt.subplots(figsize=(8,4.5));selmeans.rename(columns=LABEL).plot.bar(ax=ax,color=[COLORS[m] for m in METHODS]);ax.set_ylim(0,100);ax.set_ylabel('Teste médio (%)');ax.set_xlabel('n-way');ax.set_title('Shot/query escolhidos na validação para cada base');ax.legend(loc='lower left',ncol=2)
fig('05_selecao_validacao','Resultado depois da escolha de protocolo por validação','Cada método/base/n-way escolhe o shot/query de maior validação; desempate por menor loss de validação. O teste é consultado depois da escolha.')

lodo=[]
for m in METHODS:
 for ds in datasets:
  train=ALL[(ALL.metodo==m)&(ALL.dataset!=ds)&(ALL.n_way==5)]
  agg=train.groupby(['n_shot','n_query']).val_accuracy.mean().sort_values(ascending=False,kind='stable')
  shot,query=agg.index[0]
  row=ALL[(ALL.metodo==m)&(ALL.dataset==ds)&(ALL.n_way==5)&(ALL.n_shot==shot)&(ALL.n_query==query)].iloc[0]
  lodo.append(dict(metodo=m,dataset=ds,n_shot=shot,n_query=query,media_val_outras_bases=agg.iloc[0],val_accuracy=row.val_accuracy,test_accuracy=row.test_accuracy,test_ci95=row.test_ci95))
lodo=csv('transferencia_protocolo_leave_one_dataset_out',pd.DataFrame(lodo))
section('Escolher um protocolo que transfere entre bases',[
 'A tabela “leave-one-dataset-out” escolhe shot/query usando a validação das outras nove bases e mede o teste da décima base. n-way fica em 5 para manter a tarefa comparável. O processo é repetido para cada base/método.',
 'Isso mede transferência da escolha do protocolo entre datasets, não treinamento cruzado: o modelo de cada linha continua tendo sido treinado nas classes de treino da própria base. As configurações selecionadas por validação já existem nos CSVs.' ],lodo.groupby('metodo').agg(media_teste=('test_accuracy','mean'),menor_teste=('test_accuracy','min')).reset_index())

# Parameter effects and all heatmap panels.
for way in [2,3,5]:
 f,axs=plt.subplots(1,4,figsize=(15,4))
 for ax,m in zip(axs,METHODS):
  frame=ALL[(ALL.metodo==m)&(ALL.n_way==way)].pivot_table(index='n_shot',columns='n_query',values='test_accuracy',aggfunc='mean')
  heat(ax,frame,LABEL[m],vmin=0,vmax=100);ax.set_xlabel('n-query');ax.set_ylabel('n-shot')
 fig(f'06_heatmap_way{way}',f'Todas as combinações shot/query em {way}-way','Cada célula é a média de dez bases. Maior suporte/query usa mais informação; não representa ganho gratuito com o mesmo orçamento.')

effects=[]
for param in ['n_shot','n_query']:
 f,axs=plt.subplots(1,3,figsize=(14,4))
 for ax,way in zip(axs,[2,3,5]):
  for m in METHODS:
   frame=ALL[(ALL.metodo==m)&(ALL.n_way==way)].groupby(param).test_accuracy.mean()
   ax.plot(frame.index,frame.values,'o-',label=LABEL[m],color=COLORS[m])
  ax.set_title(f'{way}-way');ax.set_xlabel(param);ax.set_ylabel('Teste médio (%)');ax.set_xticks([1,5,10]);ax.set_ylim(0,100)
 axs[-1].legend(fontsize=8)
 fig(f'07_efeito_{param}',f'Efeito de {param}','Médias balanceadas sobre os outros parâmetros e datasets; o treinamento também muda com o protocolo. Query tem efeito transdutivo porque participa do grafo.')
 for m in METHODS:
  for low,high in [(1,5),(5,10)]:
   keys=['dataset','n_way', 'n_query' if param=='n_shot' else 'n_shot']
   left=ALL[(ALL.metodo==m)&(ALL[param]==low)]
   right=ALL[(ALL.metodo==m)&(ALL[param]==high)]
   j=left.merge(right,on=keys,suffixes=('_baixo','_alto'))
   effects.append(dict(metodo=m,parametro=param,passo=f'{low}→{high}',n_comparacoes=len(j),ganho_medio_pp=(j.test_accuracy_alto-j.test_accuracy_baixo).mean(),fracao_melhora=(j.test_accuracy_alto>j.test_accuracy_baixo).mean()))
csv('efeitos_pareados_shot_query',pd.DataFrame(effects))
section('O que muda com suporte e consultas',[
 'n-shot aumenta os exemplos rotulados por classe. n-query aumenta as imagens não rotuladas disponíveis conjuntamente no grafo transdutivo e o número de consultas usadas na métrica. Mudar esses valores altera a tarefa, o orçamento e a composição do grafo.',
 'Os efeitos abaixo comparam pares que mantêm base, método, n-way e o outro parâmetro iguais. Há 90 pares por passo/método. O percentual de melhora é descritivo; esses 90 pares compartilham datasets e não são replicações independentes.' ],pd.DataFrame(effects))

# Saturation audit.
collapsed=ALL[ALL.grafo_completo_knn].copy()
csv('colapso_knn_grafos_completos',collapsed)
collapse_summary=collapsed.groupby(['metodo','n_way','n_shot','n_query','nos']).agg(n_bases=('dataset','size'),media_teste=('test_accuracy','mean'),min_teste=('test_accuracy','min'),max_teste=('test_accuracy','max'),media_loss=('test_loss','mean')).reset_index()
section('Achado estrutural: o kNN colapsa em episódios muito pequenos',[
 f'<strong>{len(collapsed)} execuções kNN</strong> usam episódios com até seis nós. Em 2-way/1-shot/1-query há quatro nós; em 3-way/1-shot/1-query há seis. Como k=5 e o código limita k a N−1, todos os nós escolhem todos os outros.',
 'kNN-in, kNN-out e recíproco viram o mesmo grafo completo. Com autolaços, A normalizada vira J/N. Multiplicar por ela transforma todos os vetores na mesma média. A projeção posterior não consegue separar classes porque recebe representações iguais. A previsão perde informação discriminativa.',
 'Isso explica resultados próximos do acaso e loss próxima de ln(n-way), incluindo 50% com loss 0,6931. Pequenas diferenças numéricas em 3-way podem ocorrer por arredondamento/flutuante. O problema é a interação entre k fixo e episódios pequenos. Para uma próxima rodada, vale testar k menor nesses protocolos e incluir ProtoNet sem propagação.' ],collapse_summary)

# Validation/test mismatch, loss explanation.
gap=csv('gap_por_dataset_metodo',ALL.groupby(['dataset','metodo']).agg(gap_medio_pp=('gap_val_teste','mean'),media_val=('val_accuracy','mean'),media_teste=('test_accuracy','mean'),media_loss=('test_loss','mean')).reset_index())
f,ax=plt.subplots(figsize=(9,6));pivot=gap.pivot(index='dataset',columns='metodo',values='gap_medio_pp').reindex(columns=METHODS);im=heat(ax,pivot,'Validação menos teste',cmap='RdBu_r',vmin=-25,vmax=25);f.colorbar(im,ax=ax,label='Diferença (pp)')
fig('08_gap_val_teste','Diferenças entre classes de validação e de teste','Gap positivo significa validação mais fácil ou mais otimista que o teste. As classes são disjuntas: este gap, sozinho, não prova overfitting.')
f,axs=plt.subplots(1,2,figsize=(12,4.5))
for m in METHODS:
 df=ALL[ALL.metodo==m]
 axs[0].scatter(df.test_accuracy,df.test_loss,s=12,alpha=.4,label=LABEL[m],color=COLORS[m])
 axs[1].scatter(df.val_accuracy,df.test_accuracy,s=12,alpha=.4,color=COLORS[m])
axs[0].set_xlabel('Acurácia de teste (%)');axs[0].set_ylabel('Loss de teste');axs[0].legend(fontsize=8)
axs[1].plot([0,100],[0,100],'k--',lw=1);axs[1].set_xlabel('Validação (%)');axs[1].set_ylabel('Teste (%)')
fig('09_loss_validacao','Loss, acurácia e diferenças entre splits','Loss depende das probabilidades/confiança, além dos acertos. Estes CSVs não contêm probabilidades suficientes para avaliar calibração diretamente.')

# Degree sweep equivalence and baseline checks.
base=CV[(CV.modo=='grau')&(CV.alvo_min==4)&(CV.alvo_max==6)].copy()
cross=base.merge(fixed[fixed.metodo=='dgcg'],on='dataset',suffixes=('_sweep','_protocol'))
cross['diferenca_val_pp']=cross.val_accuracy_sweep-cross.val_accuracy_protocol
cross['diferenca_teste_pp']=cross.test_accuracy_sweep-cross.test_accuracy_protocol
csv('baseline_dgcg_entre_rodadas',cross[['dataset','val_accuracy_sweep','val_accuracy_protocol','test_accuracy_sweep','test_accuracy_protocol','diferenca_val_pp','diferenca_teste_pp']])
section('Grau e densidade: o que foi realmente medido',[
 'A varredura usa apenas 5-way/1-shot/5-query: 30 nós e 29 candidatos. Com esse universo completo de candidatos, grau-alvo = densidade-alvo × 29. Em episódios maiores com candidate-k/list-size limitados, a densidade configurada refere-se ao conjunto de candidatos, e não necessariamente a todas as arestas possíveis.',
 'Uma grade de graus e outra de densidades são duas formas de pedir a mesma seleção de arestas. Um alvo simples [4,4] busca grau 4; o baseline [4,6] busca o intervalo com preferência pelo centro 5. A densidade 0,125 corresponde a grau 3,625.',
 'Os CSVs contêm <strong>alvos e equivalências teóricas</strong>, não graus/densidades observados. Empates nas correlações entram juntos e o fallback evita nós sem vizinhos. O grau final pode sair do alvo. Não existe evidência suficiente aqui para chamar um valor de “densidade perfeita”.',
 f'O baseline 4–6 desta rodada difere da rodada de protocolos em até {cross.diferenca_teste_pp.abs().max():.2f} pp no teste. Os caminhos Windows/Linux sugerem ambientes distintos; causa exata e seeds não foram registradas. Por isso, os ganhos de ajuste abaixo usam o baseline <strong>da própria varredura</strong>.' ],cross[['dataset','test_accuracy_sweep','test_accuracy_protocol','diferenca_teste_pp']])

drank=CV.groupby(['modo','alvo_min','alvo_max','alvo']).agg(n_bases=('dataset','size'),media_val=('val_accuracy','mean'),media_teste=('test_accuracy','mean'),media_loss=('test_loss','mean')).reset_index()
csv('ranking_grau_densidade_recalculado',drank.sort_values('media_val',ascending=False))
f,axs=plt.subplots(1,2,figsize=(13,4.5))
for ax,mode in zip(axs,['grau','densidade']):
 df=drank[(drank.modo==mode)&(drank.alvo_min==drank.alvo_max)].sort_values('alvo_min')
 ax.plot(df.alvo_min,df.media_val,'o-',label='Validação (seleção)',color='#8056b3')
 ax.plot(df.alvo_min,df.media_teste,'s--',label='Teste (descritivo)',color='#2878b5')
 ax.axhline(base.val_accuracy.mean(),color='#8056b3',alpha=.4,lw=1,label='Baseline 4–6: validação')
 ax.set_xlabel('Grau-alvo' if mode=='grau' else 'Densidade-alvo');ax.set_ylabel('Acurácia média (%)');ax.set_title(f'Alvos de {mode}: nove bases');ax.legend(fontsize=8)
fig('10_curvas_grau_densidade','Resposta média do DGCG ao grau e à densidade','Compare validação para escolher o alvo. Alvos grandes podem aumentar mistura entre classes; alvos pequenos podem reduzir troca de informação.')

for mode in ['grau','densidade']:
 frame=CV[CV.modo==mode].pivot(index='dataset',columns='alvo',values='val_accuracy')
 order=CV[CV.modo==mode].drop_duplicates('alvo').sort_values(['alvo_min','alvo_max']).alvo.tolist()
 frame=frame.reindex(columns=order)
 f,ax=plt.subplots(figsize=(14,6));im=heat(ax,frame,f'Validação por base: {mode}',vmin=0,vmax=100);ax.set_xlabel('Alvo');f.colorbar(im,ax=ax,label='Validação (%)')
 fig(f'11_dgcg_{mode}_bases',f'Sensibilidade de {mode} em cada dataset','Plant Village está ausente porque todas as suas 28 execuções de ajuste estão incompletas.')

tuned=select(CV,['dataset'])
gains=tuned.merge(base[['dataset','val_accuracy','test_accuracy','test_ci95']],on='dataset',suffixes=('_ajuste','_baseline'))
gains['ganho_val_pp']=gains.val_accuracy_ajuste-gains.val_accuracy_baseline
gains['ganho_teste_pp']=gains.test_accuracy_ajuste-gains.test_accuracy_baseline
csv('ganhos_ajuste_local_validacao',gains)
gain_display=gains[['dataset','modo','alvo','val_accuracy_ajuste','test_accuracy_ajuste','test_accuracy_baseline','ganho_teste_pp']]
section('Ajuste escolhido separadamente para cada base',[
 f'O melhor alvo foi escolhido na validação entre as 28 configurações de cada base. Ganho médio no teste sobre o baseline da mesma rodada: <strong>{gains.ganho_teste_pp.mean():.2f} pp</strong>. Melhorou em {int((gains.ganho_teste_pp>0).sum())} das nove bases, empatou em {int((gains.ganho_teste_pp==0).sum())} e piorou em {int((gains.ganho_teste_pp<0).sum())}.',
 'O ganho de validação é otimista por construção: o máximo de 28 tentativas tende a subir. A comparação de teste após essa seleção é a medida mais útil para avaliar se o ajuste ajudou; ainda falta repetição com seeds para medir estabilidade.' ],gain_display)
f,ax=plt.subplots(figsize=(9,5));plot=gains.sort_values('ganho_teste_pp');ax.barh(plot.dataset.map(NAMES),plot.ganho_teste_pp,color=np.where(plot.ganho_teste_pp>=0,'#32a180','#d45858'));ax.axvline(0,color='#34495e');ax.set_xlabel('Ganho no teste (pp)');ax.set_title('Alvo escolhido na validação versus baseline DGCG 4–6')
fig('12_ganhos_ajuste','Quanto o ajuste realmente acrescentou no teste','Todos os ganhos usam o baseline da mesma rodada e o mesmo protocolo. Nenhum alvo foi escolhido pelo teste.')

# LODO target; use only validation of other bases; no synthetic deployment claim.
target_lodo=[]
for ds in sorted(CV.dataset.unique()):
 train=CV[CV.dataset!=ds]
 agg=train.groupby(['modo','alvo_min','alvo_max']).val_accuracy.mean().sort_values(ascending=False,kind='stable')
 mode,low,high=agg.index[0]
 row=CV[(CV.dataset==ds)&(CV.modo==mode)&(CV.alvo_min==low)&(CV.alvo_max==high)].iloc[0]
 baseline=base[base.dataset==ds].iloc[0]
 target_lodo.append(dict(dataset=ds,modo=mode,alvo_min=low,alvo_max=high,media_val_outras_bases=agg.iloc[0],test_accuracy=row.test_accuracy,baseline_test=baseline.test_accuracy,ganho_teste_pp=row.test_accuracy-baseline.test_accuracy))
target_lodo=csv('transferencia_alvo_dgcg_leave_one_dataset_out',pd.DataFrame(target_lodo))
global_best=drank.sort_values('media_val',ascending=False).iloc[0]
section('Um alvo comum ou um alvo específico por dataset?',[
 f'A melhor média de validação entre as nove bases é o modo <strong>{global_best.modo}</strong>, alvo <strong>{global_best.alvo}</strong>: {global_best.media_val:.2f}% de validação e {global_best.media_teste:.2f}% de teste. O baseline 4–6 tem {base.val_accuracy.mean():.2f}% de validação e {base.test_accuracy.mean():.2f}% de teste.',
 f'Escolhendo o alvo somente pela validação das outras oito bases e aplicando à nona, o ganho de teste sobre 4–6 é {target_lodo.ganho_teste_pp.mean():.2f} pp em média. Essa é uma checagem mais exigente de transferência de hiperparâmetro. O treinamento continua específico de cada base.',
 'Uma densidade comum candidata deve ser tratada como ponto de partida para nova validação. A relação grau/densidade depende do número de nós/candidatos. A melhor configuração encontrada neste protocolo não está automaticamente validada para 5-shot ou 10-query.' ],target_lodo)

# Compare tuned sweep with four fixed baselines on same nine bases.
combined=fixed[fixed.dataset.isin(CV.dataset.unique())].copy()
tuned2=tuned[['dataset','test_accuracy','test_ci95','val_accuracy']].copy();tuned2['metodo']='DGCG ajustado (validação)'
fixed_comp=pd.concat([combined[['dataset','metodo','test_accuracy','test_ci95','val_accuracy']],tuned2],ignore_index=True)
csv('comparacao_ajustado_e_baselines_protocolo_fixo',fixed_comp)
f,ax=plt.subplots(figsize=(11,6));frame=fixed_comp.pivot(index='dataset',columns='metodo',values='test_accuracy').reindex(columns=METHODS+['DGCG ajustado (validação)']);im=heat(ax,frame,'5-way / 1-shot / 5-query, nove bases',vmin=0,vmax=100);ax.tick_params(axis='x',labelrotation=20);f.colorbar(im,ax=ax,label='Teste (%)')
fig('13_ajuste_vs_baselines','DGCG ajustado e as baselines no mesmo protocolo','O DGCG ajustado vem de outra rodada: pequenas diferenças de execução existem. Esta comparação é descritiva; ganho de ajuste deve ser medido contra 4–6 da própria rodada.')

# Per dataset dossier and exhaustive tables.
for ds in datasets:
 subset=ALL[ALL.dataset==ds]
 choices=select(subset,['metodo','n_way'])[['metodo','n_way','n_shot','n_query','val_accuracy','test_accuracy','test_ci95','test_loss']]
 csv(f'detalhe_{ds}',choices)
 f,axs=plt.subplots(1,3,figsize=(18,4.8))
 for ax,way in zip(axs,[2,3,5]):
  frame=subset[subset.n_way==way].pivot_table(index='metodo',columns=['n_shot','n_query'],values='test_accuracy').reindex(METHODS)
  frame.columns=[f'{s}s/{q}q' for s,q in frame.columns]
  heat(ax,frame,f'{way}-way',vmin=0,vmax=100);ax.tick_params(axis='x',labelrotation=60,labelsize=7)
 fig(f'base_{ds}',f'Dossiê {NAMES[ds]}: todos os protocolos','s = shot, q = query. Cada painel compara métodos dentro do mesmo n-way. Os números são teste (%).')
 text=[f'Escolhas por validação em {NAMES[ds]}. Acurácia de teste é mostrada com o IC original em coluna separada.']
 if ds=='plant_village_micro':text.append('A busca de densidade/grau desta base não foi concluída, embora a grade de protocolos e as baselines estejam completas.')
 section(f'{NAMES[ds]}: melhores protocolos por validação',text,choices)

# Generalized effects of shot/query per dataset, all results browsable in report.
budget=ALL[ALL.n_way==5].groupby(['metodo','n_shot','n_query']).agg(media_val=('val_accuracy','mean'),media_teste=('test_accuracy','mean'),media_loss=('test_loss','mean')).reset_index()
budget['suporte_rotulado']=5*budget.n_shot
budget['consultas']=5*budget.n_query
budget['nos']=budget.suporte_rotulado+budget.consultas
csv('orcamento_protocolos_5way',budget)
f,ax=plt.subplots(figsize=(9,5))
for m in METHODS:
 sub=budget[budget.metodo==m]
 ax.scatter(sub.nos,sub.media_teste,label=LABEL[m],color=COLORS[m],s=40)
 for _,row in sub.iterrows():
  if m=='dgcg':ax.annotate(f'{int(row.n_shot)}s/{int(row.n_query)}q',(row.nos,row.media_teste),xytext=(3,5),textcoords='offset points',fontsize=8)
ax.set_xlabel('Número de nós em um episódio 5-way');ax.set_ylabel('Teste médio (%)');ax.legend();ax.set_title('Desempenho e quantidade de imagens por episódio')
fig('14_orcamento','Comparar desempenho com o orçamento de imagens','Protocolos com mais imagens tendem a resolver tarefas mais informadas. Número de nós não substitui tempo medido: custo da ordenação/propagação depende do método e da implementação.')
section('Desempenho sob orçamento explícito',[
 'Cada protocolo 5-way usa 5 × shot imagens rotuladas e 5 × query imagens não rotuladas. Compare protocolos com orçamento semelhante antes de chamar uma configuração de melhor. Esta tabela mantém 5-way para tornar explícita a quantidade de informação disponível.' ],budget)

simple=C[(C.status=='ok')&(C.modo=='grau')&(C.alvo_min==5)&(C.alvo_max==5)]
eq=base.merge(simple,on='dataset',suffixes=('_intervalo','_cinco'))
eqcols=['dataset']
for metric in METRICS:
 eq['delta_'+metric]=eq[metric+'_intervalo']-eq[metric+'_cinco']
 eqcols.append('delta_'+metric)
csv('equivalencia_baseline_4a6_grau5',eq[eqcols])
section('Redundância entre alvos e arredondamento',[
 f'Grau-alvo 5 e intervalo 4–6 têm preferência pelo mesmo centro no algoritmo. Nos dados, a maior diferença entre suas seis métricas é {max(eq["delta_"+m].abs().max() for m in METRICS):.6g}. Essa equivalência evita interpretar duas parametrizações do mesmo resultado como evidências independentes.',
 'As métricas foram exportadas com acurácia arredondada a duas casas e loss a quatro. A escolha original do checkpoint usou precisão completa; o resumo exportado não permite recuperar a ordem exata de checkpoints que empatam após arredondamento. Os desempates da análise usam menor loss de validação, regra descrita em cada seleção.' ])

bestmeans=overview.set_index('metodo').media_teste
cleanmeans=clean_over.set_index('metodo').media_teste
fixedmeans=fixed.groupby('metodo').test_accuracy.mean()
execdf=pd.DataFrame([
 dict(questao='Grade completa: DGCG versus recíproco',resultado=f'{bestmeans.dgcg-bestmeans["knn-reciprocal"]:+.3f} pp'),
 dict(questao='Sem episódios até 6 nós: DGCG versus recíproco',resultado=f'{cleanmeans.dgcg-cleanmeans["knn-reciprocal"]:+.3f} pp'),
 dict(questao='Melhor densidade média por validação',resultado=f'{global_best.alvo} ({global_best.media_val:.2f}% val; {global_best.media_teste:.2f}% teste)'),
 dict(questao='Baseline DGCG 4–6 na varredura',resultado=f'{base.test_accuracy.mean():.2f}% teste em nove bases'),
 dict(questao='Ajuste por base: ganho de teste médio',resultado=f'{gains.ganho_teste_pp.mean():+.3f} pp'),
 dict(questao='Transferência de alvo entre bases: ganho médio',resultado=f'{target_lodo.ganho_teste_pp.mean():+.3f} pp')])
csv('conclusoes_principais',execdf)
execbody=''.join(f'<p>{p}</p>' for p in [
 f'<strong>DGCG tem a maior média da grade completa ({bestmeans.dgcg:.2f}%), seguido por kNN recíproco ({bestmeans["knn-reciprocal"]:.2f}%).</strong> Retirando os dois protocolos que tornam o kNN um grafo completo, a diferença cai de {bestmeans.dgcg-bestmeans["knn-reciprocal"]:.3f} para {cleanmeans.dgcg-cleanmeans["knn-reciprocal"]:.3f} pp. Há pouca separação entre eles nessa média ajustada.',
 f'<strong>O ajuste DGCG não produziu melhora média de teste nesta rodada.</strong> Escolher o melhor grau/densidade pela validação de cada base resulta em {gains.ganho_teste_pp.mean():+.3f} pp sobre o baseline 4–6. Ganhos em algumas bases são compensados por perdas em outras.',
 f'<strong>0,125 é uma densidade-alvo candidata, não uma densidade universal demonstrada.</strong> Ela lidera a validação média ({global_best.media_val:.2f}%). O teste médio correspondente ({global_best.media_teste:.2f}%) deve ser comparado ao baseline da própria varredura ({base.test_accuracy.mean():.2f}%). O grau/densidade realmente observado não está disponível.',
 '<strong>Os testes pareados entre quatro métodos não passam de p&lt;0,05 após correção de Holm nas seis comparações da grade completa.</strong> Friedman detecta diferença global entre métodos, mas isso não determina qual par é diferente. Ainda falta avaliar variação entre seeds.',
 '<strong>A validação pode escolher parâmetros que pioram o teste.</strong> Em Manufacturing, o alvo escolhido chega a 100% de validação e cai 4,20 pp no teste contra 4–6. Em Sports, a diferença entre classes de validação e teste também é grande. Isso é consistente com splits de dificuldade distinta; não basta para diagnosticar overfitting.',
 '<strong>Próximo experimento mais informativo:</strong> protocolo fixo, candidatos 0,10/0,125/0,15 e baseline 4–6, junto às três baselines kNN e ProtoNet, com várias seeds e registro de densidade realizada.' ])+'<div class="scroll">'+tab(execdf)+'</div>'
SECTIONS.insert(0,('Principais conclusões e sua força',execbody))

section('O que testar na próxima rodada',[
 '<strong>1. Completar Plant Village na busca de grau/densidade.</strong> Seus dados de protocolo já estão completos. Comparações atuais da busca de ajuste usam nove bases.',
 '<strong>2. Fixar a pergunta experimental.</strong> Para comparar métodos, use o mesmo n-way, shot e query. Para comparar orçamento de suporte, mantenha n-way/query e varie shot. Para estudar transdução, mantenha n-way/shot e varie query.',
 '<strong>3. Repetir candidatos com pelo menos cinco seeds.</strong> Inclua kNN recíproco, kNN-out, kNN-in, DGCG 4–6 e a densidade candidata. Use mesmos splits/seeds por método; varie também split em um experimento separado para avaliar transferência entre classes.',
 '<strong>4. Registrar densidade e grau realizados por episódio.</strong> Guarde número de arestas sem autolaços, grau de entrada/saída, densidade sobre N(N−1), densidade sobre candidatos, nós isolados antes do fallback, distribuição de graus e limiar escolhido.',
 '<strong>5. Guardar predições ou acurácia por episódio, seed, configuração e versão do código.</strong> Isso permite testes pareados corretos, avaliar calibração, matrizes de confusão, custo de treinamento e reproduzir divergências entre ambientes.',
 '<strong>6. Incluir ProtoNet sem grafo e k menores em episódios pequenos.</strong> As três baselines fornecidas são Proto-SGC com topologias diferentes, não um classificador kNN clássico de vizinhos votando em rótulos. ProtoNet ajuda a medir se propagação traz valor.',
 '<strong>7. Refinar localmente a densidade candidata.</strong> Só depois de uma busca ampla, tente valores intermediários e verifique estabilidade por base/protocolo. Preserve o teste para a avaliação da configuração escolhida na validação.' ])

browser=ALL[['metodo','dataset','n_way','n_shot','n_query','nos','val_accuracy','test_accuracy','test_ci95','test_loss','gap_val_teste']].copy()
data=browser.to_dict(orient='records')
SECTIONS.append(('Explore as 1.080 combinações', '<p>Filtre método, dataset e protocolo. Clique nos cabeçalhos para ordenar. A tabela apresenta até 200 linhas por filtro; o CSV consolidado contém todas as linhas.</p><div id="filters"></div><p id="count"></p><div class="scroll"><table id="browser"></table></div>'))
css='''body{margin:0;background:#f1f5f8;color:#223244;font:16px/1.65 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:34px}h1{font-size:38px;line-height:1.15}h2{font-size:25px;line-height:1.3}header,section{background:white;padding:28px;margin-bottom:22px;border-radius:12px;border:1px solid #dbe4ed}p{max-width:1100px}nav{display:flex;gap:10px;flex-wrap:wrap}nav a{font-size:13px;color:#25618e}img{display:block;width:100%;height:auto}figure{margin:0}figcaption{font-size:14px;color:#526578;padding-top:12px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;padding:8px 10px;border-bottom:1px solid #e0e6ec;white-space:nowrap}th{background:#edf3f8;position:sticky;top:0}td:first-child,th:first-child{text-align:left}select{padding:9px;margin:5px;border:1px solid #ccd5df;border-radius:5px}#browser th{cursor:pointer}code{background:#eef3f7;padding:2px 4px}@media print{body{background:white}section{break-inside:avoid}nav,#filters{display:none}}'''
body='<header><h1>Few-shot DGCG: análise dos experimentos</h1><p>1.332 observações válidas • 10 datasets • 4 topologias • 27 protocolos • 28 alvos DGCG em 9 bases</p><p>Leitura: médias gerais → comparação fixa → efeitos de protocolo → densidade/grau → dossiês de cada base.</p><nav>'+''.join(f'<a href="#s{i}">{html.escape(t)}</a>' for i,(t,_) in enumerate(SECTIONS))+'</nav></header>'
body+=''.join(f'<section id="s{i}"><h2>{html.escape(title)}</h2>{content}</section>' for i,(title,content) in enumerate(SECTIONS))
js='''const rows=DATA;const keys=Object.keys(rows[0]);let sortKey='test_accuracy',desc=true;const filterKeys=['metodo','dataset','n_way','n_shot','n_query'];const f=document.getElementById('filters');for(const k of filterKeys){const s=document.createElement('select');s.dataset.key=k;s.innerHTML='<option value="">'+k+': todos</option>'+[...new Set(rows.map(r=>r[k]))].sort().map(v=>'<option>'+v+'</option>').join('');s.onchange=render;f.appendChild(s)}function render(){let a=rows.filter(r=>[...f.children].every(s=>!s.value||String(r[s.dataset.key])===s.value));a.sort((x,y)=>{let v=typeof x[sortKey]==='number'?x[sortKey]-y[sortKey]:String(x[sortKey]).localeCompare(String(y[sortKey]));return desc?-v:v});document.getElementById('count').textContent=a.length+' resultados; mostrando '+Math.min(200,a.length);let t=document.getElementById('browser');t.innerHTML='<thead><tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>'+a.slice(0,200).map(r=>'<tr>'+keys.map(k=>'<td>'+(typeof r[k]==='number'?(Number.isInteger(r[k])?r[k]:r[k].toFixed(3)):r[k])+'</td>').join('')+'</tr>').join('')+'</tbody>';[...t.querySelectorAll('th')].forEach((th,i)=>th.onclick=()=>{if(sortKey===keys[i])desc=!desc;else{sortKey=keys[i];desc=true}render()})}render();'''
(ROOT/'relatorio.html').write_text('<!DOCTYPE html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Análise few-shot DGCG</title><style>'+css+'</style><main>'+body+'</main><script>'+js.replace('DATA',json.dumps(data,ensure_ascii=False))+'</script></html>',encoding='utf-8')

summary={
 'raw_rows':len(A)+len(B)+len(C),'unique_valid':len(ALL)+len(CV),'incomplete':len(C)-len(CV),
 'overview':overview.to_dict('records'),'overview_no_small':clean_over.to_dict('records'),
 'best_density':global_best.to_dict(),'baseline_sweep_test':base.test_accuracy.mean(),
 'local_gain_test':gains.ganho_teste_pp.mean(),'transfer_gain_test':target_lodo.ganho_teste_pp.mean(),
 'gains_by_dataset':gain_display.to_dict('records'),'friedman_p':friedman.pvalue,
 'summary_audit':reconc,'fixed_means':fixed.groupby('metodo').test_accuracy.mean().to_dict(),
 'protocol_choices':selected.groupby(['metodo','n_way']).agg(shot_mediano=('n_shot','median'),query_mediano=('n_query','median'),test_medio=('test_accuracy','mean')).reset_index().to_dict('records')}
(ROOT/'achados.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
source_copy=ROOT/'fontes'
source_copy.mkdir(exist_ok=True)
for name in FILES:shutil.copy2(SOURCE/name,source_copy/name)
(ROOT/'ambiente_analise.json').write_text(json.dumps({'python':__import__('platform').python_version(),'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'scipy':__import__('scipy').__version__,'numero_graficos':len(FIGURES),'numero_tabelas':len(TABLES)},indent=2),encoding='utf-8')
md=['# Análise dos experimentos few-shot DGCG','',
 'Abra `relatorio.html` para o relatório completo, com explicações, tabelas e filtros. Ele funciona offline; todos os gráficos estão embutidos.',
 '',f'Foram analisadas {len(ALL)+len(CV)} observações válidas após remover 270 duplicações. Há 28 ajustes incompletos em Plant Village.',
    '', '```text',overview.to_string(index=False,float_format=lambda x:f'{x:.3f}'),'```',
 '',f'Melhor alvo comum por validação: {global_best.modo} {global_best.alvo}, validação {global_best.media_val:.2f}%, teste {global_best.media_teste:.2f}%. Baseline 4–6 teste {base.test_accuracy.mean():.2f}%.',
 f'Ganho médio de ajuste por base selecionado pela validação: {gains.ganho_teste_pp.mean():.2f} pp. Ganho em transferência leave-one-dataset-out: {target_lodo.ganho_teste_pp.mean():.2f} pp.',
 '', '## Limites das conclusões',
 'Uma rodada por configuração. IC original mede episódios, não seeds. Densidade realizada não foi registrada. Comparações entre n-way mudam a tarefa. Sem predições individuais não é possível avaliar calibração ou confusão de classes. Grau/densidade foram ajustados só em 5-way/1-shot/5-query.',
 '', '## Gráficos','']
for name,title,caption in FIGURES:md.extend([f'### {title}',f'![{title}](graficos/{name}.png)',caption,''])
(ROOT/'LEIA_ME.md').write_text('\n'.join(md),encoding='utf-8')
# Export publication-oriented selected 5-way results (selection on validation).
pub=selected[selected.n_way==5].pivot(index='dataset',columns='metodo',values='test_accuracy').reindex(columns=METHODS)
latex=['\\begin{tabular}{lrrrr}','\\hline','Dataset & kNN-in & kNN-out & kNN-rec & DGCG \\\\','\\hline']
for ds,row in pub.iterrows():latex.append(NAMES[ds]+' & '+' & '.join(f'{x:.2f}' for x in row)+' \\\\')
latex.extend(['\\hline','\\end{tabular}','% 5-way: shot/query selecionados pela validacao. Orcamentos diferentes.'])
(TABLE/'tabela_5way_selecionada_validacao.tex').write_text('\n'.join(latex),encoding='utf-8')
with zipfile.ZipFile(ROOT/'analise_completa.zip','w',zipfile.ZIP_DEFLATED) as z:
 for path in ROOT.rglob('*'):
  if path.is_file() and path.name!='analise_completa.zip':z.write(path,path.relative_to(ROOT))
print(json.dumps(summary,ensure_ascii=False,indent=2))
