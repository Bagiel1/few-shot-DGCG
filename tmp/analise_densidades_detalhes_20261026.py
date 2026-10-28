from pathlib import Path
import pandas as pd
import numpy as np

SRC=Path('/home/bagiel/Downloads')
D=pd.read_csv(SRC/'resultados_densidades_protocolos_dgcg.csv')
S=pd.read_csv(SRC/'melhores_densidades_por_protocolo_dgcg.csv')
K=['dataset','n_way','n_shot','n_query','seed']
S['ties']=S.alvos_empatados.astype(str).map(lambda v:set(map(float,v.split(';'))))

print('GANHO POR BASE, SEM N<=6')
print(S.loc[S.episode_nodes>6].groupby('dataset').agg(
    teste=('test_accuracy','mean'),baseline=('baseline_test_accuracy','mean'),
    ganho=('ganho_teste_vs_4_6_pp','mean')).to_string())
print('FREQUENCIA DOS VENCEDORES UNICOS')
print(S.loc[S.ties.map(len)==1,'alvo'].value_counts().sort_index().to_string())
print('VARIACAO COM EMPATES CONSIDERADOS')
for axis in ['n_way','n_shot','n_query']:
    fixed=[k for k in K if k!=axis]
    groups=list(S.groupby(fixed))
    no_common=sum(not set.intersection(*g.ties.tolist()) for _,g in groups)
    chosen_changed=sum(g.alvo.nunique()>1 for _,g in groups)
    print(axis,'grupos',len(groups),'escolha_mudou',chosen_changed,
          'sem_nenhum_otimo_comum',no_common)

print('TAMANHO, MEDIA DENSIDADE SELECIONADA E GRAU REAL')
print(S.groupby('episode_nodes').agg(n=('alvo','size'),rho=('alvo','median'),
    rho_real=('test_density_mean','median'),grau=('test_degree_mean','median'),
    ganho=('ganho_teste_vs_4_6_pp','mean')).to_string())
print('UM GLOBAL POR WAY, ESCOLHIDO NA VALIDACAO')
for way,group in D.groupby('n_way'):
    ranking=group.groupby('alvo').val_accuracy.mean()
    rho=ranking.idxmax()
    chosen=group.loc[group.alvo==rho]
    print(way,rho,'val',chosen.val_accuracy.mean(),'teste',chosen.test_accuracy.mean(),
          'ganho',chosen.ganho_teste_vs_4_6_pp.mean())
print('PROTOCOLO FIXO POR BASE')
print(S.loc[(S.n_way==5)&(S.n_shot==1)&(S.n_query==5),
    ['dataset','alvo','val_accuracy','test_accuracy','baseline_val_accuracy','baseline_test_accuracy',
     'ganho_teste_vs_4_6_pp']].to_string(index=False))
print('PROTOCOLO FIXO RANKING GLOBAL')
print(D.loc[(D.n_way==5)&(D.n_shot==1)&(D.n_query==5)].groupby('alvo').agg(
    val=('val_accuracy','mean'),test=('test_accuracy','mean')).sort_values('val',ascending=False).head(5).to_string())

print('ISOLAMENTO E ADJACENCIA')
print('zero_row max',D.test_zero_row_fraction.max(),'min density',D.test_density_mean.min(),
      'macro<=accuracy?',(D.test_macro_f1<=D.test_accuracy+1e-5).all())
