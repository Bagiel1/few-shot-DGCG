"""Leitura/auditoria dos CSVs recebidos; nao altera os arquivos de origem."""
from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd

SRC=Path('/home/bagiel/Downloads')
D=pd.read_csv(SRC/'resultados_densidades_protocolos_dgcg.csv')
S=pd.read_csv(SRC/'melhores_densidades_por_protocolo_dgcg.csv')
V=pd.read_csv(SRC/'variacao_densidades_dgcg.csv')
KEY=['dataset','n_way','n_shot','n_query','seed']
NKEY=KEY+['alvo']
RHO=[.025,.05,.075,.1,.125,.15,.175,.2,.25,.3,.4,.5,.6,.75,.9]

def show(title,obj):
    print('\n'+title)
    print(obj.to_string(index=False) if isinstance(obj,pd.DataFrame) else obj)

assert len(D)==4050 and len(S)==270 and len(V)==270
assert D[NKEY].duplicated().sum()==0
assert S[KEY].duplicated().sum()==0
assert set(D['alvo'])==set(RHO)
expected=set(itertools.product(sorted(D.dataset.unique()),[2,3,5],[1,5,10],[1,5,10],[7],RHO))
assert set(D[NKEY].itertuples(index=False,name=None))==expected
assert D.baseline_status.eq('compativel').all()
assert D.test_episodes.eq(1000).all() and D.val_episodes.eq(200).all()
assert np.allclose(D.episode_nodes,D.n_way*(D.n_shot+D.n_query))
assert np.allclose(D.candidatos_por_no,D.episode_nodes-1)
assert np.allclose(D.densidade_alvo,D.alvo)
assert np.allclose(D.grau_alvo_equivalente,D.alvo*(D.episode_nodes-1))
assert np.allclose(D.ganho_teste_vs_4_6_pp,D.test_accuracy-D.baseline_test_accuracy)
for split in ['val','test']:
    assert D[f'{split}_accuracy'].between(0,100).all()
    assert D[f'{split}_macro_f1'].between(0,100).all()
    assert D[f'{split}_density_mean'].between(0,1).all()
    assert np.allclose(D[f'{split}_nodes_per_episode'],D.episode_nodes)
    assert np.allclose(D[f'{split}_degree_mean']/(D.episode_nodes-1),D[f'{split}_density_mean'])
    assert np.allclose(D[f'{split}_desvio_densidade'],D[f'{split}_density_mean']-D.alvo)
    assert np.allclose(D[f'{split}_accuracy'],D[f'{split}_balanced_accuracy'],atol=1e-5)
    assert (D[f'{split}_isolated_node_fraction']==0).all()
baseline=D.groupby(KEY)[['baseline_val_accuracy','baseline_test_accuracy']].first().reset_index()
assert D.groupby(KEY)[['baseline_val_accuracy','baseline_test_accuracy']].nunique().eq(1).all().all()
# Recalcula a selecao independentemente: maior val; empate escolhe menor rho.
recomputed=D.sort_values(['val_accuracy','alvo'],ascending=[False,True]).drop_duplicates(KEY)
audit=S.merge(recomputed,on=KEY,suffixes=('_csv','_calc'),validate='one_to_one')
for col in D.select_dtypes('number').columns:
    if col not in KEY:
        assert np.allclose(audit[f'{col}_csv'],audit[f'{col}_calc'],equal_nan=True),col
for _,row in S.iterrows():
    group=D.loc[(D[KEY]==row[KEY]).all(axis=1)]
    ties=group.loc[group.val_accuracy.eq(group.val_accuracy.max()),'alvo'].sort_values().tolist()
    assert np.allclose(ties,[float(v) for v in str(row.alvos_empatados).split(';')])
    assert row.numero_alvos_testados==15
# Recalcula os tres eixos da tabela de variacao sem confiar no resumo recebido.
for _,row in V.iterrows():
    axis=row.eixo_variado
    fixed=[k for k in KEY if k!=axis]
    mask=np.ones(len(S),dtype=bool)
    for k in fixed:mask &= S[k].eq(row[k])
    group=S.loc[mask].sort_values(axis)
    assert len(group)==3
    assert group.alvo.nunique()==row.densidades_distintas
    assert np.isclose(group.alvo.max()-group.alvo.min(),row.amplitude_densidade)
    assert np.isclose(group.alvo.std(ddof=0),row.desvio_densidade)

S=S.copy()
S['num_empates']=S.alvos_empatados.astype(str).str.split(';').str.len()
S['densidade_min_empatada']=S.alvos_empatados.astype(str).map(lambda x:min(map(float,x.split(';'))))
S['densidade_max_empatada']=S.alvos_empatados.astype(str).map(lambda x:max(map(float,x.split(';'))))
show('AUDITORIA',dict(linhas=len(D),bases=D.dataset.nunique(),protocolos=len(S),
    seed=D.seed.unique().tolist(),ausentes=int(D.isna().sum().sum()),grupos_com_empates=int((S.num_empates>1).sum()),
    max_diferenca_BA_accuracy=float((D.test_accuracy-D.test_balanced_accuracy).abs().max())))

def aggregate(frame):
    return dict(n=len(frame),teste=float(frame.test_accuracy.mean()),
       f1=float(frame.test_macro_f1.mean()),baseline=float(frame.baseline_test_accuracy.mean()),
       ganho=float(frame.ganho_teste_vs_4_6_pp.mean()),mediana_ganho=float(frame.ganho_teste_vs_4_6_pp.median()),
       melhora=int((frame.ganho_teste_vs_4_6_pp>1e-6).sum()),
       empate=int((frame.ganho_teste_vs_4_6_pp.abs()<=1e-6).sum()),
       piora=int((frame.ganho_teste_vs_4_6_pp< -1e-6).sum()))

show('SELECAO POR PROTOCOLO: todos',aggregate(S))
show('SELECAO POR PROTOCOLO: N>6',aggregate(S.loc[S.episode_nodes>6]))
show('SELECAO: episodios pequenos N<=6',aggregate(S.loc[S.episode_nodes<=6]))
show('POR WAY',pd.DataFrame([dict(n_way=w,**aggregate(g)) for w,g in S.groupby('n_way')]))
show('POR BASE',pd.DataFrame([dict(dataset=ds,**aggregate(g)) for ds,g in S.groupby('dataset')]))
show('FREQUENCIA DAS ESCOLHAS',S.alvo.value_counts().sort_index())
show('EMPATES',S.num_empates.value_counts().sort_index())
show('ESCOLHAS POR WAY',pd.crosstab(S.n_way,S.alvo).reset_index())
show('VARIACAO POR EIXO',V.groupby('eixo_variado').agg(grupos=('amplitude_densidade','size'),
    mudaram=('amplitude_densidade',lambda x:(x>1e-10).sum()),
    amplitude_media=('amplitude_densidade','mean'),amplitude_max=('amplitude_densidade','max')).reset_index())
rank=D.groupby('alvo').agg(validacao=('val_accuracy','mean'),teste=('test_accuracy','mean'),
    f1=('test_macro_f1','mean'),rho_real=('test_density_mean','mean'),
    ganho=('ganho_teste_vs_4_6_pp','mean')).sort_values('validacao',ascending=False)
show('RANKING GLOBAL POR VALIDACAO',rank.reset_index())
show('RANKING N>6',D.loc[D.episode_nodes>6].groupby('alvo').agg(
    validacao=('val_accuracy','mean'),teste=('test_accuracy','mean'),
    ganho=('ganho_teste_vs_4_6_pp','mean')).sort_values('validacao',ascending=False).reset_index())
show('PROTOCOLO FIXO 5W 1S 5Q',aggregate(S.loc[(S.n_way==5)&(S.n_shot==1)&(S.n_query==5)]))
show('MAIORES MELHORAS',S.nlargest(8,'ganho_teste_vs_4_6_pp')[KEY+['alvo','test_accuracy','baseline_test_accuracy','ganho_teste_vs_4_6_pp','num_empates']])
show('MAIORES QUEDAS',S.nsmallest(8,'ganho_teste_vs_4_6_pp')[KEY+['alvo','test_accuracy','baseline_test_accuracy','ganho_teste_vs_4_6_pp','num_empates']])
show('DENSIDADE ALVO VS REAL POR TAMANHO',D.groupby('episode_nodes').agg(n=('alvo','size'),
    erro_medio=('test_desvio_densidade','mean'),
    erro_absoluto=('test_desvio_densidade',lambda x:x.abs().mean())).reset_index())
show('DENSIDADES IGUAIS POR EPISODIOS PEQUENOS',D.loc[(D.dataset=='bacteria_micro')&(D.episode_nodes==4),
    ['alvo','val_accuracy','test_accuracy','test_density_mean','test_degree_mean']])

# Uma densidade global escolhida em outras nove bases por protocolo (LODO).
external=[]
for (way,shot,query,seed),group in D.groupby(['n_way','n_shot','n_query','seed']):
    for ds in sorted(D.dataset.unique()):
        training=group.loc[group.dataset!=ds]
        means=training.groupby('alvo').val_accuracy.mean()
        target=float(means.loc[np.isclose(means,means.max(),atol=1e-12,rtol=0)].index.min())
        selected=group.loc[group.dataset.eq(ds)&group.alvo.eq(target)].iloc[0].to_dict()
        external.append(selected)
E=pd.DataFrame(external)
show('LODO: escolha em outras nove bases, mesmo protocolo',aggregate(E))
show('LODO: N>6',aggregate(E.loc[E.episode_nodes>6]))

summary={
    'auditoria':'4050 linhas unicas; 270 grupos completos; 3 CSVs reconciliados',
    'selecao_por_protocolo':aggregate(S),'selecao_sem_episodios_pequenos':aggregate(S.loc[S.episode_nodes>6]),
    'lodo':aggregate(E),'lodo_sem_pequenos':aggregate(E.loc[E.episode_nodes>6]),
    'ranking_global':rank.reset_index().to_dict('records'),
    'por_base':[dict(dataset=ds,**aggregate(g)) for ds,g in S.groupby('dataset')],
    'por_way':[dict(n_way=int(w),**aggregate(g)) for w,g in S.groupby('n_way')],
    'empates':int((S.num_empates>1).sum()),'frequencias':S.alvo.value_counts().sort_index().to_dict(),
}
Path('/tmp/fewshot_densidades_20261026_resumo.json').write_text(json.dumps(summary,indent=2))
