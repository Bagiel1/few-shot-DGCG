import pandas as pd,numpy as np,json
from pathlib import Path
out=Path('tmp/pdfs/extras_20261030')
d=pd.read_csv('/home/bagiel/Downloads/buscas_extras.csv');m=pd.read_csv('/home/bagiel/Downloads/medias_buscas_extras.csv');s=pd.read_csv('/home/bagiel/Downloads/status_buscas_extas.csv')
k=['n_way','n_shot','n_query','metodo','alvo','extra_mode','extra_phase','extras_per_class'];full=['dataset','seed']+k
assert not d.duplicated(full).any() and not s.duplicated(full).any()
assert set(map(tuple,d[full].values))==set(map(tuple,s[full].values))
assert (s.status=='complete').all()
assert len(d)==4740 and len(d[d.extras_per_class==0])==420
assert (d.groupby(['dataset','n_way','n_shot']).size()==79).all()
assert not d[['val_accuracy','test_accuracy','test_macro_f1','test_balanced_accuracy']].isna().any().any()
assert set(d.seed)=={7} and set(d.bank_per_class)=={20}
assert set(d.val_episodes)=={200} and set(d.test_episodes)=={1000}
assert (d.extras_total==d.extras_per_class*d.n_way).all()
basekeys=['dataset','seed','n_way','n_shot','n_query','metodo','alvo']
b=d[d.extras_per_class==0]
x=d.merge(b[basekeys+['test_accuracy']],on=basekeys,suffixes=('','_base'),validate='many_to_one')
err=(x.test_accuracy-x.test_accuracy_base-x.ganho_accuracy_vs_sem_extras_pp).abs().max()
cols=['val_accuracy','test_accuracy','test_macro_f1','test_balanced_accuracy','ganho_accuracy_vs_sem_extras_pp']
a=d.groupby(k)[cols].mean().reset_index();z=m.merge(a,on=k,suffixes=('','_calc'),validate='one_to_one');me=max((z[c]-z[c+'_calc']).abs().max() for c in cols)
assert err<1e-8 and me<1e-8 and len(z)==474
# means for equal weighting across datasets and protocols
print('AUDIT',len(d),'controls',len(b),'mean error',me,'gain error',err)
print('BASELINES\n',b.groupby(['metodo','alvo']).test_accuracy.mean().to_string())
print('GAIN BY SHOT / PHASE\n',d[d.extras_per_class>0].groupby(['n_shot','extra_phase']).ganho_accuracy_vs_sem_extras_pp.mean().unstack().round(3).to_string())
print('DGCG ALL gains by quantity\n',d[(d.metodo=='dgcg')&(d.extra_phase=='all')].groupby(['alvo','extra_mode','extras_per_class']).ganho_accuracy_vs_sem_extras_pp.mean().round(3).to_string())
# choose single common condition across ten bases by validation within protocol, ALL phase or control
sel=a[a.extra_phase.isin(['none','all'])].sort_values(['val_accuracy','extras_per_class'],ascending=[False,True]).groupby(['n_way','n_shot','n_query']).head(1).sort_values(['n_way','n_shot'])
ng=a[a.metodo=='no-graph'][['n_way','n_shot','test_accuracy']].rename(columns={'test_accuracy':'no_graph'})
sel=sel.merge(ng,on=['n_way','n_shot']);sel['gain_ng']=sel.test_accuracy-sel.no_graph
print('SELECTED VAL\n',sel.to_string(index=False));sel.to_csv(out/'selecao.csv',index=False)
print('ALL 20 by method\n',d[(d.extra_phase=='all')&(d.extras_per_class==20)].groupby(['metodo','alvo','extra_mode'])[['test_accuracy','ganho_accuracy_vs_sem_extras_pp']].mean().round(3).to_string())
print('GAIN per protocol DGCG all20\n',a[(a.metodo=='dgcg')&(a.extra_phase=='all')&(a.extras_per_class==20)].pivot_table(index=['n_way','n_shot'],columns=['alvo','extra_mode'],values='ganho_accuracy_vs_sem_extras_pp').round(3).to_string())
print('NO GRAPH',ng.to_string(index=False))
print('BEST TEST diagnostic\n',a.sort_values('test_accuracy',ascending=False).groupby(['n_way','n_shot']).head(1).to_string(index=False))
old=Path('runs/busca_extras_auto/resultados.csv')
if old.exists():
 oldd=pd.read_csv(old);j=d.merge(oldd,on=full,suffixes=('','_old')); print('OLD MATCH',len(j),'test max diff',(j.test_accuracy-j.test_accuracy_old).abs().max())
a.to_csv(out/'medias_conferidas.csv',index=False)
