from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('outputs/explicacao_fewshot_20261103/fontes'); O=Path('tmp/pdfs/explicacao_20261103/figuras')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':12,'savefig.facecolor':'white'})
C=['#16857c','#df8736','#5266a6','#a64d65']; D={p.stem:pd.read_json(p,lines=p.suffix=='.jsonl') for p in R.glob('res_*')}
e=D['res_e1']; raw=e[e.feat.eq('raw')]; ng=raw[raw.method.eq('no-graph')&raw.trainfree].set_index(['dataset','shot']).test
names={'bacteria_micro':'Bacteria','plankton_micro':'Plankton','flowers_micro':'Flowers','sports_micro':'Sports','birds_micro':'Birds','plant_village_micro':'PlantVillage','textures_micro':'Textures','manufacturing_micro':'Manufacturing','cars_micro':'Cars','resisc_micro':'RESISC'}
methods={'no-graph':'Sem grafo','union':'kNN-union','dgcg':'DGCG 4–6','dgcg24':'DGCG 2–4'}
def save(name):
 plt.tight_layout(pad=1.2);plt.savefig(O/(name+'.png'),dpi=180);plt.close()
def barlabels(ax,fmt='%.2f'):
 for con in ax.containers: ax.bar_label(con,fmt=fmt,fontsize=8,padding=3)
def heat(data,name,title,v=8,height=4.2):
 fig,ax=plt.subplots(figsize=(9,height));im=ax.imshow(data.values,cmap='RdBu',vmin=-v,vmax=v,aspect='auto');ax.set_xticks(range(len(data.columns)),data.columns);ax.set_yticks(range(len(data)),data.index);ax.set_title(title,pad=13)
 for i in range(len(data)):
  for j in range(len(data.columns)):
   val=data.iloc[i,j];ax.text(j,i,f'{val:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(val)>v*.65 else '#152b3c')
 fig.colorbar(im,ax=ax,label='Diferença de acurácia (pontos percentuais)',shrink=.8);save(name)
# overview
fig,axs=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
for ax,s in zip(axs,[1,5]):
 p=raw[raw.shot.eq(s)].groupby(['method','trainfree']).test.mean().unstack().reindex(methods.keys()).dropna();p.columns=['Com treino padrão','Sem treino'];p.index=[methods[x] for x in p.index];p.plot.bar(ax=ax,color=C[:2],rot=0);barlabels(ax);ax.set_title(f'{s}-shot | 10 bases');ax.set_ylim(50,81);ax.set_ylabel('Acurácia de teste (%)');ax.legend(fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.15)
save('treino')
for s in [1,5]:
 p=raw[raw.shot.eq(s)].pivot(index='dataset',columns=['method','trainfree'],values='test');h=pd.DataFrame({methods[m]:p[(m,False)]-p[(m,True)] for m in ['no-graph','union','dgcg']});h.index=h.index.map(names);heat(h,f'treino_base{s}',f'{s}-shot: treino padrão menos sem treino',v=15,height=4.6)
# init
fig,axs=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
idn=D['res_idinit'];d24=D['res_dgcg24']; b=pd.concat([raw,d24[d24.feat.eq('raw')]])
for ax,s in zip(axs,[1,5]):
 rows=[]
 for m in ['no-graph','union','dgcg24']:
  rows.append([b[(b.shot==s)&(b.method==m)&b.trainfree].test.mean(),b[(b.shot==s)&(b.method==m)&~b.trainfree].test.mean(),idn[(idn.shot==s)&(idn.method==m)].test.mean()])
 pd.DataFrame(rows,index=[methods[m] for m in ['no-graph','union','dgcg24']],columns=['Identidade 512, sem treino','Aleatória 256, treino','Identidade 512, treino']).plot.bar(ax=ax,color=C[:3],rot=0);ax.set_ylim(53,83);ax.set_title(f'{s}-shot');barlabels(ax);ax.legend(fontsize=6.8,loc='upper left');ax.set_ylabel('Teste (%)')
save('init')
# preprocessing
for tf in [False,True]:
 p=e[e.trainfree.eq(tf)].groupby(['method','shot','feat']).test.mean().unstack();h=p[['cl2n','pt-cl2n']].sub(p.raw,axis=0);h.index=[f'{methods[m]} · {s}-shot' for m,s in h.index];heat(h,'preproc_'+str(tf),('Sem treino' if tf else 'Com treino padrão')+': ganho sobre raw',v=2,height=3.7)
# DGCG degrees
fig,ax=plt.subplots(figsize=(9,3.4));p=b[b.method.isin(['dgcg','dgcg24'])].groupby(['shot','trainfree','method']).test.mean().unstack();p.index=[f'{s}-shot · '+('sem treino' if t else 'treinado') for s,t in p.index];p.rename(columns=methods).plot.bar(ax=ax,color=C[:2],rot=0);barlabels(ax);ax.set_ylim(55,79);ax.set_ylabel('Teste (%)');save('degrees')
# Prop
prop=D['res_proptf'];pbase=b[b.trainfree&b.method.isin(['union','dgcg24'])].copy();pbase['tag']='hops2';pr=pd.concat([prop,pbase]);fig,axs=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
for ax,m in zip(axs,['union','dgcg24']):
 for s,col in zip([1,5],C):
  vals=[pr[(pr.method==m)&(pr.shot==s)&(pr.tag==t)].test.mean() for t in ['hops1','hops2','hops3','hops4','alpha0.5','hops4alpha0.5']];ax.plot(range(6),vals,'o-',label=f'{s}-shot',color=col)
 ax.set_xticks(range(6),['K=1','K=2','K=3','K=4','K=2\nα=.5','K=4\nα=.5']);ax.set_title(methods[m]);ax.legend();ax.set_ylabel('Teste (%)');ax.grid(alpha=.2)
save('prop')
# topology variants all
T=D['res_topo_tf']; rows=[]
for (m,t,s),g in T[T.method.ne('no-graph')].groupby(['method','tag','shot']):
 base=T[(T.method==('dgcg' if m=='dgcgplus' else m))&(T.tag=='')&(T.shot==s)].set_index('dataset').test
 vals=g.set_index('dataset').test-base
 rows.append({'variant':('DGCG+' if m=='dgcgplus' else methods[m]+' / '+(t or 'base')),'shot':s,'delta':vals.mean(),'test':g.test.mean(),'wins':int((vals>0).sum())})
rt=pd.DataFrame(rows);h=rt.pivot(index='variant',columns='shot',values='delta');h.columns=['1-shot','5-shot'];heat(h,'topo','Topologias sem treino: ganho sobre o próprio grafo',v=2,height=6)
# trained topology
tr=D['res_topo_tr'];r=[]
for (m,s),g in tr.groupby(['method','shot']):
 ba=raw[(raw.method==m)&(raw.shot==s)&~raw.trainfree].set_index('dataset').test;delta=g.set_index('dataset').test-ba;r.append(delta.rename(f'{methods[m]} / {s}-shot'))
h=pd.concat(r,axis=1);h.index=h.index.map(names);heat(h,'topotr','RBF cosseno com treino menos grafo treinado original',v=3,height=4.8)
# homophily
fig,axs=plt.subplots(1,2,figsize=(9,4.2));corr={}
for ax,s in zip(axs,[1,5]):
 z=T[T.tag.eq('')&T.method.isin(['union','dgcg'])&T.shot.eq(s)].copy();z['gain']=[r.test-ng.loc[(r.dataset,s)] for r in z.itertuples()];corr[s]=z.hom_query.corr(z.gain)
 for m,col in zip(['union','dgcg'],C):
  zz=z[z.method.eq(m)];ax.scatter(zz.hom_query,zz.gain,label=methods[m],c=col,s=34)
  for r in zz.itertuples():
   if r.dataset in ['cars_micro','textures_micro','flowers_micro','manufacturing_micro']:ax.annotate(names[r.dataset],(r.hom_query,r.gain),fontsize=6,xytext=(2,3),textcoords='offset points')
 ax.axhline(0,color='#667788',lw=.7);ax.set_title(f'{s}-shot | r = {corr[s]:.2f} (20 pontos)');ax.set_xlabel('Homofilia nas linhas de query (%)');ax.set_ylabel('Grafo − sem grafo (pp)');ax.legend(fontsize=8);ax.grid(alpha=.12)
save('hom')
# proxy
S=D['res_supproxy'];fig,axs=plt.subplots(1,2,figsize=(9,4))
for m,col in zip(['union','dgcg'],C):
 z=S[S.method==m];axs[0].scatter(z.sup_hom,z.q_hom,c=col,label=f'{m}: r={z.sup_hom.corr(z.q_hom):.3f}')
axs[0].set_xlabel('Homofilia entre suportes (%)');axs[0].set_ylabel('Homofilia das queries (%)');axs[0].set_title('Uma média por base');axs[0].legend(fontsize=8)
p=S.pivot(index='dataset',columns='method',values='ep_corr');p.index=p.index.map(names);p.plot.barh(ax=axs[1],color=[C[1],C[0]]);axs[1].axvline(0,color='gray',lw=.7);axs[1].set_xlim(-.3,.3);axs[1].set_title('Correlação entre episódios da mesma base');axs[1].set_ylabel('');axs[1].set_xlabel('r de Pearson');axs[1].legend(fontsize=7);save('proxy')
# adaptive
A=D['res_adapt'];fig,axs=plt.subplots(1,2,figsize=(9,3.6))
x=np.linspace(0,1,101)
for lo,hi,col in [(.35,.75,C[0]),(.45,.85,C[1])]:axs[0].plot(x*100,np.clip((x-lo)/(hi-lo),0,1),label=f'lo={lo}; hi={hi}',color=col,lw=2)
axs[0].set_xlabel('Homofilia entre suportes (%)');axs[0].set_ylabel('Peso w dado à propagação');axs[0].legend(fontsize=8);axs[0].set_title('Regra usada nos testes')
vals=[];labels=[]
for m in ['union','dgcg']:
 for tag in ['adapt','adapt0.45-0.85']:
  z=A[(A.method==m)&(A.tag==tag)];vals.append(z.test.mean());labels.append(m+(' .35/.75' if tag=='adapt' else ' .45/.85'))
axs[1].barh(labels,vals,color=[C[0],C[0],C[1],C[1]]);axs[1].axvline(ng.xs(5,level='shot').mean(),color=C[2],ls='--',label='Sem grafo: 76,096%');axs[1].set_xlim(73.5,76.5);axs[1].legend(fontsize=7);axs[1].set_xlabel('Teste (%)');axs[1].set_title('Resultado final, 10 bases');save('adapt')
r=[]
for (m,t),z in A.groupby(['method','tag']):
 v=z.set_index('dataset').test-ng.xs(5,level='shot');r.append(v.rename(m+(' .35/.75' if t=='adapt' else ' .45/.85')))
h=pd.concat(r,axis=1);h.index=h.index.map(names);heat(h,'adaptbase','Adaptativo menos sem grafo: o empate médio esconde trocas',v=2,height=4.8)
# hardware
P=pd.read_csv(R/'pares.csv');fig,axs=plt.subplots(1,2,figsize=(9,3.7));axs[0].scatter(P.test_accuracy,P.test,s=17,c=C[0],alpha=.7);axs[0].plot([20,100],[20,100],color='gray',ls='--');axs[0].set_xlabel('Histórico GPU (%)');axs[0].set_ylabel('Claude CPU (%)');axs[0].set_title('80 configurações correspondentes')
P['label']=P.method.map(methods);axs[1].boxplot([P.loc[P.method==m,'delta_test'] for m in ['no-graph','union','dgcg','dgcg24']],tick_labels=[methods[m] for m in ['no-graph','union','dgcg','dgcg24']],vert=False,showmeans=True);axs[1].axvline(0,color='gray',lw=.7);axs[1].set_xlabel('Claude − histórico (pp)');axs[1].set_title('Diferenças por método');save('hardware')
# scales
M=pd.read_csv(R/'medias_comparacao.csv');fig,axs=plt.subplots(2,2,figsize=(9,6.2),sharex=True)
for ax,(base,s) in zip(axs.flat,[(x,s) for x in ['flowers','cars'] for s in [1,5]]):
 for m,col in zip(['no-graph','knn-union','dgcg'],C):
  z=M[(M.n_shot==s)&M.dataset.str.startswith(base+'_')&(M.metodo==m)].set_index('dataset').loc[[base+'_'+v for v in ['micro','mini','extended']]];ax.errorbar(range(3),z.test_accuracy_mean,yerr=z.test_accuracy_std_seed,marker='o',capsize=4,label=m,color=col)
 ax.set_xticks(range(3),['Micro','Mini','Extended']);ax.set_title(f'{base.title()} · {s}-shot');ax.set_ylabel('Teste (%) ± desvio entre seeds');ax.grid(alpha=.15);ax.legend(fontsize=7)
save('scales')
# extras matched deltas
E=pd.read_csv(R/'medias_buscas_extras.csv');z=E[E.extra_phase.eq('all')&E.extras_per_class.gt(0)].copy();z['label']=z.metodo+' / '+z.alvo.astype(str)+' / '+z.extra_mode
fig,ax=plt.subplots(figsize=(9,3.6));p=z.groupby(['label','extras_per_class']).ganho_accuracy_vs_sem_extras_pp.mean().unstack();p.T.plot(ax=ax,marker='o');ax.axhline(0,color='gray',lw=.8);ax.set_xlabel('Extras por classe');ax.set_ylabel('Ganho médio sobre controle sem extras (pp)');ax.legend(fontsize=6.5,ncol=2);ax.grid(alpha=.15);save('extras')
# table exports
for key,d in D.items():
 if 'test' in d:
  keys=[k for k in ['method','shot','feat','trainfree','tag'] if k in d];d.groupby(keys,dropna=False)[['val','test','f1']].mean().reset_index().to_csv(R.parent/(key+'_medias.csv'),index=False)
summary={'counts':{k:len(v) for k,v in D.items()},'hom_corr':corr,'hardware_mean_abs':P.delta_test.abs().mean(),'hardware_max':P.groupby('method').delta_test.apply(lambda v:v.abs().max()).to_dict(),'topo':rt.to_dict('records'),'scales':M.to_dict('records')}
(R.parent/'estatisticas.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
(R.parent/'sha256.txt').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  fontes/'+p.name for p in sorted(R.iterdir()))+'\n')
print(json.dumps(summary['hom_corr']));print(z.label.unique());print(E.extra_phase.unique());print('Charts ready',len(list(O.glob('*.png'))))
