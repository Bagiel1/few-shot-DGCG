"""Atualiza o PDF a partir dos seis CSVs recebidos, preservando a versao antiga.

Executar com o Python bundled (ReportLab/pandas). Matplotlib, ausente nesse
runtime, e usado apenas no subprocesso --charts-only com o ambiente do projeto.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf/relatorio_fewshot_2026-10-26.pdf'
WORK=ROOT/'outputs/analise_experimentos_20261026'
HISTORY=ROOT/'outputs/analise_experimentos_20261012'
FILES=['resultados_metricas.csv','melhores_graus_por_protocolo_metricas.csv',
       'variacao_intervalos_metricas.csv','resultados_densidades_protocolos_dgcg.csv',
       'melhores_densidades_por_protocolo_dgcg.csv','variacao_densidades_dgcg.csv']
KEY=['dataset','n_way','n_shot','n_query','seed']
BASE_METHODS=['knn-in','knn-out','knn-mutual','knn-union','no-graph','dgcg']
NAMES={'bacteria_micro':'Bacteria','birds_micro':'Birds','cars_micro':'Cars',
       'flowers_micro':'Flowers','manufacturing_micro':'Manufacturing','plankton_micro':'Plankton',
       'plant_village_micro':'Plant Village','resisc_micro':'RESISC','sports_micro':'Sports','textures_micro':'Textures'}
LABEL={'knn-in':'kNN-in','knn-out':'kNN-out','knn-mutual':'kNN mutual/reciprocal',
       'knn-union':'kNN-union','no-graph':'No-graph (ProtoNet)','dgcg':'DGCG 4-6',
       'grau escolhido':'DGCG grau escolhido','rho escolhida':'DGCG densidade escolhida'}
GROUP_LABELS=[LABEL[m] for m in BASE_METHODS]+[LABEL['grau escolhido'],LABEL['rho escolhida']]
DEGREES=['2:4','4:6','6:8','8:12']
RHO=[.025,.05,.075,.1,.125,.15,.175,.2,.25,.3,.4,.5,.6,.75,.9]


def n(value,digits=2):
    if pd.isna(value):return '-'
    numeric=float(value)
    if abs(numeric)<.5*10**(-digits):numeric=0.
    return f'{numeric:.{digits}f}'.replace('.',',')


def load_data(source):
    M,G,VG,D,S,VS=[pd.read_csv(source/f) for f in FILES]
    assert [len(x) for x in [M,G,VG,D,S,VS]]==[2430,270,270,4050,270,270]
    assert set(M.dataset)==set(D.dataset)==set(NAMES)
    assert set(M.metodo)==set(BASE_METHODS)
    expected=list(itertools.product(sorted(NAMES),[2,3,5],[1,5,10],[1,5,10],[7]))
    for df in [G,S]:
        assert not df[KEY].duplicated().any()
        assert set(df[KEY].itertuples(index=False,name=None))==set(expected)
    assert not M[KEY+['metodo','alvo']].duplicated().any()
    assert not D[KEY+['alvo']].duplicated().any()
    assert not D.isna().any().any()
    for m in BASE_METHODS:
        group=M.loc[M.metodo.eq(m)]
        assert len(group)==(1080 if m=='dgcg' else 270)
        assert set(group[KEY].itertuples(index=False,name=None))==set(expected)
        if m=='dgcg':
            assert set(group.alvo)==set(DEGREES)
            assert group.groupby(KEY).size().eq(4).all()
    assert set(D.alvo)==set(RHO) and D.groupby(KEY).size().eq(15).all()
    assert D.seed.eq(7).all() and M.seed.eq(7).all()
    assert D.baseline_status.eq('compativel').all()
    for df in [M,D]:
        assert np.allclose(df.episode_nodes,df.n_way*(df.n_shot+df.n_query))
        assert df.val_episodes.eq(200).all() and df.test_episodes.eq(1000).all()
        for split in ['val','test']:
            for metric in ['accuracy','macro_f1','balanced_accuracy']:
                assert df[f'{split}_{metric}'].between(0,100).all()
            assert np.allclose(df[f'{split}_accuracy'],df[f'{split}_balanced_accuracy'],atol=1e-5)
            assert np.allclose(df[f'{split}_degree_mean']/(df.episode_nodes-1),df[f'{split}_density_mean'])
    # Escolhas recalculadas sem usar teste, inclusive regra de desempate.
    new_g=[];new_s=[]
    for _,group in M.loc[M.metodo.eq('dgcg')].groupby(KEY):
        ties=group.loc[group.val_accuracy.eq(group.val_accuracy.max())]
        winner=ties.loc[ties.alvo.eq('4:6')]
        new_g.append((winner.iloc[0] if len(winner) else ties.sort_values('alvo').iloc[0]))
    for _,group in D.groupby(KEY):
        new_s.append(group.sort_values(['val_accuracy','alvo'],ascending=[False,True]).iloc[0])
    for received,recomputed in [(G,pd.DataFrame(new_g)),(S,pd.DataFrame(new_s))]:
        merged=received.merge(recomputed,on=KEY,suffixes=('_csv','_calc'),validate='one_to_one')
        assert merged.alvo_csv.astype(str).eq(merged.alvo_calc.astype(str)).all()
        for col in ['val_accuracy','test_accuracy','test_macro_f1','test_loss']:
            assert np.allclose(merged[col+'_csv'],merged[col+'_calc'])
    for chosen,raw in [(G,M.loc[M.metodo.eq('dgcg')]),(S,D)]:
        for _,r in chosen.iterrows():
            group=raw.loc[(raw[KEY]==r[KEY]).all(axis=1)]
            candidates=group.loc[group.val_accuracy.eq(group.val_accuracy.max()),'alvo']
            if chosen is S:
                assert set(map(float,r.alvos_empatados.split(';')))==set(candidates)
            else:
                assert set(r.alvos_empatados.split(';'))==set(candidates)
    for chosen,variation,is_density in [(G,VG,False),(S,VS,True)]:
        for _,r in variation.iterrows():
            axis=r.eixo_variado;fixed=[k for k in KEY if k!=axis]
            group=chosen.loc[(chosen[fixed]==r[fixed]).all(axis=1)]
            assert len(group)==3
            assert group.alvo.nunique()==r['densidades_distintas' if is_density else 'intervalos_distintos']
    B=M.loc[~M.metodo.eq('dgcg')|M.alvo.eq('4:6')].copy()
    comparison=D.merge(B.loc[B.metodo.eq('dgcg'),KEY+['test_accuracy','val_accuracy']],on=KEY,
                       suffixes=('','_baseline'),validate='many_to_one')
    assert np.allclose(comparison.test_accuracy_baseline,comparison.baseline_test_accuracy)
    assert np.allclose(comparison.val_accuracy_baseline,comparison.baseline_val_accuracy)
    assert np.allclose(D.ganho_teste_vs_4_6_pp,D.test_accuracy-D.baseline_test_accuracy)
    C=pd.concat([B.assign(exibicao=B.metodo.map(LABEL)),
        G.assign(exibicao=LABEL['grau escolhido']),S.assign(exibicao=LABEL['rho escolhida'])],ignore_index=True)
    return M,G,VG,D,S,VS,B,C


def charts(data):
    os.environ.setdefault('MPLCONFIGDIR',str(WORK/'_matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    M,G,VG,D,S,VS,B,C=data
    folder=WORK/'graficos';folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    specs=[]
    def save(fig,name,title,caption):
        fig.tight_layout(pad=1.2)
        fig.savefig(folder/(name+'.png'),dpi=175,bbox_inches='tight')
        plt.close(fig);specs.append(dict(name=name,title=title,caption=caption))
    colors=['#3976a4']*6+['#8256a6','#2c987b']
    fig,axes=plt.subplots(2,2,figsize=(13.5,8))
    for row,metric in enumerate(['test_accuracy','test_macro_f1']):
        for col,threshold in enumerate([0,6]):
            ax=axes[row,col];means=C.loc[C.episode_nodes>threshold].groupby('exibicao')[metric].mean().reindex(GROUP_LABELS)
            ax.barh(range(8),means,color=colors)
            ax.set_yticks(range(8),GROUP_LABELS,fontsize=9);ax.invert_yaxis()
            ax.set(xlim=(0,100),xlabel='Teste (%)',title=('Acuracia' if row==0 else 'F1 macro')+(' - grade completa' if col==0 else ' - N > 6'))
            for i,value in enumerate(means):ax.text(value+.4,i,f'{value:.2f}',va='center',fontsize=8)
    save(fig,'01_medias_novas','Metodos - acuracia e F1 macro','Dez bases. Escolhas de grau/densidade feitas pela validacao em cada base/protocolo.')
    for metric,title in [('test_accuracy','Acuracia'),('test_macro_f1','F1 macro')]:
        matrix=B.pivot_table(index='dataset',columns='metodo',values=metric,aggfunc='mean').reindex(index=sorted(NAMES),columns=BASE_METHODS)
        fig,ax=plt.subplots(figsize=(12,6.8));im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto')
        ax.set_yticks(range(10),[NAMES[x] for x in matrix.index]);ax.set_xticks(range(6),[LABEL[x] for x in matrix.columns],rotation=15,ha='right')
        for (i,j),value in np.ndenumerate(matrix.to_numpy()):ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white' if value<65 else 'black')
        fig.colorbar(im,ax=ax,label='Teste (%)')
        save(fig,metric+'_bases',title+' por base - seis metodos','Media dos mesmos 27 protocolos; DGCG com alvo 4-6. Sem escolha de shot/query pelo teste.')
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.5),sharey=True)
    for ax,way in zip(axes,[2,3,5]):
        group=M.loc[M.metodo.eq('dgcg')&M.n_way.eq(way)].groupby('alvo')
        for metric,label in [('val_accuracy','Validacao'),('test_accuracy','Teste descritivo')]:
            ax.plot(range(4),group[metric].mean().reindex(DEGREES),marker='o',label=label)
        ax.set(xticks=range(4),xticklabels=DEGREES,title=f'{way}-way',xlabel='Intervalo de grau-alvo',ylabel='Acuracia (%)');ax.legend(fontsize=9)
    save(fig,'04_grau_por_way','Grau-alvo - resposta por n-way','Dez bases; medias balanceadas em shot/query. Intervalos aplicados no treino, validacao e teste.')
    for cut in [0,6]:
        fig,axes=plt.subplots(1,3,figsize=(13.8,4.5),sharey=True)
        for ax,way in zip(axes,[2,3,5]):
            group=D.loc[D.n_way.eq(way)&D.episode_nodes.gt(cut)].groupby('alvo')
            for metric,label in [('val_accuracy','Validacao'),('test_accuracy','Teste descritivo')]:
                ax.plot(RHO,group[metric].mean().reindex(RHO),marker='o',markersize=3,label=label)
            base=B.loc[B.metodo.eq('dgcg')&B.n_way.eq(way)&B.episode_nodes.gt(cut),'test_accuracy'].mean()
            ax.axhline(base,ls=':',color='gray',label='4-6: teste')
            ax.set(title=f'{way}-way',xlabel='Densidade-alvo',ylabel='Acuracia (%)');ax.legend(fontsize=8)
        save(fig,f'05_densidade_corte{cut}','Densidade - resposta por n-way'+(' (N > 6)' if cut else ''),
             'Dez bases; mesmas combinacoes por alvo. Teste descritivo, nao usado para escolher a densidade.')
    fig,axes=plt.subplots(2,2,figsize=(12,7.5))
    for ax,nodes in zip(axes.flat,[4,6,30,100]):
        group=D.loc[D.episode_nodes.eq(nodes)].groupby('alvo').test_density_mean.mean()
        ax.plot(RHO,group.reindex(RHO),marker='o',markersize=4,label='Densidade realizada')
        ax.plot([0,1],[0,1],ls='--',color='gray',label='Alvo = realizado')
        ax.set(title=f'{nodes} nos',xlabel='Densidade-alvo',ylabel='Densidade de teste',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
    save(fig,'06_alvo_real','Densidade-alvo versus densidade realizada','Media por tamanho do episodio. Empates e fallback anti-isolamento afastam o resultado do alvo.')
    fig,ax=plt.subplots(figsize=(12,6))
    selected=S.copy()
    for i,(cut,label) in enumerate([(0,'Grade completa'),(6,'N > 6')]):
        vals=selected.loc[selected.episode_nodes.gt(cut)].groupby('dataset').ganho_teste_vs_4_6_pp.mean().reindex(sorted(NAMES))
        ax.bar(np.arange(10)+(i-.5)*.36,vals,width=.36,label=label)
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(10),[NAMES[x] for x in sorted(NAMES)],rotation=25,ha='right')
    ax.set(ylabel='Ganho medio no teste (pp)');ax.legend()
    save(fig,'07_ganhos_base','Densidade escolhida versus DGCG 4-6','Escolha local pela validacao; compara os mesmos protocolos. As duas barras usam conjuntos de tamanhos diferentes.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,chosen,is_rho in zip(axes,[G,S],[False,True]):
        vals=chosen.alvo.value_counts().sort_index()
        ax.bar(range(len(vals)),vals.values)
        ax.set_xticks(range(len(vals)),[str(x) for x in vals.index],rotation=45,ha='right')
        ax.set(title='Densidades' if is_rho else 'Intervalos de grau',ylabel='Protocolos escolhidos (n=270)',xlabel='Alvo escolhido pela validacao')
    save(fig,'08_frequencia','Frequencia das escolhas na validacao','Empates: prefere 4-6 entre graus; menor rho entre densidades. Frequencia nao significa otimo unico.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    dg=M.loc[M.metodo.eq('dgcg')&M.alvo.eq('4:6')]
    stats=dg.groupby('episode_nodes').agg(grau=('test_degree_mean','mean'),rho=('test_density_mean','mean'))
    axes[0].plot(stats.index,stats.grau,marker='o');axes[0].axhspan(4,6,alpha=.15,color='green')
    axes[0].set(xlabel='Nos do episodio',ylabel='Grau medio realizado',title='Alvo fixo 4-6')
    axes[1].plot(stats.index,stats.rho,marker='o')
    axes[1].set(xlabel='Nos do episodio',ylabel='Densidade realizada',title='Mesmo grau-alvo, densidade diferente')
    save(fig,'09_grau_real','DGCG 4-6 - grau e densidade realizados','Topologia antes de autolacos e normalizacao. Medias sobre bases/protocolos com o mesmo tamanho.')
    for dataset in sorted(NAMES):
        fig,axes=plt.subplots(3,2,figsize=(11,7.5))
        for i,way in enumerate([2,3,5]):
            for j,chosen in enumerate([G,S]):
                ax=axes[i,j]
                group=chosen.loc[chosen.dataset.eq(dataset)&chosen.n_way.eq(way)]
                matrix=np.zeros((3,3))
                texts=[['']*3 for _ in range(3)]
                for _,r in group.iterrows():
                    si=[1,5,10].index(r.n_shot);qi=[1,5,10].index(r.n_query)
                    matrix[si,qi]=float(r.alvo) if j else np.mean([float(v) for v in r.alvo.split(':')])
                    texts[si][qi]=f'{float(r.alvo):g}' if j else r.alvo
                ax.imshow(matrix,vmin=0 if j else 2,vmax=.5 if j else 10,cmap='viridis',aspect='auto')
                for si in range(3):
                    for qi in range(3):ax.text(qi,si,texts[si][qi],ha='center',va='center',color='white',fontsize=11,
                                             bbox=dict(facecolor='black',alpha=.35,edgecolor='none'))
                ax.set(xticks=range(3),xticklabels=[1,5,10],yticks=range(3),yticklabels=[1,5,10],xlabel='n-query',ylabel='n-shot',
                       title=f'{way}-way: '+('densidade' if j else 'grau'))
        save(fig,'base_escolhas_'+dataset,NAMES[dataset]+' - escolhas por protocolo',
             'Maior validacao; empates resolvidos por regra fixa. Valores das celulas sao alvos, nao graus/densidades realizados.')
    # Dez paineis historicos redesenhados na vertical, com os mesmos 108 valores por base.
    old=pd.read_csv(HISTORY/'tabelas/resultados_unificados_1080.csv')
    for dataset in sorted(NAMES):
        fig,axes=plt.subplots(3,1,figsize=(11.5,7))
        for ax,way in zip(axes,[2,3,5]):
            cols=[(s,q) for s in [1,5,10] for q in [1,5,10]]
            methods=['knn-in','knn-out','knn-reciprocal','dgcg']
            matrix=np.array([[old.loc[old.dataset.eq(dataset)&old.n_way.eq(way)&old.metodo.eq(m)&old.n_shot.eq(s)&old.n_query.eq(q),'test_accuracy'].iloc[0]
                              for s,q in cols] for m in methods])
            ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto')
            ax.set_xticks(range(9),[f'{s}s/{q}q' for s,q in cols],fontsize=9)
            ax.set_yticks(range(4),['kNN-in','kNN-out','kNN reciprocal','DGCG 4-6'],fontsize=9)
            ax.set_title(f'{way}-way',loc='left',fontsize=11)
            for (i,j),value in np.ndenumerate(matrix):
                ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=9,color='white' if value<65 else 'black')
        name='historico_base_'+dataset
        fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=175,bbox_inches='tight');plt.close(fig)
    (WORK/'graficos_novos.json').write_text(json.dumps(specs,indent=2,ensure_ascii=False))


def build_pdf(data):
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4,landscape
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Table,TableStyle,Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from PIL import Image
    from pypdf import PdfReader
    M,G,VG,D,S,VS,B,C=data
    pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    W,H=landscape(A4);margin=34;cw=W-2*margin
    pdf=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
    pdf.setTitle('Few-shot DGCG - resultados atualizados em 26/10/2026')
    pdf.setAuthor('Gabriel Maia Brito')
    pages=[]
    def start(title,subtitle='',section='Resultados atuais'):
        if pages:pdf.showPage()
        page=len(pages)+1;pages.append(dict(page=page,title=title,section=section))
        pdf.bookmarkPage('p'+str(page));pdf.addOutlineEntry(title,'p'+str(page),level=0,closed=False)
        pdf.setFillColor(colors.black);pdf.setFont('DV',8)
        pdf.drawString(margin,H-22,f'IC | Few-shot DGCG | Atualizado em 26/10/2026 | {section}')
        size=15
        assert pdfmetrics.stringWidth(title,'DVB',size)<=cw,title
        pdf.setFont('DVB',size);pdf.drawString(margin,H-48,title)
        if subtitle:
            assert pdfmetrics.stringWidth(subtitle,'DV',8)<=cw,subtitle
            pdf.setFont('DV',8);pdf.drawString(margin,H-66,subtitle)
        pdf.setStrokeColor(colors.HexColor('#bfc4c8'));pdf.line(margin,37,W-margin,37)
        pdf.setFont('DV',8);pdf.setFillColor(colors.HexColor('#555555'))
        pdf.drawString(margin,24,'Acuracia/F1 em %. IC95: episodios. pp: pontos percentuais. Uma seed (7).')
        pdf.drawRightString(W-margin,24,str(page));pdf.setFillColor(colors.black)
        return H-84
    def lines(y,items,size=8.4):
        pdf.setFillColor(colors.black);pdf.setFont('DV',size)
        for item in items:
            assert pdfmetrics.stringWidth(item,'DV',size)<cw,item
            assert y>=45,(len(pages),item)
            pdf.drawString(margin,y,item);y-=14
        return y-5
    def heading(y,title):
        pdf.setFont('DVB',10);pdf.drawString(margin,y,title);return y-15
    def table(y,headers,rows,widths=None,size=8.5,pad=2):
        widths=widths or [cw/len(headers)]*len(headers)
        assert abs(sum(widths)-cw)<1
        plain=ParagraphStyle('cell',fontName='DV',fontSize=size,leading=size+1.4)
        bold=ParagraphStyle('head',fontName='DVB',fontSize=size,leading=size+1.4)
        t=Table([[Paragraph(str(x),bold) for x in headers]]+
                [[Paragraph(str(x),plain) for x in row] for row in rows],colWidths=widths)
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),
            ('LINEBELOW',(0,0),(-1,0),.6,colors.black),
            ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#cccccc')),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad)]))
        _,height=t.wrap(cw,H)
        assert y-height>=44,(len(pages),y-height)
        t.drawOn(pdf,margin,y-height)
        return y-height-15
    def figure(path,title,caption,section='Resultados atuais'):
        start(title,caption,section)
        width,height=Image.open(path).size
        scale=min(cw/width,(H-145)/height)
        dw,dh=width*scale,height*scale
        pdf.drawImage(str(path),margin+(cw-dw)/2,65+(H-145-dh)/2,width=dw,height=dh)
    def method_rows(frame):
        agg=frame.groupby('exibicao')[['test_accuracy','test_macro_f1','test_balanced_accuracy','test_loss']].mean().reindex(GROUP_LABELS)
        return [[idx]+[n(v,4 if col=='test_loss' else 2) for col,v in row.items()] for idx,row in agg.iterrows()]
    def fixed(frame):return frame.loc[frame.n_way.eq(5)&frame.n_shot.eq(1)&frame.n_query.eq(5)]
    y=start('Few-shot DGCG - resultados consolidados','6.480 execucoes novas: 2.430 metodos/graus + 4.050 densidades | dez bases Micro')
    y=lines(y,[
        'Way: 2/3/5; shot: 1/5/10; query: 1/5/10. Split em classes: 10 treino / 5 validacao / 5 teste.',
        'ResNet-18 ImageNet congelada; SGC com 2 saltos; kNN k=5 euclidiano; GRaNDe desligado.',
        'DGCG: euclidiano, RBO p=0,9, L=200, top-k=40. Baterias novas sem referencias externas ao episodio.',
        '200 episodios de validacao e 1.000 de teste por execucao nos CSVs. Checkpoints escolhidos pela validacao.',
    ])
    y=heading(y,'Teste - media dos 27 protocolos por base (270 resultados por linha)')
    y=table(y,['Metodo / regra','Acuracia','F1 macro','Acuracia balanceada','Loss'],method_rows(C),[255,115,115,165,cw-650],size=8.2,pad=1.5)
    y=heading(y,'Teste - excluindo episodios com ate seis nos (250 resultados por linha)')
    y=table(y,['Metodo / regra','Acuracia','F1 macro','Acuracia balanceada','Loss'],method_rows(C.loc[C.episode_nodes>6]),[255,115,115,165,cw-650],size=8.2,pad=1.5)
    lines(y,['Grau/densidade escolhidos: ajuste por base/protocolo na validacao, nao um unico hiperparametro global.'],size=8)

    y=start('Auditoria e comparacao por n-way','Mesmo conjunto de bases/protocolos por metodo; nao agrega configuracoes faltantes')
    y=table(y,['Bateria','Linhas unicas','Bases','Protocolos/base','Alvos DGCG'],[
        ['Metodos e graus','2.430','10','27','4 intervalos'],['Densidades','4.050','10','27','15 densidades'],
        ['Resumos de escolhas','270 + 270','10','27','Recalculados e reconciliados'],
    ],[245,140,100,140,cw-625])
    y=lines(y,[
        'Nenhuma combinacao planejada ausente ou duplicada. Os campos vazios de grau nas baselines kNN/no-graph sao nao aplicaveis.',
        'Baselines 4-6 citadas nos 4.050 resultados conferem com os CSVs dos 2.430 testes.',
        'F1 macro e recall macro calculados por episodio; depois, media dos episodios. Nao mistura rotulos locais entre episodios.',
        'Com consultas balanceadas por classe, acuracia balanceada coincide com acuracia comum (salvo arredondamento).',
    ],size=8)
    y=heading(y,'Medias por n-way - acuracia e F1 macro de teste')
    rows=[]
    for label in GROUP_LABELS:
        group=C.loc[C.exibicao.eq(label)]
        rows.append([label]+[n(group.loc[group.n_way.eq(w),metric].mean())
                    for metric in ['test_accuracy','test_macro_f1'] for w in [2,3,5]])
    table(y,['Metodo','Acc 2-way','Acc 3-way','Acc 5-way','F1 2-way','F1 3-way','F1 5-way'],rows,[230]+[(cw-230)/6]*6,size=8.2)

    for metric,title in [('test_accuracy','Acuracia'),('test_macro_f1','F1 macro')]:
        y=start(title+' por base - metodos e ajustes','Media dos 27 protocolos; escolhas de grau/densidade feitas apenas pela validacao')
        matrix=C.pivot_table(index='dataset',columns='exibicao',values=metric,aggfunc='mean').reindex(index=sorted(NAMES),columns=GROUP_LABELS)
        headers=['Base','kNN-in','kNN-out','Mutual','Union','No-graph','DGCG 4-6','Grau escolhido','Rho escolhida']
        y=table(y,headers,[[NAMES[idx]]+[n(v) for v in row] for idx,row in matrix.iterrows()],
                [145]+[(cw-145)/8]*8,size=8,pad=4)
        lines(y,['Mutual = reciprocal (intersecao das escolhas). Union liga se pelo menos um no escolher o outro.',
                 'No-graph: mesma projecao e classificador por prototipos, sem propagacao entre imagens.'],size=8)

    y=start('DGCG - intervalos de grau e variacao','Quatro intervalos aplicados em cada uma das 270 combinacoes de base/protocolo')
    rows=[]
    for target,group in M.loc[M.metodo.eq('dgcg')].groupby('alvo'):
        rows.append([target,n(group.val_accuracy.mean()),n(group.test_accuracy.mean()),n(group.test_macro_f1.mean()),
                     n(group.test_degree_mean.mean(),3),n(group.test_density_mean.mean(),3)])
    y=table(y,['Intervalo-alvo','Val. media','Teste medio','F1 teste','Grau realizado','Densidade realizada'],rows)
    y=heading(y,'Variacao da escolha ao mudar um eixo, mantendo os demais fixos')
    rows=[]
    for axis,group in VG.groupby('eixo_variado'):
        rows.append([axis,len(group),int(group.amplitude_centro.gt(1e-10).sum()),n(group.amplitude_centro.mean(),3),n(group.amplitude_centro.max(),3)])
    y=table(y,['Eixo','Grupos','Escolha mudou','Amplitude media do centro','Maior amplitude'],rows)
    tied=int(G.alvos_empatados.str.contains(';').sum())
    y=lines(y,[f'Empates de validacao em {tied}/270 protocolos; prefere-se 4-6 quando empatado, depois o menor intervalo.',
        'Intervalos solicitados podem ser limitados pelo numero de candidatos. A topologia tambem depende de empates/fallback.',
        'Variacao da escolha nao demonstra, por si so, diferenca estatistica entre intervalos.'],size=8.2)
    y=heading(y,'DGCG 4-6 - fracao de episodios de teste cuja media de grau ficou fora do alvo efetivo')
    dg=M.loc[M.metodo.eq('dgcg')&M.alvo.eq('4:6')]
    y=table(y,['n-way','Fracao media fora do alvo','Grau medio realizado','Densidade media'],[
        [w,n(g.test_episode_mean_outside_target_fraction.mean(),4),n(g.test_degree_mean.mean(),3),n(g.test_density_mean.mean(),3)]
        for w,g in dg.groupby('n_way')],size=8.2)
    y=heading(y,'Densidade - variacao da escolha ao mudar um eixo')
    table(y,['Eixo','Grupos','Escolha mudou','Amplitude media de rho','Maior amplitude'],[
        [axis,len(g),int(g.amplitude_densidade.gt(1e-10).sum()),n(g.amplitude_densidade.mean(),3),n(g.amplitude_densidade.max(),3)]
        for axis,g in VS.groupby('eixo_variado')],size=8.2,pad=1.5)

    y=start('DGCG - ranking das densidades','Media dos mesmos 270 protocolos/base; ordenacao por validacao, nunca por teste')
    rank=D.groupby('alvo').agg(val=('val_accuracy','mean'),test=('test_accuracy','mean'),f1=('test_macro_f1','mean'),
                               rho=('test_density_mean','mean'),gain=('ganho_teste_vs_4_6_pp','mean')).sort_values('val',ascending=False)
    y=table(y,['Rho-alvo','Validacao','Teste','F1 teste','Rho realizada','Ganho vs 4-6 (pp)'],[
        [n(idx,3),n(r.val),n(r.test),n(r.f1),n(r.rho,4),n(r.gain)] for idx,r in rank.iterrows()],size=8.5,pad=3)
    y=lines(y,[
        'Rho=0,05 lidera a validacao media da grade inteira; a diferenca para 0,075 e pequena.',
        'No protocolo fixo 5-way/1-shot/5-query, o ranking global pode ser diferente.',
        'Sao alvos de densidade, nao densidades realizadas garantidas. Nao existe um otimo universal demonstrado.',
    ],size=8.3)

    y=start('Sensibilidade ao tamanho do episodio','Escolhas por base/protocolo; mesmas configuracoes na baseline e nas duas regras de ajuste')
    filters=[('Grade completa',lambda x:x),('N > 6',lambda x:x.loc[x.episode_nodes>6]),
             ('N > 10',lambda x:x.loc[x.episode_nodes>10]),('5-way/1-shot/5-query',fixed)]
    rows=[]
    for label,fn in filters:
        b=fn(B.loc[B.metodo.eq('dgcg')]);s=fn(S);g=fn(G);ng=fn(B.loc[B.metodo.eq('no-graph')])
        rows.append([label,len(s),n(ng.test_accuracy.mean()),n(b.test_accuracy.mean()),n(g.test_accuracy.mean()),
                     n(s.test_accuracy.mean()),n(s.test_accuracy.mean()-b.test_accuracy.mean())])
    y=table(y,['Filtro','n','No-graph','DGCG 4-6','Grau escolhido','Rho escolhida','Ganho rho (pp)'],rows,
            [190,35,95,100,115,115,cw-650],size=8.2)
    y=heading(y,'Ganhos por base, excluindo episodios com ate seis nos')
    rows=[]
    for ds in sorted(NAMES):
        b=B.loc[B.metodo.eq('dgcg')&B.dataset.eq(ds)&B.episode_nodes.gt(6)]
        g=G.loc[G.dataset.eq(ds)&G.episode_nodes.gt(6)];s=S.loc[S.dataset.eq(ds)&S.episode_nodes.gt(6)]
        rows.append([NAMES[ds],n(b.test_accuracy.mean()),n(g.test_accuracy.mean()),n(s.test_accuracy.mean()),
                     n(g.test_accuracy.mean()-b.test_accuracy.mean()),n(s.test_accuracy.mean()-b.test_accuracy.mean())])
    y=table(y,['Base','DGCG 4-6','Grau escolhido','Rho escolhida','Ganho grau (pp)','Ganho rho (pp)'],rows,
            [145]+[(cw-145)/5]*5,size=8,pad=2)
    lines(y,['Os cortes de tamanho sao diagnosticos; os resultados completos permanecem apresentados.',
             'Nos 270 protocolos, a escolha de rho melhorou 159, empatou 2 e piorou 109 (tolerancia 1e-6 pp).'],size=8)

    y=start('Protocolo fixo - 5-way / 1-shot / 5-query','Dez bases; 30 imagens por episodio. Teste dos checkpoints escolhidos na validacao.')
    matrix=fixed(B).pivot(index='dataset',columns='metodo',values='test_accuracy').reindex(index=sorted(NAMES),columns=BASE_METHODS)
    y=table(y,['Base','kNN-in','kNN-out','Mutual','Union','No-graph','DGCG 4-6'],
            [[NAMES[idx]]+[n(v) for v in row] for idx,row in matrix.iterrows()],
            [145]+[(cw-145)/6]*6,size=8,pad=2)
    y=heading(y,'Ajustes escolhidos na validacao - teste e F1 macro')
    rows=[]
    for ds in sorted(NAMES):
        g=fixed(G).loc[lambda x:x.dataset.eq(ds)].iloc[0];s=fixed(S).loc[lambda x:x.dataset.eq(ds)].iloc[0]
        rows.append([NAMES[ds],g.alvo,n(g.test_accuracy),n(s.alvo,3),n(s.test_accuracy)+' +/- '+n(s.test_ci95),n(s.test_macro_f1),n(s.ganho_teste_vs_4_6_pp)])
    y=table(y,['Base','Grau','Teste grau','Rho','Teste rho +/- IC95','F1 rho','Ganho rho (pp)'],rows,
            [145,65,85,65,180,110,cw-650],size=8,pad=2)
    lines(y,['O IC95 acima mede variacao entre episodios de um treinamento, nao repeticoes de treinamento.'],size=8)

    y=start('Flowers Micro - referencias e contextos separados','Resultados anteriores e valores informados por Gabriel; nao misturar com as baterias episodicas novas')
    y=heading(y,'Contextos de rankings - 5-way/1-shot/5-query')
    y=table(y,['Rankings no treino','Rankings na avaliacao','Teste','IC95','Loss','Origem'],[
        ['Episodicos','Episodicos','83,69','0,62','0,4970','CSV historico'],
        ['Todas as imagens de treino','Episodicos, sem banco','83,43','0,63','0,5299','Execucao local anterior'],
        ['Todas as imagens de treino','Treino + episodio','75,08','-','-','Informado por Gabriel'],
        ['Todas as imagens de treino','Toda a particao de teste','89,26','0,52','0,3778','Checkpoint fixo; execucao local'],
    ],[160,205,65,65,75,cw-570],size=8.2,pad=3)
    y=heading(y,'Banco reservado - total de extras por episodio, nao por classe')
    y=table(y,['Extras','Extras por classe (5-way)','Teste','Ganho vs N=0 (pp)','Origem'],[
        ['0','0','82,66','0,00','Informado por Gabriel'],['30','6','86,71','4,05','Informado por Gabriel'],
        ['50','10','87,58','4,92','Informado por Gabriel'],['100','20','86,94','4,28','Informado por Gabriel'],
    ],[80,195,95,190,cw-560],size=8.5,pad=3)
    lines(y,[
        'N=0 com banco reservado (82,66%) nao e o antigo sem banco (83,43%): o pool de episodios muda.',
        'Banco: 20 imagens por classe reservadas; suporte/consultas sorteados das outras 20. Extras na validacao e no teste.',
        'Extras afetam ranked lists/correlacao; nao sao nos do grafo. O amostrador garante classes do episodio usando rotulos.',
        'Referencias externas ao episodio ampliam o protocolo. Nao apresentar 87,58% ou 89,26% como melhoria episodica equivalente.',
        'Para 82,66/86,71/87,58/86,94 e 75,08 nao foram fornecidos F1, loss e IC95; esses campos nao sao inferidos.',
    ],size=8)

    y=start('Densidade solicitada e realizada','Grau topologico: vizinhos nao nulos por linha da adjacencia, sem autolacos e antes da normalizacao')
    rows=[]
    for nodes,group in D.groupby('episode_nodes'):
        chosen=S.loc[S.episode_nodes.eq(nodes)]
        rows.append([nodes,len(group),n(group.test_desvio_densidade.abs().mean(),5),n(chosen.alvo.median(),4),
                     n(chosen.test_density_mean.median(),4),n(chosen.test_degree_mean.median(),3)])
    y=table(y,['Nos N','Execucoes','Erro abs. medio rho','Rho escolhida mediana','Rho real mediana','Grau real mediana'],rows,size=8.1,pad=2)
    lines(y,[
        'Erro absoluto: media de todas as densidades/bases/protocolos daquele tamanho. Medianas: somente configuracoes escolhidas.',
        'Com estes defaults, candidatos=N-1; grau-alvo=rho*(N-1). Empates e fallback podem impedir atingir o alvo.',
        'Bacteria com N=4: alvos 0,025 a 0,25 tiveram o mesmo teste (87,15%) e rho realizada=0,3333; grau medio=1.',
        'Ausencia de no completamente isolado nao significa grau por linha sempre positivo em grafos dirigidos.',
    ],size=8)

    specs=json.loads((WORK/'graficos_novos.json').read_text())
    for spec in specs:
        figure(WORK/'graficos'/(spec['name']+'.png'),spec['title'],spec['caption'])
    history_titles={
        '01_medias_metodos':'Medias dos metodos','02_metodos_por_base':'Metodos por base',
        '03_ganho_knnin':'Diferenca para kNN-in','04_protocolo_default':'Protocolo fixo',
        '05_selecao_validacao':'Selecao de shot/query','06_heatmap_way2':'Protocolos 2-way',
        '06_heatmap_way3':'Protocolos 3-way','06_heatmap_way5':'Protocolos 5-way',
        '07_efeito_n_shot':'Efeito de n-shot','07_efeito_n_query':'Efeito de n-query',
        '08_gap_val_teste':'Validacao menos teste','09_loss_validacao':'Loss e acuracia',
        '10_curvas_grau_densidade':'Grau e densidade - medias','11_dgcg_grau_bases':'Grau - validacao por base',
        '11_dgcg_densidade_bases':'Densidade - validacao por base','12_ganhos_ajuste':'Ajuste versus grau 4-6',
        '13_ajuste_vs_baselines':'DGCG ajustado e baselines','14_orcamento':'Imagens por episodio',
    }
    for filename,title in sorted(history_titles.items()):
        density_plot=filename.startswith(('10_','11_','12_','13_'))
        caption=('Varredura antiga: nove bases, 5-way/1-shot/5-query; Plant Village incompleto. Nao sao as 4.050 execucoes novas.'
                 if density_plot else 'Dados do HTML de 12/10/2026: quatro metodos, dez bases. Sem union/no-graph/F1; preservado para referencia.')
        figure(HISTORY/'graficos'/(filename+'.png'),'Historico - '+title,caption,'Historico 12/10/2026')
    for ds in sorted(NAMES):
        figure(WORK/'graficos'/('historico_base_'+ds+'.png'),'Historico - '+NAMES[ds]+' - protocolos',
               'Mesmos dados dos paineis do HTML de 12/10/2026; quatro metodos. Layout vertical para leitura.', 'Historico 12/10/2026')

    y=start('Fontes e limites da comparacao','Resultados cientificos observados; nao foram criados valores para metricas ausentes')
    y=heading(y,'Seis CSVs recebidos em 26/10/2026')
    for filename in FILES:y=lines(y,[filename],size=8.8)
    y=lines(y,[
        'Copias integrais e hashes SHA-256 preservados na pasta da analise; graficos e tabelas sao recalculados desses arquivos.',
        'Historico: 17 CSVs e 28 graficos do HTML de 12/10/2026. Valores globais/referencias extras: fontes indicadas na pagina propria.',
        'Acuracia, F1 macro e acuracia balanceada sao medias episodicas. Loss e entropia cruzada media.',
        'Medias de conjuntos diferentes (way, shot, query, bases ou acesso a referencias) nao representam a mesma tarefa.',
        'Escolhas de grau/densidade feitas pela validacao em cada base/protocolo. As medias adaptativas nao sao um unico modelo global.',
        'As buscas usam quatro ou 15 configuracoes por base/protocolo; cada baseline fixa usa uma configuracao.',
        'Uma seed (7). IC95 por episodio nao demonstra estabilidade entre seeds nem significancia de diferencas entre metodos.',
        'Nao foram fornecidas predicoes individuais, pares de erros ou replicacoes de treinamento para testes novos de significancia.',
        'Contagem de episodios de treino segue o comando default (2.000), mas nao e verificavel apenas pelos CSVs recebidos.',
        'Mutual e reciprocal sao o mesmo metodo; nao contam como baselines independentes.',
    ],size=8.2)
    y=heading(y,'Empates e regras')
    degree_ties=int(G.alvos_empatados.str.contains(';').sum());rho_ties=int(S.alvos_empatados.str.contains(';').sum())
    lines(y,[
        f'Graus: {degree_ties}/270 grupos com empates; densidades: {rho_ties}/270. Todos os alvos testados foram mantidos nos CSVs.',
        'Empates de validacao nao sao resolvidos pelo teste. Graus preferem 4-6; densidades preferem menor rho.',
        'Vencedor numerico em uma seed nao estabelece um grau ou densidade perfeito.',
    ],size=8.2)
    pdf.save()
    reader=PdfReader(OUT);assert len(reader.pages)==len(pages)
    texts=[page.extract_text() or '' for page in reader.pages]
    assert all(len(t)>80 for t in texts)
    assert all(value in texts[0] for value in ['6.480','2.430','4.050','F1 macro','No-graph'])
    full='\n'.join(texts)
    assert all(value in full for value in ['82,66','87,58','79,58','79,78','62,13','62,38','kNN-union'])
    assert len(history_titles)+len(NAMES)==28
    (WORK/'paginacao.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False))
    print(json.dumps(dict(pdf=str(OUT),paginas=len(pages),graficos_novos=len(specs),graficos_historicos=28),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--charts-only',action='store_true')
    p.add_argument('--skip-charts',action='store_true');args=p.parse_args()
    source=WORK/'fontes' if args.charts_only else Path('/home/bagiel/Downloads')
    data=load_data(source)
    if args.charts_only:
        charts(data);return
    (WORK/'fontes').mkdir(parents=True,exist_ok=True)
    (WORK/'tabelas').mkdir(exist_ok=True)
    hashes={}
    for filename in FILES:
        raw=(source/filename).read_bytes();hashes[filename]=hashlib.sha256(raw).hexdigest()
        shutil.copy2(source/filename,WORK/'fontes'/filename)
    M,G,VG,D,S,VS,B,C=data
    C.groupby('exibicao')[['test_accuracy','test_macro_f1','test_balanced_accuracy','test_loss']].mean().reindex(GROUP_LABELS).to_csv(WORK/'tabelas/medias_metodos.csv')
    C.loc[C.episode_nodes>6].groupby('exibicao')[['test_accuracy','test_macro_f1','test_balanced_accuracy','test_loss']].mean().reindex(GROUP_LABELS).to_csv(WORK/'tabelas/medias_sem_episodios_ate6.csv')
    audit=dict(arquivos_sha256=hashes,linhas_metodos=len(M),linhas_densidades=len(D),
               metodos=BASE_METHODS,seeds=[7],grupos_por_bateria=270,
               validacao_episodios=200,teste_episodios=1000,duplicatas=0,grupos_incompletos=0)
    (WORK/'auditoria.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    if not args.skip_charts:
        subprocess.run([str(ROOT/'venv/bin/python'),str(Path(__file__).resolve()),'--charts-only'],check=True,cwd=ROOT)
    build_pdf(data)


if __name__=='__main__':main()
