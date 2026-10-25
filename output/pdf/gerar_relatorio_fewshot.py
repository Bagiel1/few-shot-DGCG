"""Relatorio factual: tabelas e todos os graficos do relatorio HTML."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import ParagraphStyle
from pypdf import PdfReader

PROJECT=Path(__file__).resolve().parents[2]
ANALYSIS=PROJECT/'outputs/analise_experimentos_20261012'
TABLES=ANALYSIS/'tabelas'
GRAPHS=ANALYSIS/'graficos'
OUT=Path(__file__).resolve().parent/'relatorio_fewshot_2026-10-13.pdf'
pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
W,H=landscape(A4)
M=34
CW=W-2*M
C=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
C.setTitle('Few-shot DGCG - resultados experimentais - 13/10/2026')
C.setAuthor('Gabriel Maia Brito')
PAGE=0
NAMES={'bacteria_micro':'Bacteria','birds_micro':'Birds','cars_micro':'Cars',
 'flowers_micro':'Flowers','manufacturing_micro':'Manufacturing','plankton_micro':'Plankton',
 'plant_village_micro':'Plant Village','resisc_micro':'RESISC','sports_micro':'Sports','textures_micro':'Textures'}
METHODS=['knn-in','knn-out','knn-reciprocal','dgcg']
LABEL={'knn-in':'kNN-in','knn-out':'kNN-out','knn-reciprocal':'kNN reciproco','dgcg':'DGCG'}

def read(name):
    with (TABLES/(name+'.csv')).open() as f:return list(csv.DictReader(f))

def n(value,digits=2):
    if value is None or value=='':return '-'
    return f'{float(value):.{digits}f}'.replace('.',',')

def start(title,subtitle=''):
    global PAGE
    if PAGE:C.showPage()
    PAGE+=1
    C.setFillColor(colors.black)
    C.setFont('DV',8)
    C.drawString(M,H-22,'IC | Few-shot DGCG | 13/10/2026')
    C.setFont('DVB',15)
    C.drawString(M,H-48,title)
    if subtitle:
        C.setFont('DV',8.5);C.drawString(M,H-66,subtitle)
    C.setStrokeColor(colors.HexColor('#bfc4c8'))
    C.line(M,37,W-M,37)
    C.setFont('DV',8);C.setFillColor(colors.HexColor('#555555'))
    C.drawString(M,24,'Acuracia em %. IC: intervalo de confianca de 95%. pp: pontos percentuais.')
    C.drawRightString(W-M,24,str(PAGE))
    C.setFillColor(colors.black)
    return H-84

def lines(y,items,size=8.5):
    C.setFont('DV',size)
    for item in items:
        assert pdfmetrics.stringWidth(item,'DV',size)<CW, item
        C.drawString(M,y,item);y-=14
    return y-4

def heading(y,title):
    C.setFont('DVB',10);C.drawString(M,y,title)
    return y-15

def table(y,headers,rows,widths=None,size=9,pad=3):
    widths=widths or [CW/len(headers)]*len(headers)
    assert sum(widths)<=CW+1
    style=ParagraphStyle('cell',fontName='DV',fontSize=size,leading=size+1.3)
    bold=ParagraphStyle('head',fontName='DVB',fontSize=size,leading=size+1.3)
    data=[[Paragraph(str(x),bold) for x in headers]]
    data += [[Paragraph(str(x).replace('–','-'),style) for x in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LINEBELOW',(0,0),(-1,0),.55,colors.black),
        ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#cccccc')),
        ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad),
    ]))
    _,height=t.wrap(CW,H)
    if y-height<44:raise ValueError(f'Tabela ultrapassa pagina {PAGE}: bottom={y-height:.1f}')
    t.drawOn(C,M,y-height)
    return y-height-15

allrows=read('resultados_unificados_1080')
fixed=read('comparacao_fixa_5way_1shot_5query')
summary=json.loads((ANALYSIS/'achados.json').read_text())

y=start('Configuracao e medias gerais','Meta-Album Micro | rankings episodicos | GRaNDe desligado')
y=lines(y,[
 '10 bases; 20 classes por base; 40 imagens por classe; divisao de classes 10 treino / 5 validacao / 5 teste.',
 'Grade: n-way = 2, 3, 5; n-shot = 1, 5, 10; n-query = 1, 5, 10. 27 protocolos por metodo/base.',
 'ResNet-18 ImageNet congelada; SGC com 2 saltos; kNN k=5, euclidiano, binario.',
 'DGCG euclidiano + RBO; L=200; top-k=40; p=0,9; baseline de grau 4-6; seed=7.',
 '2.000 episodios de treino; 200 de validacao por checagem; 1.000 de teste. Checkpoint escolhido por validacao.',
 'CSV: 1.630 linhas brutas; 270 repeticoes kNN-in; 28 ajustes incompletos em Plant Village; 1.332 observacoes validas.',
])
y=heading(y,'Medias de teste - dez bases')
without={r['metodo']:r for r in read('medias_metodos_sem_episodios_ate6nos')}
rows=[]
for row in summary['overview']:
 m=row['metodo'];rows.append([LABEL[m],n(row['media_teste']),n(without[m]['media_teste']),n(fmean(float(r['test_accuracy']) for r in fixed if r['metodo']==m)),n(row['media_loss'],4)])
y=table(y,['Metodo','27 protocolos','25 protocolos (N > 6)','Fixo: 5-way/1-shot/5-query','Loss: grade completa'],rows,[140,120,175,205,CW-640])
y=heading(y,'DGCG: medias por n-way - dez bases; nove combinacoes shot/query por n-way')
y=table(y,['n-way','Acuracia de teste','Acerto por acaso'],[[str(way),n(fmean(float(r['test_accuracy']) for r in allrows if r['metodo']=='dgcg' and int(r['n_way'])==way)),n(100/way)] for way in [2,3,5]],[140,250,CW-390])
lines(y,['Os dois protocolos excluidos usam 2/3-way, 1-shot, 1-query. Com k=5, os grafos kNN ficam completos.'])

y=start('Flowers Micro - contextos de rankings e referencias','5-way / 1-shot / 5-query | L=200 | top-k=40 | grau-alvo 4-6')
y=heading(y,'Baselines e DGCG episodico - resultados dos CSVs')
flowers=sorted([r for r in fixed if r['dataset']=='flowers_micro'],key=lambda r:METHODS.index(r['metodo']))
y=table(y,['Metodo','Validacao','Teste','IC teste','Loss teste'],[[LABEL[r['metodo']],n(r['val_accuracy']),n(r['test_accuracy']),n(r['test_ci95']),n(r['test_loss'],4)] for r in flowers])
y=heading(y,'Contextos globais - resultados disponiveis')
context=[
 ['Todas as imagens de treino','Somente episodio','83,43','0,63','0,5299','Execucao local'],
 ['Todas as imagens de treino','Treino + episodio','75,08','-','-','Informado por Gabriel'],
 ['Todas as imagens de treino','Toda a particao de teste','89,26','0,52','0,3778','Execucao local'],
]
y=table(y,['Rankings no treino','Rankings na avaliacao','Teste','IC teste','Loss teste','Fonte'],context,[155,200,65,65,80,CW-565],size=8.5)
y=heading(y,'Banco reservado de referencias extras - resultados informados por Gabriel')
y=table(y,['Extras por episodio','Teste','Fonte'],[
 ['0 (padrao anterior, sem extras)','83,43','Execucao local; sem reserva de banco'],
 ['30','86,71','Informado por Gabriel'],['50','87,58','Informado por Gabriel'],['100','86,94','Informado por Gabriel'],
],[230,140,CW-370],size=8.5,pad=2)
lines(y,[
 'Extras: imagens nao rotuladas das classes novas, reservadas fora de suporte/consultas; grafo somente episodico.',
 'O banco altera o pool de avaliacao. Sem flag e N=0 com banco reservado nao sao a mesma amostragem.',
 'Os resultados com 30/50/100 extras nao vieram acompanhados de IC, loss, validacao ou logs.',
 '89,26% usa a particao inteira nao rotulada: protocolo transdutivo global, diferente do episodico.',
],size=7.7)

y=start('DGCG - varredura de grau e densidade','Nove bases completas | 5-way / 1-shot / 5-query | 30 nos; 29 candidatos por no')
rank=sorted(read('ranking_grau_densidade_recalculado'),key=lambda r:(0 if r['modo']=='grau' else 1,float(r['alvo_min']),float(r['alvo_max'])))
y=table(y,['Modo','Alvo','Grau equivalente','Validacao media','Teste medio','Loss teste'],[
 [r['modo'],r['alvo'],f"{n(float(r['alvo_min'])*(29 if r['modo']=='densidade' else 1),3)}"+(' - '+n(float(r['alvo_max']),3) if r['alvo_min']!=r['alvo_max'] else ''),n(r['media_val']),n(r['media_teste']),n(r['media_loss'],4)]
 for r in rank],[110,105,160,140,135,CW-650],size=8.4,pad=1)
lines(y,[
 'Mesmo alvo no treino, validacao e teste. Cada alvo corresponde a um modelo treinado com essa configuracao.',
 'Baseline 4-6: validacao 67,22%; teste 62,35%. Densidade 0,125: validacao 67,74%; teste 62,64%.',
 'Densidade-alvo = grau-alvo / 29 neste protocolo. Densidade realizada nao foi registrada.',
 'Plant Village: 28 execucoes incompletas, excluidas desta varredura.',
],size=8)

y=start('DGCG - selecao por base e comparacoes','Alvo escolhido pela maior validacao; ganho medido no teste contra grau 4-6 da mesma rodada')
gains=sorted(read('ganhos_ajuste_local_validacao'),key=lambda r:r['dataset'])
y=table(y,['Base','Modo / alvo','Val. 4-6','Val. escolhida','Teste 4-6','Teste escolhido','Ganho (pp)'],[
 [NAMES[r['dataset']],r['modo']+' '+r['alvo'],n(r['val_accuracy_baseline']),n(r['val_accuracy_ajuste']),n(r['test_accuracy_baseline']),n(r['test_accuracy_ajuste']),n(r['ganho_teste_pp'])] for r in gains
],[145,125,95,115,95,115,CW-690],size=8,pad=3)
y=lines(y,['Ganho medio da escolha por base: -0,02 pp. Escolha do alvo em outras oito bases: ganho medio -0,02 pp.'],size=8)
y=heading(y,'Comparacoes pareadas: media dos 27 protocolos em cada uma das dez bases')
tests=read('comparacoes_pareadas_por_base')
y=table(y,['Metodo 1','Metodo 2','Diferenca media (pp)','p Wilcoxon','p Holm'],[
 [LABEL[r['metodo_1']],LABEL[r['metodo_2']],n(r['ganho_medio_pp'],3),n(r['p_wilcoxon'],4),n(r['p_holm'],4)] for r in tests
],[145,145,185,145,CW-620],size=8,pad=2)
lines(y,['Friedman: p=0,01694. Unidade: dataset. IC original mede episodios, nao variacao entre treinamentos/seeds.'],size=8)

selected=read('protocolos_selecionados_por_validacao_nway')
for way in [2,3,5]:
 y=start(f'Protocolos escolhidos na validacao - {way}-way','Dez bases | escolha de shot/query por metodo/base | teste do checkpoint escolhido')
 chosen=sorted([r for r in selected if int(r['n_way'])==way],key=lambda r:(r['dataset'],METHODS.index(r['metodo'])))
 y=table(y,['Base','Metodo','Shot','Query','Validacao','Teste','IC teste','Loss teste'],[
  [NAMES[r['dataset']],LABEL[r['metodo']],r['n_shot'],r['n_query'],n(r['val_accuracy']),n(r['test_accuracy']),n(r['test_ci95']),n(r['test_loss'],4)] for r in chosen
 ],[145,125,45,45,100,100,90,CW-650],size=7.8,pad=.65)
 lines(y,['Configuracoes podem usar quantidades diferentes de suporte e consultas.'],size=7.8)

FIGURES=[
 ('01_medias_metodos','Medias dos metodos','Dez bases; grade completa e grade sem os dois protocolos com ate seis nos.'),
 ('02_metodos_por_base','Metodos por base','Teste medio nos mesmos 27 protocolos; dez bases.'),
 ('03_ganho_knnin','Diferenca para kNN-in','Teste medio dos 27 protocolos; valores em pp.'),
 ('04_protocolo_default','Protocolo fixo','5-way / 1-shot / 5-query; dez bases; teste.'),
 ('05_selecao_validacao','Selecao de shot/query pela validacao','Escolha individual por metodo/base/n-way; teste medio em dez bases.'),
 ('06_heatmap_way2','Combinacoes de protocolo - 2-way','Teste medio em dez bases; shot e query variaveis.'),
 ('06_heatmap_way3','Combinacoes de protocolo - 3-way','Teste medio em dez bases; shot e query variaveis.'),
 ('06_heatmap_way5','Combinacoes de protocolo - 5-way','Teste medio em dez bases; shot e query variaveis.'),
 ('07_efeito_n_shot','Efeito de n-shot','Medias balanceadas nos demais parametros da grade e nas dez bases.'),
 ('07_efeito_n_query','Efeito de n-query','Medias balanceadas nos demais parametros da grade e nas dez bases.'),
 ('08_gap_val_teste','Validacao menos teste','Diferenca media em pp; 27 protocolos por metodo/base.'),
 ('09_loss_validacao','Loss, acuracia e splits','Dados da grade completa; cada ponto e uma configuracao/base.'),
 ('10_curvas_grau_densidade','Grau e densidade - resposta media','Nove bases; somente 5-way / 1-shot / 5-query. Roxo: validacao; azul: teste.'),
 ('11_dgcg_grau_bases','Grau - validacao por base','Nove bases completas; 5-way / 1-shot / 5-query.'),
 ('11_dgcg_densidade_bases','Densidade - validacao por base','Nove bases completas; 5-way / 1-shot / 5-query.'),
 ('12_ganhos_ajuste','Ajuste escolhido versus baseline 4-6','Alvo escolhido pela validacao em cada base; diferenca no teste, em pp.'),
 ('13_ajuste_vs_baselines','DGCG ajustado e baselines','Nove bases; 5-way / 1-shot / 5-query. Rodadas distintas.'),
 ('14_orcamento','Quantidade de imagens por episodio','5-way; suporte e consultas variaveis; teste medio em dez bases.'),
]
for filename,title,caption in FIGURES:
 start(title,caption)
 path=GRAPHS/(filename+'.png')
 width,height=Image.open(path).size
 scale=min(CW/width, (H-150)/height)
 dw,dh=width*scale,height*scale
 C.drawImage(str(path),M+(CW-dw)/2,80+(H-150-dh)/2,width=dw,height=dh)

# Os mesmos dados dos dez graficos por base do HTML, em paineis verticais para
# manter texto/numeros legiveis no PDF. Cores em escala fixa de 0 a 100%.
ANCHORS=[(68,1,84),(59,82,139),(33,145,140),(94,201,98),(253,231,37)]
def color(value):
    t=max(0,min(1,value/100))*4
    j=min(3,int(t));a=t-j
    rgb=[(ANCHORS[j][i]*(1-a)+ANCHORS[j+1][i]*a)/255 for i in range(3)]
    return colors.Color(*rgb)

for ds in sorted(NAMES):
 start(f'{NAMES[ds]} - todos os protocolos','Teste (%). Paineis dos graficos por base do HTML; s=shot, q=query.')
 for panel,way in enumerate([2,3,5]):
  y=H-92-panel*146
  C.setFont('DVB',11);C.drawString(M,y,f'{way}-way')
  x0=M+120;cellw=(CW-120)/9
  C.setFont('DV',8)
  for j,(shot,query) in enumerate([(s,q) for s in [1,5,10] for q in [1,5,10]]):
   C.drawCentredString(x0+(j+.5)*cellw,y-17,f'{shot}s/{query}q')
  for i,m in enumerate(METHODS):
   bottom=y-41-i*22
   C.setFillColor(colors.black);C.setFont('DV',9)
   C.drawRightString(x0-8,bottom+7,LABEL[m])
   for j,(shot,query) in enumerate([(s,q) for s in [1,5,10] for q in [1,5,10]]):
    matching=[r for r in allrows if r['dataset']==ds and r['metodo']==m and int(r['n_way'])==way and int(r['n_shot'])==shot and int(r['n_query'])==query]
    assert len(matching)==1
    v=float(matching[0]['test_accuracy'])
    C.setFillColor(color(v));C.rect(x0+j*cellw,bottom,cellw,22,stroke=0,fill=1)
    C.setFillColor(colors.white if v<65 else colors.black)
    C.setFont('DV',9);C.drawCentredString(x0+(j+.5)*cellw,bottom+7,n(v,1))
 C.setFillColor(colors.black)

y=start('Fontes e registro','Dados recebidos e resultados locais; sem inferencias de metricas ausentes')
y=lines(y,[
 'CSV originais: 17 arquivos fornecidos por Gabriel; auditoria e agregacoes do relatorio HTML de 12/10/2026.',
 'Contextos globais: execucao local de 13/10, checkpoint fixo para comparar avaliacao episodica e teste global.',
 'Resultados 75,08% e extras 30=86,71%; 50=87,58%; 100=86,94%: informados por Gabriel nesta conversa.',
 'Nos novos resultados informados, IC, loss e validacao nao foram fornecidos.',
 'Nao houve teste de significancia entre os novos contextos nem repeticoes identificadas por seed.',
],size=8.5)
y=heading(y,'Arquivos de entrada do relatorio HTML')
for path in sorted((ANALYSIS/'fontes').glob('*.csv')):
 y=lines(y,[path.name],size=8)
C.save()
reader=PdfReader(OUT)
assert len(reader.pages)==PAGE
texts=[p.extract_text() or '' for p in reader.pages]
assert all(len(t)>50 for t in texts)
assert all(s in texts[1] for s in ['86,71','87,58','86,94','75,08','89,26','83,43'])
assert len(FIGURES)+len(NAMES)==28
print(f'PDF: {OUT}\nPaginas: {PAGE}\nGraficos: 28\nTabelas e novos resultados verificados.')
