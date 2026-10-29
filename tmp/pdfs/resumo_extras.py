from pathlib import Path
import csv
from statistics import mean
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'runs/busca_extras_auto'
OUT = ROOT / 'output/pdf/resumo_experimento_extras.pdf'
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
rows, avgs, status = [read(SRC / f) for f in ('resultados.csv','medias.csv','status.csv')]
assert len(rows)==470 and len(status)==470 and all(r['status']=='complete' for r in status)
old = [r for r in read(ROOT/'outputs/analise_experimentos_20261026/fontes/resultados_metricas.csv') if r['metodo']=='no-graph']
def fmt(v, signed=False): return (f'{float(v):+.2f}' if signed else f'{float(v):.2f}').replace('.',',')
def get(method, degree='-', mode='none', phase='none'):
    return next(r for r in avgs if (r['metodo'],r['alvo'],r['extra_mode'],r['extra_phase'])==(method,degree,mode,phase))
pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DVB',italic='DV',boldItalic='DVB')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='T',fontName='DVB',fontSize=19,leading=24,spaceAfter=12))
styles.add(ParagraphStyle(name='H',fontName='DVB',fontSize=11,leading=15,spaceBefore=12,spaceAfter=7))
styles.add(ParagraphStyle(name='B',fontName='DV',fontSize=9.5,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name='S',fontName='DV',fontSize=8,leading=11,spaceAfter=7))
styles.add(ParagraphStyle(name='TH',fontName='DVB',fontSize=8,leading=11,textColor=colors.white))
styles.add(ParagraphStyle(name='TD',fontName='DV',fontSize=8.3,leading=12))
story=[]
def p(t,s='B'): story.append(Paragraph(t,styles[s]))
def table(headers,data,widths):
    content=[[Paragraph(str(v),styles['TH']) for v in headers]]+[[Paragraph(str(v),styles['TD']) for v in row] for row in data]
    t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#263746')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f4f6'),colors.white]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.extend([t,Spacer(1,8)])
def page(title): story.append(PageBreak()); p(title,'T')

p('Few-shot: experimento com imagens extras','T')
p('470 execuções concluídas • 10 bases • seed 7','S')
p('O teste comparou imagens extras usadas apenas para criar as ranked lists com imagens extras incluídas na propagação do grafo. Também separou em quais fases elas aparecem.')
p('Configuração usada','H')
table(['Item','Configuração'],[
['Episódio','2 classes; 10 suportes e 10 queries por classe: 40 imagens.'],
['Extras ativos','10 imagens por classe, 20 por episódio.'],
['Banco reservado','20 imagens por classe em cada divisão, inclusive no controle sem extras. As imagens reservadas ficam fora do suporte e das queries.'],
['Avaliação','200 episódios de validação e 1.000 de teste por execução.'],
['Métodos','kNN-in, out, mutual e union (k = 5); no graph; DGCG 2-4 e 4-6.']], [110,405])
p('De onde vêm as imagens','H')
p('Treino usa extras do treino; validação usa extras da validação; teste usa extras do teste. As classes dessas divisões são separadas. Os extras pertencem às classes do episódio: os rótulos são usados para montar a amostra balanceada, mas não são entregues ao modelo como supervisão.')
p('Ranked lists: o grafo mantém as 40 imagens do episódio. As extras influenciam as correlações e as conexões. Propagação: as 20 extras viram nós, formando um grafo com 60 imagens nas fases em que estão ativas. Os protótipos usam o suporte; a loss e as métricas usam as queries. Não há pseudorrótulos.')
p('Fases testadas','H')
table(['Configuração','Treino','Validação','Teste'],[['Sem extras','Não','Não','Não'],['Só treino','Sim','Não','Não'],['Só validação','Não','Sim','Não'],['Só teste','Não','Não','Sim'],['Treino + validação','Sim','Sim','Não'],['Todas','Sim','Sim','Sim']],[185,110,110,110])
p('Por base: 7 controles + 20 configurações DGCG + 20 configurações kNN = 47. Nas 10 bases: 470. No graph aparece apenas como controle.','S')

page('O que mudou nos resultados')
p('Valores abaixo: média da acurácia de teste nas 10 bases, com o mesmo peso para cada base. Nas colunas com extras, eles estão ativos no treino, validação e teste.')
data=[]
for method,deg,label in [('dgcg','2:4','DGCG 2-4'),('dgcg','4:6','DGCG 4-6'),('knn-in','-','kNN-in'),('knn-out','-','kNN-out'),('knn-mutual','-','kNN-mutual'),('knn-union','-','kNN-union'),('no-graph','-','No graph')]:
    data.append([label,fmt(get(method,deg)['test_accuracy'])+'%',fmt(get(method,deg,'rankings','all')['test_accuracy'])+'%' if method=='dgcg' else '-',fmt(get(method,deg,'propagation','all')['test_accuracy'])+'%' if method!='no-graph' else '-'])
table(['Método','Sem extras','Só ranked lists','Propagação'],data,[155,110,125,125])
p('Leitura dos resultados','H')
p('<b>DGCG 2-4:</b> extras só nas ranked lists aumentaram a média em 0,38 ponto percentual. Na propagação, ela caiu 0,40 ponto. <b>DGCG 4-6:</b> os ganhos foram 0,30 ponto nas ranked lists e 0,50 ponto na propagação.')
p('<b>No graph continuou competitivo:</b> 90,80% de acurácia, contra 90,83% do kNN-union com extras nas três fases. No F1 macro, foram 90,66% e 90,64%, respectivamente. Essa diferença não sustenta uma vantagem clara do grafo.')
p('Por que antes no graph tinha 78,30%?','H')
same=[r for r in old if (r['n_way'],r['n_shot'],r['n_query'])==('2','10','10')]
table(['Comparação do no graph','Acurácia'],[
['Bateria anterior: média dos 27 protocolos',fmt(mean(float(r['test_accuracy']) for r in old))+'%'],
['Bateria anterior: apenas 2-way / 10-shot / 10-query',fmt(mean(float(r['test_accuracy']) for r in same))+'%'],
['Bateria atual: 2-way / 10-shot / 10-query',fmt(get('no-graph')['test_accuracy'])+'%']],[405,110])
p('Os 78,30% juntavam 2, 3 e 5 ways, com 1, 5 e 10 shots e queries. A bateria atual usa apenas duas classes e dez suportes por classe. Por isso, os números gerais dos dois estudos não são diretamente comparáveis.')
p('No mesmo protocolo, a diferença é de aproximadamente 0,45 ponto. A bateria atual também reserva o banco de extras no controle, alterando o conjunto disponível para os episódios. Essa diferença residual não demonstra uma melhoria do método.')

page('Em qual fase os extras ajudaram?')
p('Cada valor é a mudança da acurácia média de teste em relação ao mesmo método sem extras, nesta bateria. Positivo indica melhora; negativo, piora. Unidade: pontos percentuais (pp).')
phases=['train','val','test','train-val','all']
data=[]
for method,deg,mode,label in [('dgcg','2:4','rankings','DGCG 2-4 / rankings'),('dgcg','2:4','propagation','DGCG 2-4 / propagação'),('dgcg','4:6','rankings','DGCG 4-6 / rankings'),('dgcg','4:6','propagation','DGCG 4-6 / propagação')]+[(m,'-','propagation',m+' / propagação') for m in ['knn-in','knn-out','knn-mutual','knn-union']]:
    data.append([label]+[fmt(get(method,deg,mode,ph)['ganho_accuracy_vs_sem_extras_pp'],True) for ph in phases])
table(['Método e uso das extras','Só treino','Só val.','Só teste','Treino + val.','Todas'],data,[190,65,65,65,65,65])
p('Como interpretar','H')
p('<b>Só treino:</b> o modelo aprende com extras, mas validação e teste usam apenas suporte e queries. <b>Só validação:</b> os extras podem mudar qual checkpoint é escolhido; por isso o teste pode mudar mesmo sem extras no teste. <b>Só teste:</b> avalia o efeito dos extras na inferência, mantendo treino e validação sem extras.')
p('No DGCG 4-6, usar extras apenas nas ranked lists do teste melhorou 8 das 10 bases, com ganho médio de 0,38 pp. Usar propagação nas três fases melhorou 6 das 10 bases e teve ganho médio de 0,50 pp. Melhor média e maior número de bases beneficiadas não são a mesma coisa.')
p('O que dá para concluir','H')
p('Os extras têm efeito, mas o ganho depende do método e da fase. O intervalo 2-4 não foi sempre superior ao 4-6. Nesta configuração, no graph alcançou desempenho próximo dos melhores resultados com extras.')
p('Essas comparações descrevem os resultados de teste; não escolhem um vencedor pela validação. Uma seed e um único protocolo não mostram se os ganhos são estáveis. Repetir com outras seeds e em 5-way permite avaliar isso.')
p('Usar extras do teste caracteriza uma avaliação com dados adicionais sem rótulos. A comparação deve informar esse acesso. As divisões de classes permanecem separadas, mas o orçamento de imagens é maior que no episódio sem extras.','S')
p('Fontes e conferência','H')
p('Bateria atual: busca_extras_auto/resultados.csv, medias.csv, status.csv e manifest.json. Conferidos 470 resultados únicos, 47 por base e médias consistentes com os resultados individuais. Comparação histórica: resultados_metricas.csv arquivado em analise_experimentos_20261026.','S')

def footer(c,d):
    c.setFont('DV',8); c.setFillColor(colors.HexColor('#61717a'))
    c.drawString(40,25,'Few-shot | Imagens extras | 28/10/2026')
    c.drawRightString(A4[0]-40,25,str(d.page))
OUT.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(OUT)
assert len(reader.pages)==3,len(reader.pages)
print(OUT)
print('Páginas:',len(reader.pages))
