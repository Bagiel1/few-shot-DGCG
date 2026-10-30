from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pandas as pd
from pathlib import Path
root=Path('tmp/pdfs/extras_20261030');dest=Path('output/pdf/analise_extras_protocolos_2026-10-30.pdf')
pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVu-Bold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='DejaVu',fontSize=9.3,leading=13,spaceAfter=7,textColor=colors.HexColor('#243449')))
styles.add(ParagraphStyle(name='TitleCustom',fontName='DejaVu-Bold',fontSize=20,leading=24,spaceAfter=9,textColor=colors.HexColor('#142E47')))
styles.add(ParagraphStyle(name='HeadingCustom',fontName='DejaVu-Bold',fontSize=11.5,leading=15,spaceBefore=10,spaceAfter=7,textColor=colors.HexColor('#147D85')))
styles.add(ParagraphStyle(name='SmallCustom',fontName='DejaVu',fontSize=7.7,leading=10.5,spaceAfter=6,textColor=colors.HexColor('#526071')))
styles.add(ParagraphStyle(name='CellCustom',fontName='DejaVu',fontSize=8,leading=10))
flow=[]
def p(t,style='BodyCustom'):flow.append(Paragraph(t,styles[style]))
def h(t):p(t,'HeadingCustom')
def table(rows,widths):
 rr=[[Paragraph(str(v),styles['CellCustom']) for v in r] for r in rows]
 t=Table(rr,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DFEDF1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F6F8')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#B6CED7'))]))
 flow.append(t);flow.append(Spacer(1,7))
def num(v,signed=False):return (f'{v:+.2f}' if signed else f'{v:.2f}').replace('.',',')
p('Few-shot: análise dos extras','TitleCustom')
p('30/10/2026 • Nova bateria de protocolos • Síntese em 2 páginas','SmallCustom')
p('<b>Resultado central:</b> extras nas três fases produziram ganhos modestos e dependentes do método; mais imagens não significaram melhoria contínua.. Em 1-shot, kNN-union ficou bem acima do modelo sem grafo, mas a maior parte dessa diferença já existia sem extras..')
h('O que foi concluído e conferido')
p('Foram conferidas <b>4.740 combinações únicas, todas completas</b>, incluindo 420 controles: dez bases, seis protocolos (2/5-way × 1/5/10-shot), 10 queries por classe e seed 7.. Há 5/10/20 extras por classe, banco fixo de 20/classe, 200 episódios de validação e 1.000 de teste.. As 474 médias e os ganhos fornecidos coincidem com o recálculo, até precisão numérica..')
p('Conforme o protocolo documentado, extras vêm das classes do episódio, sem distratoras, e seus rótulos servem ao amostrador.. Em <b>rankings</b>, alteram as conexões; em <b>propagação</b>, também entram como nós.. O banco é reservado inclusive no controle..','SmallCustom')
h('Comparação por protocolo, com escolha pela validação')
p('Para cada protocolo, selecionamos uma configuração comum às dez bases pela maior acurácia média de validação, entre controles e extras nas três fases.. A tabela mostra seu teste; não escolhe o maior teste retrospectivamente..')
sel=pd.read_csv(root/'selecao.csv');rows=[['Way / shot','Escolha por validação¹','Teste (%)','Δ controle²','Sem grafo (%)']]
for _,r in sel.iterrows():
 name='DGCG 4-6' if r.metodo=='dgcg' else 'kNN-union'
 rows.append([f'{int(r.n_way)} / {int(r.n_shot)}',f'{name}, {int(r.extras_per_class)}/classe',num(r.test_accuracy),num(r.ganho_accuracy_vs_sem_extras_pp,True),num(r.no_graph)])
table(rows,[64,165,76,87,113])
p('¹ Todas as escolhas usam propagação e extras em treino, validação e teste.. ² Diferença em pontos percentuais (pp) contra o mesmo método/alvo sem extras, na mesma base e protocolo, depois promediada.. Queries = 10 em todas as linhas..','SmallCustom')
p('Em 2-way/1-shot, a configuração escolhida supera o sem grafo em <b>5,69 pp</b>, mas as extras acrescentam apenas <b>0,62 pp</b> sobre kNN-union sem extras.. Em 5-way/1-shot, são <b>6,68 pp</b> versus sem grafo e <b>0,91 pp</b> atribuíveis à condição com extras..')
p('A escolha pela validação não garante liderança no teste: em 2-way/10-shot, 5-way/5-shot e 5-way/10-shot, o sem grafo supera a configuração escolhida.. Não comparar acurácia entre ways como se fossem a mesma dificuldade..')
flow.append(PageBreak())
p('Quando e quantas extras ajudam','TitleCustom')
h('Efeito da fase: ganho sobre o próprio controle (pp)')
d=pd.read_csv('/home/bagiel/Downloads/buscas_extras.csv')
g=d[d.extras_per_class>0].groupby(['metodo','alvo','extra_mode','extra_phase']).ganho_accuracy_vs_sem_extras_pp.mean().unstack()
rows=[['Método / uso das extras','Só treino','Treino + val.','Todas as fases']]
for method,target,mode in [('dgcg','2:4','rankings'),('dgcg','2:4','propagation'),('dgcg','4:6','rankings'),('dgcg','4:6','propagation'),('knn-in','-','propagation'),('knn-out','-','propagation'),('knn-mutual','-','propagation'),('knn-union','-','propagation')]:
 r=g.loc[(method,target,mode)];label=(f'DGCG {target.replace(":","-")} / '+('rankings' if mode=='rankings' else 'propagação')) if method=='dgcg' else method+' / propagação'
 rows.append([label,num(r['train'],True),num(r['train-val'],True),num(r['all'],True)])
table(rows,[223,88,94,100])
p('Cada célula resume 180 comparações: dez bases × seis protocolos × três quantidades, com pesos iguais.. São médias descritivas de condições relacionadas, não 180 replicações independentes..','SmallCustom')
p('Nas 1.440 correspondências entre treino+validação e todas as fases, as métricas de acurácia de validação são idênticas; ativar extras também no teste eleva o teste em <b>0,316 pp</b> na média.. Isso sustenta investigar seu uso na avaliação, sem concluir benefício para toda configuração..')
h('Quantidade: o efeito não é crescente')
q=d[(d.metodo=='dgcg')&(d.extra_phase=='all')].groupby(['alvo','extra_mode','extras_per_class']).ganho_accuracy_vs_sem_extras_pp.mean().unstack()
rows=[['DGCG / todas as fases','5 por classe','10 por classe','20 por classe']]
for target,mode in [('2:4','rankings'),('2:4','propagation'),('4:6','rankings'),('4:6','propagation')]:
 r=q.loc[(target,mode)];rows.append([target.replace(':','-')+' / '+('rankings' if mode=='rankings' else 'propagação')]+[num(r[n],True) for n in [5,10,20]])
table(rows,[223,94,94,94])
p('Ganhos em pp, média de dez bases e seis protocolos.. DGCG 2-4 com propagação passa de +0,41 para -0,23 pp ao aumentar de 5 para 20 extras/classe.. No DGCG 4-6, 10 extras/classe têm a maior média observada nos dois modos, mas isso não estabelece uma quantidade ótima universal..','SmallCustom')
h('Relação com o que já sabíamos')
p('A bateria anterior de 470 execuções cobria apenas 2-way/10-shot/10-query e 10 extras/classe.. Agora há evidência de outros regimes, sobretudo 1-shot.. No recorte 2-way/10-shot, o sem grafo passou de 90,7965% na rodada anterior para 90,7535% nesta.. Há 310 configurações em comum: diferença absoluta média de 0,297 pp e máxima de 3,640 pp no teste; a causa não é identificável pelos CSVs.. Não juntar as rodadas como réplicas equivalentes..')
p('<b>Conclusão:</b> manter o sem grafo como referência, separar ganhos do grafo dos ganhos das extras e repetir candidatos com seeds pareadas.. Há só uma seed; os CSVs não trazem predições por episódio nem comprovam identidade dos checkpoints.. Os ICs por execução não medem variação entre treinamentos..')
p('<b>Fontes:</b> buscas_extras.csv (resultados), medias_buscas_extras.csv (médias) e status_buscas_extas.csv (conclusão), recebidos em 30/10/2026.. Histórico: CONTEXTO_FEWSHOT.md, nota ELO “IC - Few-shot” e runs/busca_extras_auto/resultados.csv.. Não foram executados novos treinos..','SmallCustom')
def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#CDD8DF'));c.line(45,37,550,37);c.setFont('DejaVu',8);c.setFillColor(colors.HexColor('#526071'));c.drawString(45,24,'IC • Few-shot DGCG • Análise descritiva');c.drawRightString(550,24,str(doc.page))
SimpleDocTemplate(str(dest),pagesize=(595.28,841.89),rightMargin=45,leftMargin=45,topMargin=38,bottomMargin=48).build(flow,onFirstPage=footer,onLaterPages=footer)
print(dest.resolve())
