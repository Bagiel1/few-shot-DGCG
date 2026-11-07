"""Resumo independente, com os CSVs auditados das duas baterias novas."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from PIL import Image as PILImage
from pypdf import PdfReader

from atualizar_relatorio_fewshot import load_data, WORK, NAMES, GROUP_LABELS, DEGREES, n

OUT = Path(__file__).parent / 'resumo_fewshot_resultados_novos_2026-10-26.pdf'
M, G, VG, D, S, VS, B, C = load_data(WORK / 'fontes')
pdfmetrics.registerFont(TTFont('DV', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DVB', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DVB')
PAGE = landscape(A4)
WIDTH = PAGE[0] - 72
styles = {
    'title': ParagraphStyle('title', fontName='DVB', fontSize=19, leading=24, spaceAfter=9),
    'sub': ParagraphStyle('sub', fontName='DVB', fontSize=11, leading=15, spaceBefore=9, spaceAfter=6),
    'body': ParagraphStyle('body', fontName='DV', fontSize=10, leading=14, spaceAfter=7),
    'small': ParagraphStyle('small', fontName='DV', fontSize=8.3, leading=11.5, spaceAfter=5),
    'head': ParagraphStyle('head', fontName='DVB', fontSize=8.4, leading=11, textColor=colors.white),
    'cell': ParagraphStyle('cell', fontName='DV', fontSize=9.2, leading=12),
}
story = []


def p(text, style='body'):
    story.append(Paragraph(text, styles[style]))


def table(headers, rows, widths, fs=9.2):
    data = [[Paragraph(x, styles['head']) for x in headers]] + rows
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 1), (-1, -1), 'DV'), ('FONTSIZE', (0, 1), (-1, -1), fs),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#263746')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f4f5')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 9), ('RIGHTPADDING', (0, 0), (-1, -1), 9),
        ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#b8c0c5')),
    ]))
    story.extend([t, Spacer(1, 8)])


def figure(name, maxheight):
    path = WORK / 'graficos' / name
    with PILImage.open(path) as img:
        w, h = img.size
    scale = min(WIDTH / w, maxheight / h)
    story.append(Image(str(path), width=w*scale, height=h*scale))
    story.append(Spacer(1, 5))


def newpage(title):
    story.append(PageBreak())
    p(title, 'title')


p('Few-shot | resultados novos', 'title')
p('2.430 execuções de métodos/graus + 4.050 de densidades. Dez bases, 27 protocolos por base.', 'body')
p('Ways: 2, 3 e 5; shots e queries: 1, 5 e 10 por classe. Seed 7; 200 episódios de validação e 1.000 de teste por execução. Sem imagens extras; ranked lists episódicas. kNN: k = 5.', 'small')
p('N = ways × (shots + queries). Alvos selecionados pela validação por base/protocolo: 4 intervalos de grau ou 15 densidades. Acurácia e F1 em %; ganhos em pontos percentuais (pp).', 'small')
p('Médias de teste (%)', 'sub')
means = C.groupby('exibicao')[['test_accuracy', 'test_macro_f1', 'test_balanced_accuracy']].mean()
filtered = C.loc[C.episode_nodes.gt(6)].groupby('exibicao')[['test_accuracy', 'test_macro_f1']].mean()
short = ['kNN-in', 'kNN-out', 'kNN mutual', 'kNN union', 'No-graph', 'DGCG 4-6', 'DGCG grau selecionado', 'DGCG densidade selecionada']
rows = []
for key, label in zip(GROUP_LABELS, short):
    rows.append([label] + [n(means.loc[key, col]) for col in means.columns]
                + [n(filtered.loc[key, col]) for col in filtered.columns])
table(['Método', 'Acurácia', 'F1 macro', 'Acurácia<br/>balanceada', 'Acurácia<br/>N > 6', 'F1 macro<br/>N > 6'], rows, [250, 92, 92, 115, 110, WIDTH-659])

newpage('Efeito do grau-alvo por número de classes')
figure('04_grau_por_way.png', 225)
p('Frequência de mudança do alvo escolhido na validação', 'sub')
rows = []
for axis, label in [('n_way', 'Ways'), ('n_shot', 'Shots'), ('n_query', 'Queries')]:
    vg = VG.loc[VG.eixo_variado.eq(axis)]
    vs = VS.loc[VS.eixo_variado.eq(axis)]
    rows.append([label, f'{vg.intervalos_distintos.gt(1).sum()}/{len(vg)}',
                 f'{vs.densidades_distintas.gt(1).sum()}/{len(vs)}'])
table(['Parâmetro variado', 'Intervalo selecionado mudou<br/>(grupos / total)', 'Densidade selecionada mudou<br/>(grupos / total)'], rows, [250, 260, WIDTH-510])

newpage('Densidade | efeito e sensibilidade')
p('Acurácia por densidade-alvo | episódios com N > 6', 'sub')
figure('05_densidade_corte6.png', 220)
rows = []
for label, mask in [('Grade completa', S.episode_nodes.gt(0)), ('N > 6', S.episode_nodes.gt(6)),
                    ('N > 10', S.episode_nodes.gt(10)),
                    ('5-way / 1-shot / 5-query', S.n_way.eq(5) & S.n_shot.eq(1) & S.n_query.eq(5))]:
    q = S.loc[mask]
    rows.append([label, str(len(q)), n(q.baseline_test_accuracy.mean()), n(q.test_accuracy.mean()),
                 n(q.ganho_teste_vs_4_6_pp.mean())])
table(['Recorte', 'Combinações<br/>base/protocolo', 'DGCG 4-6 (%)', 'Densidade<br/>selecionada (%)', 'Ganho (pp)'], rows, [240, 95, 145, 165, WIDTH-645])

newpage('Resultados por base | média dos 27 protocolos')
rows = []
for dataset in sorted(NAMES):
    base = B.loc[B.dataset.eq(dataset) & B.metodo.eq('dgcg')]
    degree = G.loc[G.dataset.eq(dataset)]
    density = S.loc[S.dataset.eq(dataset)]
    rows.append([NAMES[dataset], n(base.test_accuracy.mean()), n(base.test_macro_f1.mean()),
                 n(degree.test_accuracy.mean()), n(density.test_accuracy.mean()), n(density.test_macro_f1.mean()),
                 n(density.loc[density.episode_nodes.gt(6)].ganho_teste_vs_4_6_pp.mean())])
table(['Base', '4-6<br/>Acurácia', '4-6<br/>F1 macro', 'Grau sel.<br/>Acurácia', 'Dens. sel.<br/>Acurácia', 'Dens. sel.<br/>F1 macro', 'Dens. vs. 4-6<br/>Ganho N > 6 (pp)'], rows, [145, 94, 94, 100, 105, 105, WIDTH-643])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#cdd2d6'))
    canvas.line(36, 30, PAGE[0]-36, 30)
    canvas.setFont('DV', 8)
    canvas.setFillColor(colors.HexColor('#62696f'))
    canvas.drawString(36, 18, 'Few-shot / DGCG | novas baterias | 26/10/2026')
    canvas.drawRightString(PAGE[0]-36, 18, f'{doc.page} / 4')
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=PAGE, leftMargin=36, rightMargin=36,
                        topMargin=28, bottomMargin=40, title='Few-shot - resumo dos resultados novos',
                        author='')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
assert len(PdfReader(OUT).pages) == 4, 'Rever paginação antes de entregar.'
print(OUT)
