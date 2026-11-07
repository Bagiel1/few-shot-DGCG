from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/pdf/experimentos_1_a_4_com_comentarios.pdf"
OUTPUT = ROOT / "output/pdf/experimentos_1_a_4_com_comentarios_e_novos.pdf"
DATA = ROOT / "resultados/explicacao_fewshot_20261103/fontes"

PAGE_W, PAGE_H = landscape(A4)
BLUE = "#4f79a7"
ORANGE = "#f28e2b"
GRAY = "#777777"
METHODS = {"no-graph": "Sem grafo", "union": "kNN-union", "dgcg": "DGCG 4-6"}


def fmt(value: float, signed: bool = False) -> str:
    pattern = "+.2f" if signed else ".2f"
    return format(float(value), pattern).replace(".", ",")


def footer(pdf: canvas.Canvas, page_number: int) -> None:
    pdf.setStrokeColor(colors.HexColor("#b7b7b7"))
    pdf.setLineWidth(0.5)
    pdf.line(36, 30, PAGE_W - 36, 30)
    pdf.setFillColor(colors.HexColor("#666666"))
    pdf.setFont("Helvetica", 6.5)
    pdf.drawString(36, 18, "Experimentos few-shot - baterias 1 a 7")
    pdf.drawRightString(PAGE_W - 36, 18, f"Página {page_number}")
    pdf.setFillColor(colors.black)


def title(pdf: canvas.Canvas, text: str) -> None:
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(36, PAGE_H - 38, text)


def section_label(pdf: canvas.Canvas, text: str, y: float) -> None:
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(36, y, text)


def table(pdf: canvas.Canvas, rows: list[list[str]], widths: list[float], x: float, y_top: float,
          row_h: float = 14, font_size: float = 7.2) -> float:
    total_w = sum(widths)
    y = y_top
    for row_index, row in enumerate(rows):
        fill = colors.HexColor("#dddddd") if row_index == 0 else colors.white
        pdf.setFillColor(fill)
        pdf.rect(x, y - row_h, total_w, row_h, stroke=0, fill=1)
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica-Bold" if row_index == 0 else "Helvetica", font_size)
        cursor = x
        for cell, width in zip(row, widths):
            pdf.drawString(cursor + 3, y - row_h + 4, str(cell))
            cursor += width
        y -= row_h
    pdf.setStrokeColor(colors.HexColor("#777777"))
    pdf.setLineWidth(0.35)
    for idx in range(len(rows) + 1):
        line_y = y_top - idx * row_h
        pdf.line(x, line_y, x + total_w, line_y)
    cursor = x
    pdf.line(cursor, y_top, cursor, y)
    for width in widths:
        cursor += width
        pdf.line(cursor, y_top, cursor, y)
    pdf.setStrokeColor(colors.black)
    return y


def add_plot(pdf: canvas.Canvas, fig: plt.Figure, x: float, y: float, w: float, h: float) -> None:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    pdf.drawImage(ImageReader(buffer), x, y, width=w, height=h, preserveAspectRatio=True, anchor="c")


def config_table(pdf: canvas.Canvas, values: list[str]) -> None:
    rows = [
        ["Bases", "Protocolo", "Seed", "Embedding", "Treino", "Execuções"],
        values,
    ]
    table(pdf, rows, [110, 140, 60, 120, 180, 75], 36, PAGE_H - 57, row_h=12, font_size=6.4)


def page_sem_treino(pdf: canvas.Canvas, e1: pd.DataFrame) -> None:
    title(pdf, "5. Experimento sem treino")
    config_table(pdf, ["10 bases Micro", "5-way; 1/5-shot; 10-query", "7", "512 sem treino; 256 treinado", "Sem treino vs. treino com 2.000 episódios", "120"])

    raw = e1[(e1.feat == "raw") & e1.method.isin(METHODS)]
    summary = raw.groupby(["shot", "method", "trainfree"])[["val", "test", "f1"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.0), sharey=True)
    for ax, shot in zip(axes, [1, 5]):
        part = summary[summary.shot == shot]
        labels = [METHODS[m] for m in METHODS]
        trained = [part[(part.method == m) & (~part.trainfree)].test.iloc[0] for m in METHODS]
        free = [part[(part.method == m) & part.trainfree].test.iloc[0] for m in METHODS]
        x = np.arange(3)
        ax.bar(x - 0.18, trained, 0.36, label="Com treino padrão", color=BLUE)
        ax.bar(x + 0.18, free, 0.36, label="Sem treino", color=ORANGE)
        ax.set_xticks(x, labels)
        ax.set_ylim(50, 80)
        ax.set_title(f"{shot}-shot | média das 10 bases", fontsize=10, weight="bold")
        ax.grid(axis="y", alpha=0.2)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    axes[0].set_ylabel("Acurácia de teste (%)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    add_plot(pdf, fig, 52, 295, 738, 205)

    section_label(pdf, "Resultados médios das dez bases", 286)
    rows = [["Shot", "Método", "Condição", "Validação (%)", "Teste (%)", "Macro-F1 teste (%)", "Treino - sem treino (pp)"]]
    for shot in [1, 5]:
        for method in METHODS:
            pair = summary[(summary.shot == shot) & (summary.method == method)].set_index("trainfree")
            delta = pair.loc[False, "test"] - pair.loc[True, "test"]
            for trainfree in [False, True]:
                r = pair.loc[trainfree]
                rows.append([str(shot), METHODS[method], "Sem treino" if trainfree else "Com treino padrão",
                             fmt(r.val), fmt(r.test), fmt(r.f1), fmt(delta, True) if not trainfree else "-"])
    table(pdf, rows, [44, 100, 108, 100, 90, 112, 135], 36, 272, row_h=14, font_size=7.1)
    pdf.setFont("Helvetica-Bold", 10.2)
    pdf.drawString(36, 78, "Resultado interessante, mas testado somente em uma seed (ainda não testei em mais). Sem treino, utilizando somente")
    pdf.drawString(36, 64, "centróide das resnet padrão, vários resultados superam com treinamento. Próxima etapa seria testar com outros")
    pdf.drawString(36, 50, "descritores.")
    footer(pdf, 11)
    pdf.showPage()


def page_homofilia(pdf: canvas.Canvas, e1: pd.DataFrame, topo: pd.DataFrame) -> None:
    title(pdf, "6. Experimento de homofilia")
    config_table(pdf, ["10 bases Micro", "5-way; 1/5-shot; 10-query", "7", "512", "Sem treino; 2 propagações (K=2)", "40 + diagnóstico"])

    ng = e1[(e1.feat == "raw") & e1.trainfree & (e1.method == "no-graph")].set_index(["dataset", "shot"]).test
    base = topo[(topo.tag == "") & topo.method.isin(["union", "dgcg"])].copy()
    base["gain"] = [r.test - ng.loc[(r.dataset, r.shot)] for r in base.itertuples()]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.35))
    correlations = {}
    for ax, shot in zip(axes, [1, 5]):
        part = base[base.shot == shot]
        correlations[shot] = part.hom_query.corr(part.gain)
        for method, color in [("union", BLUE), ("dgcg", ORANGE)]:
            z = part[part.method == method]
            ax.scatter(z.hom_query, z.gain, s=35, color=color, label=METHODS[method])
        ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.set_title(f"{shot}-shot | r = {correlations[shot]:.2f}", fontsize=10, weight="bold")
        ax.set_xlabel("Homofilia nas linhas de query (%)")
        ax.set_ylabel("Grafo - sem grafo (pp)")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=7)
    fig.tight_layout()
    add_plot(pdf, fig, 50, 275, 742, 225)

    pdf.setFont("Helvetica-Bold", 8.3)
    pdf.drawString(36, 268, f"Correlação conjunta nos 20 pontos: r = {fmt(correlations[1])} em 1-shot e r = {fmt(correlations[5])} em 5-shot")
    section_label(pdf, "Resultados médios das dez bases", 252)
    rows = [["Shot", "Método", "Homofilia média (%)", "Teste (%)", "Sem grafo (%)", "Ganho vs. sem grafo (pp)"]]
    for shot in [1, 5]:
        no_graph_mean = ng.xs(shot, level="shot").mean()
        for method in ["union", "dgcg"]:
            z = base[(base.shot == shot) & (base.method == method)]
            rows.append([str(shot), METHODS[method], fmt(z.hom_query.mean()), fmt(z.test.mean()), fmt(no_graph_mean),
                         fmt(z.gain.mean(), True)])
    table(pdf, rows, [55, 125, 145, 110, 110, 145], 36, 239, row_h=16, font_size=7.3)

    section_label(pdf, "Definição usada", 144)
    definition = [["Métrica", "Cálculo", "Episódios"],
                  ["Homofilia das queries", "Fração do peso das arestas que liga imagens da mesma classe, sem autolaços", "200 por base e grafo"]]
    table(pdf, definition, [135, 425, 130], 36, 133, row_h=18, font_size=7.2)
    pdf.setFont("Helvetica-Bold", 10.2)
    pdf.drawString(40, 79, "aqui o correlation r significa correlação Pearson, a proxima etapa seria avaliar se a homofilia pode")
    pdf.drawString(40, 65, "influenciar utilizar grafo ou não. Falta ainda testar a homofilia com treinamento para ver se segue o")
    pdf.drawString(40, 51, "mesmo padrão")
    footer(pdf, 12)
    pdf.showPage()


def page_ponderacao(pdf: canvas.Canvas, e1: pd.DataFrame, topo: pd.DataFrame, trained: pd.DataFrame) -> None:
    title(pdf, "7. Experimento de ponderação das arestas")
    config_table(pdf, ["10 bases Micro", "5-way; 1/5-shot; 10-query", "7", "512 sem treino; 256 treinado", "Sem treino e confirmação treinada", "140"])

    choices = {
        "union": [("", "Original"), ("cosrbf", "RBF cosseno")],
        "dgcg": [("", "Original"), ("cosrbf", "RBF cosseno"), ("topo:wrbo", "RBO")],
    }
    records = []
    for method, tags in choices.items():
        for shot in [1, 5]:
            baseline = topo[(topo.method == method) & (topo.shot == shot) & (topo.tag == "")].test.mean()
            for tag, label in tags:
                z = topo[(topo.method == method) & (topo.shot == shot) & (topo.tag == tag)]
                records.append({"method": method, "shot": shot, "label": label, "test": z.test.mean(),
                                "gain": z.test.mean() - baseline})
    weighted = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.15), sharey=True)
    for ax, method in zip(axes, ["union", "dgcg"]):
        p = weighted[(weighted.method == method) & (weighted.label != "Original")]
        labels = list(dict.fromkeys(p.label))
        x = np.arange(len(labels))
        for offset, shot, color in [(-0.18, 1, BLUE), (0.18, 5, ORANGE)]:
            vals = [p[(p.label == label) & (p.shot == shot)].gain.iloc[0] for label in labels]
            ax.bar(x + offset, vals, 0.36, color=color, label=f"{shot}-shot")
        ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_title(METHODS[method], fontsize=10, weight="bold")
        ax.grid(axis="y", alpha=0.18)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Ganho sobre o grafo original (pp)")
    fig.tight_layout()
    add_plot(pdf, fig, 52, 320, 738, 185)

    section_label(pdf, "Ponderações sem treino - média das dez bases", 311)
    rows = [["Método", "Ponderação", "Teste 1-shot (%)", "Ganho 1-shot (pp)", "Teste 5-shot (%)", "Ganho 5-shot (pp)"]]
    for method in ["union", "dgcg"]:
        for label in list(dict.fromkeys(weighted[weighted.method == method].label)):
            p = weighted[(weighted.method == method) & (weighted.label == label)].set_index("shot")
            rows.append([METHODS[method], label, fmt(p.loc[1, "test"]), fmt(p.loc[1, "gain"], True),
                         fmt(p.loc[5, "test"]), fmt(p.loc[5, "gain"], True)])
    table(pdf, rows, [105, 120, 110, 110, 110, 110], 36, 297, row_h=14, font_size=7.1)

    section_label(pdf, "Confirmação com treino - somente RBF cosseno", 154)
    raw = e1[(e1.feat == "raw") & (~e1.trainfree)]
    rows2 = [["Shot", "Método", "Grafo treinado original (%)", "RBF cosseno treinado (%)", "Ganho (pp)"]]
    for shot in [1, 5]:
        for method in ["union", "dgcg"]:
            base = raw[(raw.method == method) & (raw.shot == shot)].test.mean()
            result = trained[(trained.method == method) & (trained.shot == shot)].test.mean()
            rows2.append([str(shot), METHODS[method], fmt(base), fmt(result), fmt(result - base, True)])
    table(pdf, rows2, [55, 125, 170, 170, 110], 36, 141, row_h=16, font_size=7.2)
    footer(pdf, 13)
    pdf.showPage()


def main() -> None:
    e1 = pd.read_json(DATA / "res_e1.jsonl", lines=True)
    topo = pd.read_json(DATA / "res_topo_tf.jsonl", lines=True)
    trained = pd.read_json(DATA / "res_topo_tr.jsonl", lines=True)

    stream = BytesIO()
    pdf = canvas.Canvas(stream, pagesize=(PAGE_W, PAGE_H))
    page_sem_treino(pdf, e1)
    page_homofilia(pdf, e1, topo)
    page_ponderacao(pdf, e1, topo, trained)
    pdf.save()
    stream.seek(0)

    writer = PdfWriter()
    source = PdfReader(SOURCE)
    appendix = PdfReader(stream)
    for page in source.pages:
        writer.add_page(page)
    for page in appendix.pages:
        writer.add_page(page)
    writer.add_metadata(source.metadata or {})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as handle:
        writer.write(handle)


if __name__ == "__main__":
    main()
