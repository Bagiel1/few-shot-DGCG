from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "resultados/relatorios/experimentos_1_a_4.pdf"
OUTPUT = ROOT / "output/pdf/experimentos_1_a_4_com_comentarios.pdf"


COMMENTS = {
    0: {
        "x": 38,
        "y": 100,
        "size": 14,
        "leading": 17,
        "lines": [
            "Lucas, aqui estão os resultados médios das 10 bases Micro em 27 protocolos",
            "(way,shot,query) padrão e sem ponderação",
        ],
    },
    1: {
        "x": 56,
        "y": 94,
        "size": 14,
        "leading": 17,
        "lines": [
            "Aqui são os resultados individuais por protocolo nas 10 bases",
        ],
    },
    2: {
        "x": 38,
        "y": 116,
        "size": 13.5,
        "leading": 17,
        "lines": [
            "Para cada protocolo de way-shot-query foram executados 4 diferentes intervalos de grau",
            "para o DGCG sendo uma média das 10 bases",
            "",
            "Para episodios com menos de 50 nós, o grau 2-4 geralmente é melhor, mas ao crescer o",
            "número de nós o grau 4-6 se torna melhor",
        ],
    },
    3: {
        "x": 38,
        "y": 48,
        "size": 13.5,
        "leading": 17,
        "lines": [
            "A mesma coisa da busca com grau, mas dessa vez a busca pela densidade especifica, nas 10",
            "bases e 27 protocolos de way-shot-query, no final compara com o resultado do DGCG 4-6",
        ],
    },
    4: {
        "x": 62,
        "y": 174,
        "size": 11.5,
        "leading": 14.5,
        "lines": [
            "Esse experimento é um pouco mais complicado de explicar.",
            "Para cada protocolo de way-shot-query dito, é realizado os treinos e validações alterando o",
            "grau alvo (2-4, 4-6, 6-8, 8-12), depois das 200 validações, seleciona o intervalo que teve a",
            "maior validação e calcula o teste.",
            "Isso é feito também para a densidade variando de 0,025 a 0,90. Como essa busca por",
            "densidade era mais específica e possui mais valores, para episodios pequenos teve",
            "vantagem.",
            "Estou fazendo a busca da densidade não de forma binária, mas uma busca exata.",
        ],
    },
    5: {
        "x": 92,
        "y": 350,
        "size": 10.5,
        "leading": 13,
        "lines": [
            "A coluna azul é a melhor configuração de extras - a mesma configuração sem extras e a laranja é em relação ao",
            "nograph",
        ],
        "second": {
            "x": 58,
            "y": 196,
            "size": 11.5,
            "leading": 14,
            "lines": [
                "Somente os resultados de controle para comparação com as mesmas configurações",
            ],
        },
    },
    6: {
        "x": 62,
        "y": 228,
        "size": 11.5,
        "leading": 14,
        "lines": [
            "Esses são os melhores resultados do experimento com imagens extras",
            "Utilizei imagens extras de duas formas - Somente na criação das listas ranqueadas (sem",
            "propagar no grafo) ou na criação das listas ranqueadas e na participação da propagação do",
            "grafo. A segunda forma foi a melhor configuração, portanto na tabela de cima so temos o",
            "modo propagation para não lotar de resultados.",
            "",
            "Além disso, temos a fase que utilizei as imagens extras, testei 3 formas: A 1 sendo imagens",
            "extras somente no treino, a 2 sendo imagens extras somente no treino e validação e a 3 no",
            "treino, validação e teste (all). A 3 forma também foi a melhor sendo so mostrado a fase all.",
            "Testei com 5, 10 e 20 imagens extras também.",
        ],
    },
    7: {
        "x": 64,
        "y": 170,
        "size": 10.8,
        "leading": 14,
        "lines": [
            "Aqui é a comparação dos métodos em bases maiores flowers e cars micro, mini e extended.",
            "Importante ressaltar que mesmo a base sendo maior, quem dita o tamanho do episódio é o way-shot-query, portanto",
            "os episódios são do mesmo tamanho.",
            "",
            "Importante ressaltar também que as densidades do DGCG utilizadas foram calculadas fazendo aquela busca de 15",
            "densidades no Micro da Flowers e do Cars e essa a melhor densidade foi utilizada no Mini e Extended.",
            "",
            "Não é muito consistente pois na flowers DGCG ganha as vezes, porém no Cars não.",
        ],
    },
}


def draw_block(pdf: canvas.Canvas, block: dict) -> None:
    pdf.setFont("Helvetica-Bold", block["size"])
    y = block["y"]
    for line in block["lines"]:
        if line:
            pdf.drawString(block["x"], y, line)
        y -= block["leading"]


def overlay_for_page(width: float, height: float, block: dict) -> PdfReader:
    stream = BytesIO()
    pdf = canvas.Canvas(stream, pagesize=(width, height))
    draw_block(pdf, block)
    if "second" in block:
        draw_block(pdf, block["second"])
    pdf.save()
    stream.seek(0)
    return PdfReader(stream)


def main() -> None:
    reader = PdfReader(SOURCE)
    writer = PdfWriter()
    for index, page in enumerate(reader.pages):
        if index in COMMENTS:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            overlay = overlay_for_page(width, height, COMMENTS[index])
            page.merge_page(overlay.pages[0])
        writer.add_page(page)
    writer.add_metadata(reader.metadata or {})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as stream:
        writer.write(stream)


if __name__ == "__main__":
    main()
