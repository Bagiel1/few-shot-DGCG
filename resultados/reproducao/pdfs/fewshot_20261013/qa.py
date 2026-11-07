from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

root = Path(__file__).resolve().parent
pages = sorted(root.glob('pagina-*.png'))
for group in range(3):
    sheet = Image.new('RGB', (1560, 1560), '#dddddd')
    for index, path in enumerate(pages[group*12:(group+1)*12]):
        page = Image.open(path).convert('RGB')
        page.thumbnail((510, 360))
        x, y = (index%3)*520+5, (index//3)*390+25
        sheet.paste(page, (x, y))
        ImageDraw.Draw(sheet).text((x, y-18), path.stem, fill='black')
    sheet.save(root / f'contato-{group+1}.png')
