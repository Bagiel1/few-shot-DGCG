from pathlib import Path
from PIL import Image,ImageDraw
from pypdf import PdfReader

folder=Path(__file__).resolve().parent
pages=sorted(folder.glob('pagina-*.png'))
assert len(pages)==59,len(pages)
for offset in range(0,len(pages),12):
    sheet=Image.new('RGB',(1560,1560),'#dddddd')
    for index,path in enumerate(pages[offset:offset+12]):
        page=Image.open(path).convert('RGB');page.thumbnail((510,360))
        x=(index%3)*520+5;y=(index//3)*390+25
        sheet.paste(page,(x,y));ImageDraw.Draw(sheet).text((x,y-18),path.stem,fill='black')
    sheet.save(folder/f'contato-{offset//12+1}.png')
pdf=folder.parents[2]/'output/pdf/relatorio_fewshot_2026-10-26.pdf'
reader=PdfReader(pdf)
assert len(reader.pages)==59
for index,page in enumerate(reader.pages):
    text=page.extract_text()
    assert '26/10/2026' in text
    assert str(index+1) in text
for index in [1,8,9]:
    im=Image.open(folder/f'detalhe-{index:02}.png').convert('L')
    bbox=im.point(lambda v:0 if v>180 else 255).getbbox()
    print(index,im.size,bbox)
print('Paginas e montagens verificadas.')
