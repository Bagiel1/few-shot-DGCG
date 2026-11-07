from pathlib import Path
from xml.sax.saxutils import escape
import json,csv
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
ROOT=Path.cwd(); F=ROOT/'tmp/pdfs/explicacao_20261103/figuras'; OUT=ROOT/'resultados/relatorios/entendendo_resultados_fewshot.pdf'
for n,file in [('D','DejaVuSans.ttf'),('DB','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(n,'/usr/share/fonts/truetype/dejavu/'+file))
navy=colors.HexColor('#163449');teal=colors.HexColor('#16857c');light=colors.HexColor('#edf4f5')
styles={
 'title':ParagraphStyle('title',fontName='DB',fontSize=23,leading=28,textColor=navy,spaceAfter=12),
 'sub':ParagraphStyle('sub',fontName='DB',fontSize=11,leading=15,textColor=teal,spaceAfter=8),
 'body':ParagraphStyle('body',fontName='D',fontSize=10,leading=15,spaceAfter=9,textColor=navy),
 'small':ParagraphStyle('small',fontName='D',fontSize=8,leading=11,spaceAfter=6,textColor=navy),
 'cell':ParagraphStyle('cell',fontName='D',fontSize=8,leading=11,textColor=navy),
 'note':ParagraphStyle('note',fontName='DB',fontSize=11,leading=16,textColor=teal,spaceAfter=12),
}
story=[]; pages=[]
def p(t,style='body'):return Paragraph(t,styles[style])
def text(t):story.append(p(t))
def title(n,t,kicker):
 if story:story.append(PageBreak())
 pages.append(t);story.extend([p(f'{n:02d}  /  {kicker.upper()}','sub'),p(t,'title')])
def fig(name,h=None):
 from PIL import Image as PI
 f=F/(name+'.png');w,hh=PI.open(f).size;wi=495;he=wi*hh/w
 if h and he>h:wi*=h/he;he=h
 story.extend([Image(str(f),width=wi,height=he),Spacer(1,5)])
def note(t):story.append(p(t,'note'))
def tab(rows,widths=None):
 cells=[[p(str(c),'cell') for c in row] for row in rows];t=Table(cells,colWidths=widths or [495/len(rows[0])]*len(rows[0]),hAlign='LEFT',repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),light),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),1,teal),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce4e8'))]));story.extend([t,Spacer(1,10)])
def label(t):story.append(p(t,'sub'))
# 1
title(1,'O que está acontecendo com o few-shot?','Guia de leitura dos experimentos')
note('O treino não falhou em geral; o efeito muda com o método, a base e a forma de iniciar a representação..')
text('Gabriel • análise dos arquivos disponíveis em 03/11/2026.. Este guia reúne os testes recentes do Claude e os conecta ao histórico, às escalas e aos extras.. O objetivo é entender os resultados e separar o que foi medido do que ainda é hipótese..')
tab([['Pergunta','Resposta que os dados sustentam'],['Os novos resultados batem com os antigos?','Sim, aproximadamente nas configurações correspondentes; há diferenças pontuais de até 1,13 pp nos grafos DGCG..'],['Treinar piora?','O no-graph padrão perde na média; os grafos ganham com treino.. Com identidade 512 + treino, o no-graph melhora na média sobre identidade sem treino..'],['O grafo faz sentido?','No 1-shot, sim, com ganho médio expressivo; no 5-shot, vários grafos prejudicam uma representação já forte..'],['O adaptativo é a solução?','Recupera o prejuízo da propagação, mas ainda não demonstra ganho sobre o baseline sem grafo..'],['Qual caminho parece promissor?','Fortalecer os controles e medir utilidade da propagação, sem compromisso de usar DGCG..']],[145,350])
text('<b>Roteiro:</b> páginas 2–4 explicam protocolo e reprodução; 5–9 tratam de treino e representação; 10–13 de grafos; 14–17 de homofilia e adaptação; 18–19 retomam escalas e extras; 20–23 organizam hipóteses, próximos testes e fontes..')
text('<b>Como ler números:</b> 76,10% é acurácia; passar de 73,45% para 76,10% é ganhar 2,65 pontos percentuais (pp), não 2,65% relativo.. As médias dos testes recentes dão o mesmo peso a cada uma das dez bases..')
#2
title(2,'Antes dos resultados: qual problema foi testado?','Protocolo')
tab([['Peça','Significado neste projeto'],['5-way','Cada episódio contém cinco classes; acertar ao acaso daria 20% de acurácia..'],['1-shot / 5-shot','Uma ou cinco imagens rotuladas de suporte por classe..'],['10-query','Dez imagens a classificar por classe: 50 queries por episódio..'],['Grafo transdutivo','Suportes e queries participam do mesmo grafo; os rótulos das queries não são usados na predição..'],['55 / 75 nós','5 × (1 + 10) = 55; 5 × (5 + 10) = 75.. Sem extras..'],['Micro','20 classes, 40 imagens por classe; split desta bateria: 10 classes de treino, 5 de validação e 5 de teste..']],[140,355])
label('O caminho de uma previsão')
text('Imagem → ResNet-18 pré-treinada em ImageNet e congelada → vetor de 512 valores → eventual pré-processamento → grafo e propagação → projeção Θ → LayerNorm → normalização L2 → comparação com o protótipo de cada classe..')
text('O protótipo é a média das representações dos suportes daquela classe.. A query recebe a classe com menor distância ao protótipo.. O grafo pode ajudar trazendo contexto de outras imagens, mas uma ligação errada também pode aproximar classes diferentes..')
text('<b>O que é treinado:</b> a projeção Θ, os parâmetros afins da LayerNorm e a escala dos logits.. A ResNet fica congelada.. Portanto, este é um treino episódico de uma cabeça sobre features prontas; não é um teste de toda forma possível de meta-aprendizado ou de fine-tuning da rede..')
note('Mais dados no conjunto não significam mais nós no episódio.. O orçamento do episódio é uma escolha separada..')
#3
title(3,'O que terminou — e o que não chegou a rodar','Inventário auditável')
counts=json.loads((ROOT/'outputs/explicacao_fewshot_20261103/estatisticas.json').read_text())['counts']
tab([['Arquivo','Linhas','O que compara'],['res_e1.jsonl',360,'3 métodos × 2 shots × 3 pré-processamentos × com/sem treino × 10 bases'],['res_dgcg24.jsonl',80,'DGCG 2–4, raw/CL2N, com/sem treino'],['res_idinit.jsonl',60,'Treino iniciado na identidade 512, três métodos'],['res_proptf.jsonl',200,'Saltos e mistura com identidade, sem treino'],['res_topo_tf.jsonl',360,'18 configurações de topologia, dois shots'],['res_topo_tr.jsonl',40,'RBF cosseno com treino: union e DGCG'],['res_adapt.jsonl',40,'Dois grafos × dois pares de limiares, 5-shot'],['res_supproxy.json',20,'Diagnóstico por base/método, não treino de modelos']],[145,45,305])
text('<b>São 1.240 linhas de resultados e 20 registros de diagnóstico nesta fotografia dos arquivos..</b> Algumas linhas repetem controles entre arquivos; isso não representa 1.240 repetições independentes.. Todos os novos resultados usam seed 7, ResNet-18 a 128 px e CPU, conforme configuração e origem informadas..')
text('O RBF cosseno com treino e o adaptativo estão completos para suas grades efetivamente executadas.. RBO² com treino e DGCG+ com treino não rodaram: o filtro da grade não correspondeu às variantes planejadas.. Ausência de resultado não é derrota do método..')
text('Treino padrão: 2.000 episódios; validação com 200 episódios a cada 200; teste com 1.000 episódios.. Os resultados de escalas, à parte, têm três seeds: 7, 17 e 27..')
#4
title(4,'CPU local e GPU da VM: os resultados se reconhecem','Reprodução do histórico')
fig('hardware')
text('<b>Como ler:</b> à esquerda, cada ponto é uma configuração correspondente; a diagonal representa resultados idênticos.. À direita, zero significa ausência de diferença entre Claude e histórico.. Cada método reúne 20 pares: dez bases × dois shots..')
text('Nos 80 pares, a diferença absoluta média é <b>0,119 pp</b>.. As maiores diferenças absolutas são 0,072 pp no no-graph, 0,078 no union, 1,134 no DGCG 4–6 e 1,014 no DGCG 2–4.. As conclusões gerais são compatíveis, mas não há igualdade numérica completa..')
text('<b>Hipótese:</b> diferenças numéricas, bibliotecas e desempates em rankings podem afetar mais um grafo discreto.. Porém, CPU/GPU não foi isolado experimentalmente; não podemos atribuir toda diferença ao dispositivo.. Isso exigiria mesmas features, episódios, código e configuração em ambos os ambientes..')
note('O contraste “com treino × sem treino” do Claude ocorre dentro da CPU.. A troca de dispositivo não explica esse contraste interno..')
#5
title(5,'Treinar ajuda o grafo, mas piora o no-graph padrão','Visão geral')
fig('treino')
text('<b>Como ler:</b> compare as duas barras do mesmo método, dentro do mesmo shot.. O eixo começa em 50% para facilitar a comparação; diferenças devem ser lidas pelos valores, não pela proporção visual da altura..')
text('No 1-shot, o treino padrão muda o no-graph em <b>−0,75 pp</b>, union em <b>+1,12 pp</b> e DGCG 4–6 em <b>+0,71 pp</b>.. No 5-shot, os efeitos são <b>−2,45 pp</b>, <b>+1,39 pp</b> e <b>+1,30 pp</b>, respectivamente.. Portanto, “treino não ajuda” não descreve os três métodos..')
text('Sem treino, union ganha 4,62 pp sobre no-graph no 1-shot, mas perde 2,65 pp no 5-shot.. Com um único suporte por classe, contexto adicional pode ajudar a estimar a classe; com cinco, o protótipo inicial é mais bem estimado e a mistura pode trazer mais erro do que informação útil.. Essa explicação é plausível, não uma causa já demonstrada..')
note('A conclusão válida é sobre a configuração testada.. A comparação padrão ainda mistura efeito do treino, dimensão e inicialização..')
#6
title(6,'1-shot: a média esconde ganhos e perdas grandes','Resultados por base')
fig('treino_base1')
text('<b>Como ler:</b> cada célula é acurácia com treino padrão menos acurácia sem treino.. Azul/positivo favorece treino; vermelho/negativo favorece sem treino.. A comparação mantém base, grafo, protocolo e pré-processamento raw..')
text('Textures recebe um benefício muito maior do treino do que a maioria das outras bases.. Esse ganho compensa várias quedas na média.. Assim, uma média perto de zero não significa que treinar quase não muda cada problema..')
text('<b>Hipótese:</b> a geometria das features ImageNet já separa bem algumas bases, mas precisa de adaptação em outras.. A cabeça pode aprender uma métrica útil para Textures e, em outras bases, perder propriedades que transferiam para classes novas.. Precisamos de curvas de treino/validação e controles de inicialização para separar essas possibilidades..')
#7
title(7,'5-shot: o baseline sem treino fica difícil de superar','Resultados por base')
fig('treino_base5')
text('<b>Como ler:</b> o mesmo contraste da página anterior, agora com cinco suportes por classe.. Observe a coluna sem grafo antes de comparar métodos diferentes..')
text('Há quedas fortes no no-graph padrão em várias bases e ganho grande em Textures.. Isso torna inadequado defender um único número médio como descrição de todos os domínios.. Nos grafos, o treino frequentemente recupera parte do dano da propagação, sem necessariamente superar o baseline forte sem grafo..')
text('<b>O que ainda não sabemos:</b> se a queda vem principalmente de compressão para 256 dimensões, de inicialização aleatória, de especialização às classes de treino, de regularização ou de seleção de checkpoint.. O próximo experimento já dá uma pista importante, mas não resolve todos esses fatores..')
#8
title(8,'A identidade muda a história do treino','Controle de representação')
fig('init')
text('<b>Três condições:</b> identidade 512 sem treino; projeção aleatória 256 com treino padrão; identidade 512 com treino.. No-graph passa de 61,08 para <b>62,85%</b> no 1-shot e de 76,10 para <b>76,29%</b> no 5-shot ao treinar a condição iniciada na identidade..')
text('Isso enfraquece bastante a interpretação de que “aprender é o problema”.. Preservar o ponto de partida parece importante.. A comparação identidade 512 com/sem treino é mais direta; já identidade 512 versus aleatória 256 muda dois fatores simultaneamente.. Separar dimensão de inicialização exige aleatória 512 + treino e um controle adequado de projeção 256..')
text('<b>Detalhe essencial:</b> “sem treino” ainda aplica LayerNorm inicializada e normalização L2.. O rótulo raw significa ausência de CL2N/PT antes da cabeça, não ausência de qualquer transformação.. Aqui o DGCG é <b>2–4</b>, pois é a variante executada no estudo de identidade..')
#9
title(9,'Há dois cuidados escondidos no pipeline','Implementação e pré-processamento')
fig('preproc_False',175);fig('preproc_True',175)
text('<b>CL2N:</b> subtrai o centro calculado nas features de treino e normaliza cada vetor.. <b>PT+CL2N:</b> aplica raiz quadrada aos valores não negativos antes de centralizar/normalizar.. Nos gráficos, zero é o raw do mesmo método e shot.. Melhorias pequenas ou dependentes da condição não justificam dizer que uma transformação vence sempre..')
text('<b>Checkpoint inicial:</b> o código começa com melhor validação = −∞ e só avalia depois de treinar.. Assim, o estado inicial não concorre com os checkpoints treinados.. “Melhor checkpoint” significa melhor entre os checkpoints avaliados, não melhor que o ponto de partida.. Avaliar o episódio zero é um controle prioritário..')
text('<b>O que o otimizador altera:</b> Θ, escala/deslocamento da LayerNorm e escala dos logits.. A escala positiva dos logits não muda o argmax com distâncias fixas, mas participa dos gradientes.. Atribuir toda melhora ou queda apenas a Θ seria incompleto..')
#10
title(10,'DGCG 2–4 ou 4–6: depende do tamanho do episódio','Quantidade de conexões')
fig('degrees')
text('<b>Como ler:</b> cada grupo mantém shot e condição de treino; só muda a faixa de grau do DGCG.. Aqui os episódios têm 55 ou 75 nós.. Com treino, 4–6 supera 2–4 na média: 64,87 versus 63,61% no 1-shot; 75,14 versus 74,12% no 5-shot..')
text('Isso não contradiz o histórico em que faixas menores podiam ser melhores em protocolos muito pequenos.. Uma conexão extra em um grafo com poucos nós muda muito sua densidade; em um episódio maior, uma faixa muito baixa pode limitar a informação que circula..')
text('<b>Limite da conclusão:</b> o resultado apoia dependência do protocolo, não uma lei de que grau ideal cresce sempre com o número de nós.. Shot, número de classes, qualidade das features e construção do grafo também mudam a tarefa.. Densidade escolhida por validação é outra configuração; não deve ser confundida com DGCG fixo 4–6..')
note('Escolher a melhor faixa por teste seria otimista.. A escolha de hiperparâmetros deve continuar na validação..')
#11
title(11,'Mais propagação nem sempre traz mais informação','Saltos e mistura com identidade')
fig('prop')
text('<b>K</b> é o número de saltos.. Em K=2, cada nó pode receber informação por caminhos de dois passos.. A mistura usa A efetiva = (1−α)Â + αI antes dos saltos; α=0,5 preserva mais a própria representação.. Não é uma média final entre duas classificações prontas..')
text('No union 5-shot, aumentar de dois para quatro saltos reduz a média de <b>73,45 para 71,87%</b>.. Usar α=0,5 com dois saltos recupera para <b>75,24%</b>, ainda abaixo do no-graph 76,10%.. No 1-shot, as mudanças são bem menores na média..')
text('<b>Hipótese coerente:</b> mistura excessiva pode apagar distinções entre classes ou espalhar ligações erradas.. A queda com mais saltos e a recuperação com identidade são evidências compatíveis com esse mecanismo, mas não o comprovam sozinhas.. Para testar, medir separação intra/interclasse e erros por query antes e depois de cada salto..')
#12
title(12,'Qualidade da aresta: o que cada intervenção fez','Topologias sem treino')
fig('topo',345)
text('<b>Como ler:</b> diferenças contra o próprio grafo original, em raw e sem treino.. Azul significa recuperação; não significa superar o no-graph.. DGCG+ é comparado ao DGCG 4–6..')
text('<b>Pesos:</b> RBO/RBO² usam concordância de rankings; SNN usa vizinhos compartilhados; cosrbf usa similaridade cosseno convertida em peso.. <b>Estrutura:</b> sym une sentidos, mut mantém reciprocidade, prune remove ligações com RBO baixo.. <b>Suportes:</b> sup substitui seu bloco por ligações de mesma classe, supcut apenas remove ligações entre classes distintas nesse bloco.. DGCG+ e GRaNDe são variantes adicionais do construtor avaliadas pelo harness..')
text('O sinal mais consistente é pesar conexões no DGCG 5-shot: cosrbf e RBO² recuperam aproximadamente 1 pp; DGCG+ chega a +1,57 pp, mas sua média 75,42% ainda fica abaixo de 76,10% sem grafo.. A poda melhora cerca de 0,43 pp no 5-shot e piora no 1-shot: dizer que nunca ajuda seria excessivo..')
#13
title(13,'Com treino, os pesos cosseno continuam ajudando pouco','Confirmação parcial das variantes')
fig('topotr')
text('<b>Como ler:</b> compare RBF cosseno treinado ao mesmo grafo treinado sem esse peso.. São 40 execuções, não apenas o recorte parcial inicial do README..')
text('No 1-shot, o ganho médio é <b>+0,09 pp no union</b> e <b>+0,40 pp no DGCG</b>, com melhora em 6/10 e 4/10 bases.. No 5-shot, os ganhos sobem para <b>+0,94 e +0,67 pp</b>, com 7/10 bases melhores em ambos.. A distribuição por base importa: ganho médio positivo não equivale a melhora universal..')
text('A média final 5-shot é 75,78% no union e 75,81% no DGCG.. Essas condições ainda não superam a média 76,10% do no-graph sem treino.. Isso sugere benefício incremental dos pesos, sem sustentar que resolvem o problema geral..')
note('RBO² treinado e DGCG+ treinado não foram executados.. Não há conclusão sobre eles nesta bateria..')
#14
title(14,'Homofilia explica parte do padrão, não toda a utilidade','Diagnóstico das arestas')
fig('hom')
text('<b>Homofilia</b> é a fração do peso das arestas que liga imagens da mesma classe, sem autolaços.. Aqui usamos as linhas das queries.. Cada ponto combina uma base e um grafo; são 20 pontos por shot.. A correlação com ganho do grafo é <b>0,65 no 1-shot e 0,86 no 5-shot</b>..')
text('Bases cujas conexões respeitam melhor as classes tendem a sofrer menos ou ganhar mais com propagação.. Mas os dois métodos na mesma base não são observações independentes, e correlação não demonstra causalidade.. DGCG pode ter homofilia maior e acurácia menor que union: alcance, direção, grau e quais nós são conectados também importam..')
text('<b>Dois cuidados:</b> homofilia usa rótulos de query apenas para diagnóstico e foi medida em 200 episódios separados dos 1.000 de acurácia.. Não é uma correlação por episódio.. Além disso, Cars com cerca de 34–39% está acima do acaso de ligações (~19%); chamar isso simplesmente de “quase aleatório” seria impreciso..')
#15
title(15,'O proxy é excelente entre bases e fraco por episódio','O ponto mais importante para a hipótese adaptativa')
fig('proxy')
text('<b>À esquerda:</b> uma média por base.. Bases com suportes bem conectados também têm queries bem conectadas: r≈0,998 no union e 0,999 no DGCG.. <b>À direita:</b> a correlação entre os episódios de cada base, já calculada nos arquivos.. Sua média é <b>−0,059 no union</b> e <b>0,088 no DGCG</b>.. São 150 episódios por base/método neste diagnóstico..')
text('Analogia: saber que uma cidade é mais quente que outra não garante prever a temperatura de cada hora do dia.. Aqui, o proxy identifica características gerais das bases, mas quase não acompanha a variação de qualidade entre episódios da mesma base..')
text('<b>Consequência:</b> ainda não temos evidência forte de que a regra identifica precisamente quando cada episódio merece mais propagação.. Ela pode estar funcionando principalmente como uma redução média da intensidade do grafo.. Isso deve ser comparado a um peso fixo calibrado na validação e a pesos embaralhados entre episódios da mesma base..')
#16
title(16,'O adaptativo recuperou o dano — e chegou ao empate','Resultado final nas dez bases')
fig('adapt')
text('A regra calcula a homofilia entre suportes do episódio: h.. Depois usa <b>w = clip((h−lo)/(hi−lo), 0, 1)</b> e mistura <b>wÂ + (1−w)I</b> antes dos saltos.. h baixo reduz propagação; h alto a mantém.. Só usa rótulos dos suportes, disponíveis no problema..')
text('O union original tinha 73,45%.. Os dois limiares chegam a 75,84 e 76,10%.. O DGCG sai de 73,84% para 75,72 e 76,00%.. O controle sem grafo é <b>76,0964%</b>; union com 0,45/0,85 dá <b>76,1018%</b>, diferença de apenas <b>+0,0054 pp</b>.. Arredondar e chamar isso de vitória seria enganoso..')
text('Os dois pares de limiares foram definidos a priori segundo o registro do experimento, não selecionados pela validação.. Mostramos os dois; escolher agora o melhor pelo teste introduziria viés.. O resultado sustenta recuperação, mas não superioridade..')
text('<b>1-shot:</b> não há dois suportes da mesma classe.. Esse proxy não mede conexões positivas; o código volta a w=1.. A regra precisa de outro sinal para atuar nesse regime..')
#17
title(17,'O empate médio também esconde quem ganha e perde','Adaptativo por base')
fig('adaptbase')
text('<b>Como ler:</b> cada célula compara a regra adaptativa com no-graph sem treino na mesma base.. Assim, o gráfico responde à pergunta de pesquisa mais exigente: usar o grafo acrescentou valor ao baseline forte?')
text('Union 0,45/0,85 ganha em 5/10 bases.. Sports ganha <b>0,94 pp</b>, Manufacturing perde <b>1,10 pp</b>; várias outras diferenças ficam perto de zero.. Não há vitória generalizada, e uma seed não permite tratar diferenças pequenas como estáveis..')
text('<b>Não confundir melhora relativa:</b> ganhar aproximadamente 2,65 pp sobre um grafo prejudicial é útil para controle de dano.. Para defender um método novo, falta mostrar que ele adiciona informação além de preservar a representação original.. Uma regra fixa simples pode conseguir resultado semelhante: esse controle ainda falta..')
#18
title(18,'Micro, Mini e Extended: o grafo continua do mesmo tamanho','Ligação com nossos testes na VM')
fig('scales',345)
text('<b>Como ler:</b> cada ponto é média de três seeds; barras mostram desvio entre seeds, não intervalo de confiança da média.. Dentro de cada painel, compare métodos na mesma versão.. Os alvos de densidade do DGCG foram selecionados no Micro e mantidos nas demais versões..')
text('Aumentar a base amplia as classes/imagens disponíveis para amostrar episódios, mas o grafo continua com 55 ou 75 nós e o orçamento continua com 2.000 episódios de treino.. Portanto, esses testes medem transferência da configuração para outra versão, não escalabilidade para grafos maiores..')
text('Micro → Mini também muda classes de avaliação; a seed controla split, inicialização e episódios.. Não atribua toda mudança a “mais imagens”.. Esses testes usam a cabeça treinada padrão, enquanto o baseline forte identidade sem treino foi estudado nas bases Micro.. Ainda falta levá-lo às versões maiores..')
#19
title(19,'Extras: muitos testes, pouco sinal estável','Por que esse caminho perdeu prioridade')
fig('extras')
text('<b>Como ler:</b> cada curva representa método/faixa e modo de extras; o valor é ganho sobre seu controle sem extras, médio nas bases e seis protocolos da bateria de 4.740 execuções.. Este gráfico usa extras presentes em todas as fases e agrupa protocolos; não mostra uma curva individual por base..')
text('No DGCG, extras podem entrar apenas no contexto dos rankings ou também na propagação.. Nos kNN, o uso exibido é na propagação.. São intervenções diferentes: alterar rankings não equivale a colocar todos os extras como nós no grafo de classificação..')
text('Os ganhos médios das configurações DGCG/union ficam na ordem de décimos de pp ao agregar as quantidades, sem um crescimento monotônico convincente.. Isso combina com sua decisão de não priorizar extras por enquanto.. Não prova que nunca funcionam; mostra que a busca realizada não trouxe um avanço amplo e simples..')
text('Controles com banco reservado precisam ser comparados dentro da mesma bateria.. Experimentos antigos com contexto global no teste mudam a informação disponível e não devem ser tratados como vitória sob o orçamento original.. A análise detalhada das baterias anteriores permanece no relatório principal..')
#20
title(20,'Quais explicações fazem sentido?','Fatos, hipóteses e como distinguir')
tab([['Hipótese','Evidência compatível','Teste que falta'],['Ponto de partida ruim ou compressão prejudicial','Identidade 512 + treino supera o treino padrão e recupera o no-graph..','Comparar identidade 512, aleatória 512 e projeção 256; manter episódios e orçamento iguais..'],['Especialização às classes de treino prejudica transferência','No-graph perde em várias bases; Textures tem ganho grande..','Curvas de treino e validação por base; distância entre classes antes/depois; variação de regularização..'],['Propagação excessiva mistura classes','Mais saltos pioram 5-shot; identidade parcial recupera..','Medir separação intra/interclasse, mudança por query e alcance dos suportes por salto..'],['Arestas melhor ponderadas preservam informação','RBO²/cosseno ajudam DGCG 5-shot; ganhos modestos..','Comparar pesos mantendo estrutura/grau e orçamento; repetir seeds..'],['Adaptativo só regula intensidade média','Empata com no-graph; proxy fraco dentro da base..','Peso fixo validado, peso adaptativo e pesos embaralhados dentro da mesma base..'],['Diferença CPU/GPU causa divergências pontuais','DGCG diverge mais que no-graph/union..','Rodar ambos com cache, episódios, versões e código idênticos; comparar adjacências..']],[125,180,190])
note('Nenhuma dessas hipóteses exige que o trabalho use DGCG.. A questão científica pode ser quando e por que a propagação ajuda no few-shot..')
text('“Aresta correta” e “aresta útil” não são sinônimos.. Uma aresta entre imagens da mesma classe pode ser redundante; remover uma aresta pode alterar normalização, caminhos e predições de vários nós.. Um diagnóstico de utilidade deve medir mudança no erro ou na margem da query, não apenas igualdade de rótulos..')
#21
title(21,'O próximo teste deve eliminar uma dúvida de cada vez','Plano mínimo com prioridade')
tab([['Prioridade','Experimento','Decisão que ele permite'],['1 • baseline','Mesmas seeds/episódios: sem grafo identidade 512 sem treino; identidade 512 com treino; aleatória 512 com treino; padrão 256.. Incluir estado inicial na validação..','Distinguir ganho do aprendizado de efeito da representação inicial e da dimensão..'],['2 • adaptação','Comparar w=0, w=1, grade curta de w fixo validada, adaptativo e w adaptativo embaralhado dentro da base..','Ver se o proxy traz informação por episódio ou apenas reduz propagação..'],['3 • repetição','Seeds 7/17/27 nos controles escolhidos; depois Flowers/Cars Mini e Extended com baseline forte..','Ver estabilidade e se o padrão sobrevive além de Micro..'],['4 • diagnóstico','Salvar predição/margem por query antes/depois do grafo, erros corrigidos e criados, homofilia e alcance até suportes..','Separar aresta semelhante de conexão que realmente ajuda a decisão..']],[80,255,160])
text('<b>Regra de avaliação:</b> configurar na validação, congelar a escolha e só então avaliar teste.. Registrar seeds, classes, episódios, features e dispositivo.. Para comparar erros, salvar resultados por episódio compartilhado; o resumo agregado atual não permite reconstruir a incerteza de todas as diferenças pareadas..')
text('<b>Experimento de hipótese, não método final:</b> usar rótulos de query para diagnosticar quais arestas corrigem/criam erros pode revelar mecanismos.. Esses rótulos não podem entrar na regra usada em teste.. Um “oráculo” serve de teto diagnóstico, não de resultado competitivo válido..')
note('Não executei novos treinos para produzir este relatório.. A prioridade é tornar a próxima rodada mais informativa, não aumentar a grade sem uma pergunta clara..')
#22
title(22,'Isso faz sentido em relação à literatura?','Interpretação e limites estatísticos')
text('Não há uma regra de que treinar uma cabeça sempre melhora classes novas.. O resultado depende da representação inicial, do objetivo, do orçamento e do domínio.. Ao mesmo tempo, estes arquivos não sustentam a conclusão geral de que treino ou meta-aprendizado sejam desnecessários.. Aqui, identidade + treino já melhora o no-graph na média, e os grafos também se beneficiam do treino padrão..')
text('<b>Referência próxima:</b> Hu et al., CVPR 2022, <i>Pushing the Limits of Simple Pipelines for Few-Shot Learning</i>, estuda pipelines simples e representações pré-treinadas fortes, com ganhos adicionais por adaptação.. Serve para justificar um baseline forte, não para afirmar que fine-tuning costuma piorar.. <link href="https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Pushing_the_Limits_of_Simple_Pipelines_for_Few-Shot_Learning_External_CVPR_2022_paper.pdf" color="#16857c">Artigo no CVF</link>..')
text('<b>Exemplo de tensão entre adaptação e generalização:</b> <i>Parameter-Efficient Fine-Tuning via Meta-Regularizer</i>, IJCV 2026, discute desempenho de adaptação de prompts e generalização para classes novas.. É outro cenário, com CLIP; ajuda a entender que o fenômeno é possível, mas não demonstra a causa dos nossos resultados com ResNet.. <link href="https://link.springer.com/article/10.1007/s11263-025-02693-z" color="#16857c">Artigo na Springer</link>..')
label('O que nossos números ainda não autorizam')
text('Mil episódios estimam desempenho de uma execução no seu split; não equivalem a mil seeds de treinamento.. Os ICs episódicos não incorporam toda a variação de split, inicialização, escolha de hiperparâmetros ou dispositivo.. Uma diferença de 0,01 pp não vira evidência de superioridade só porque há muitos episódios..')
text('As dez bases também não tornam todos os resultados independentes: os mesmos dados reaparecem em métodos e variantes.. Os gráficos deste guia mostram padrões descritivos, sem testes de significância ou barras inventadas para diferenças que não têm resultados pareados disponíveis..')
note('Minha leitura: os resultados são coerentes, mas a contribuição ainda precisa ser isolada.. O achado mais sólido é que o baseline e a intensidade da propagação mudam a conclusão..')
#23
title(23,'Fontes, rastreabilidade e o que foi preservado','Apêndice de reprodução')
text('Este PDF usa uma cópia dos resultados existentes, sem editar o código experimental nem substituir o relatório principal.. Os arquivos foram fotografados em 03/11/2026; rodadas posteriores precisam de uma nova consolidação..')
tab([['Local no projeto','Uso'],['outputs/testes_claude_20261102/resultados/','Origem dos sete JSONL de resultados e do JSON de proxy..'],['outputs/explicacao_fewshot_20261103/fontes/','Cópias exatas usadas neste guia; incluem comparação histórica, médias das escalas e médias de extras..'],['outputs/explicacao_fewshot_20261103/sha256.txt','Hashes para verificar a integridade de cada fonte copiada..'],['outputs/explicacao_fewshot_20261103/res_*_medias.csv','Tabelas de médias por configuração, incluindo validação, teste e macro-F1..'],['outputs/explicacao_fewshot_20261103/estatisticas.json','Contagens, correlações, resumo das variantes e médias das escalas..'],['tmp/pdfs/explicacao_20261103/','Scripts charts.py/build.py, gráficos e arquivos de inspeção visual..'],['proto_sgc/model.py; proto_sgc/training.py','Arquitetura, parâmetros treináveis e seleção de checkpoint auditados..'],['outputs/testes_claude_20261102/scripts/','Harness e wrappers de topologia/adaptação auditados..']],[225,270])
text('<b>Convenções:</b> “DGCG” nos novos testes significa 4–6, exceto onde indicado 2–4.. “Sem treino” mantém o backbone pré-treinado e a cabeça identidade com suas normalizações.. Acurácias agregadas são médias aritméticas entre bases; valores em pp são diferenças dessas acurácias..')
text('O relatório principal anterior, relatorio_fewshot.pdf, permanece disponível com a análise histórica extensa.. Este guia é complementar e prioriza a interpretação dos testes recentes, incluindo correções de leitura em relação aos comentários preliminares do README..')
# Build
W,H=595.276,841.89
def page(c,doc):
 c.setStrokeColor(teal);c.setLineWidth(1);c.line(50,H-31,W-50,H-31);c.setFont('D',8);c.setFillColor(navy);c.drawString(50,27,'FEW-SHOT  /  GUIA DOS RESULTADOS  /  03 NOV 2026');c.drawRightString(W-50,27,f'{doc.page:02d}')
doc=SimpleDocTemplate(str(OUT),pagesize=(W,H),rightMargin=50,leftMargin=50,topMargin=46,bottomMargin=47,title='Entendendo os resultados de few-shot',author='ELO • análise para Gabriel')
doc.build(story,onFirstPage=page,onLaterPages=page)
print(OUT)
