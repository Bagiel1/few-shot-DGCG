import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const baseDir = `${root}/resultados/consolidacao_confiavel_20261103`;
const data = JSON.parse(await fs.readFile(`${baseDir}/workbook_data.json`, "utf8"));
const outputPath = `${baseDir}/resultados_fewshot_fonte_unica.xlsx`;
const previewDir = "/tmp/fewshot_fonte_unica_previews";
await fs.mkdir(previewDir, { recursive: true });

const wb = Workbook.create();
const summary = wb.worksheets.add("Leia primeiro");
const results = wb.worksheets.add("Resultados");
const extras = wb.worksheets.add("Extras");
const sources = wb.worksheets.add("Fontes");
const audit = wb.worksheets.add("Auditoria");
const glossary = wb.worksheets.add("Glossário");

const navy = "#17324D";
const teal = "#167C73";
const lightTeal = "#E6F2F0";
const lightBlue = "#EAF0F6";
const amber = "#FFF2CC";
const red = "#FCE8E6";
const gray = "#E7EAED";
const text = "#233342";
const font = "Arial";

for (const sheet of [summary, results, extras, sources, audit, glossary]) {
  sheet.showGridLines = false;
  sheet.getRange("A1:AZ1000").format.font = { name: font, size: 10, color: text };
}
summary.tabColor = navy;
results.tabColor = teal;
extras.tabColor = "#4F81BD";
sources.tabColor = "#7F8C8D";
audit.tabColor = "#C58B22";
glossary.tabColor = "#8E6C9E";

function header(range) {
  range.format = {
    fill: navy,
    font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
}

function title(sheet, titleText, context) {
  sheet.getRange("A2").values = [[titleText]];
  sheet.getRange("A2").format.font = { name: font, size: 16, bold: true, color: navy };
  sheet.getRange("A3").values = [[context]];
  sheet.getRange("A3").format.font = { name: font, size: 10, italic: true, color: "#5A6B78" };
  sheet.getRange("A4:H4").format.borders = { bottom: { style: "thin", color: teal } };
}

function writeMatrix(sheet, anchor, matrix) {
  sheet.getRange(anchor).write(matrix);
}

function letter(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    n -= 1;
    out = String.fromCharCode(65 + (n % 26)) + out;
    n = Math.floor(n / 26);
  }
  return out;
}

const resultKeys = Object.keys(data.catalog[0]);
const resultHeaders = [
  "ID", "Bateria", "Status", "Escopo das bases", "Versão", "Way", "Shot", "Query", "Seeds",
  "Dispositivo", "Treino", "Dimensão", "Inicialização", "Pré-processamento", "Método", "Configuração",
  "Regra de agregação", "Observações", "Validação (%)", "Teste (%)", "Desvio teste", "F1 macro (%)",
  "Grupo de comparação", "Comparabilidade", "Alerta", "Fonte", "Filtro na fonte",
];
const resultRows = data.catalog.map(row => resultKeys.map(key => row[key]));
title(results, "Resultados canônicos", "Cada linha tem escopo suficiente para impedir comparações entre agregações incompatíveis..");
results.getRange("A4").write([resultHeaders, ...resultRows]);
header(results.getRange(`A4:${letter(resultHeaders.length - 1)}4`));
const resultsTable = results.tables.add(`A4:${letter(resultHeaders.length - 1)}${4 + resultRows.length}`, true, "ResultadosCanonicos");
resultsTable.style = "TableStyleMedium2";
results.freezePanes.freezeRows(4);
results.freezePanes.freezeColumns(4);
results.getRange(`R5:R${4 + resultRows.length}`).format.numberFormat = "0";
results.getRange(`S5:V${4 + resultRows.length}`).format.numberFormat = "0.0000";
results.getRange(`A4:${letter(resultHeaders.length - 1)}${4 + resultRows.length}`).format.verticalAlignment = "center";
results.getRange(`A5:Q${4 + resultRows.length}`).format.wrapText = true;
results.getRange(`W5:AA${4 + resultRows.length}`).format.wrapText = true;
const resultWidths = [28, 25, 12, 28, 12, 8, 8, 8, 10, 27, 25, 10, 15, 16, 22, 28, 34, 12, 13, 13, 13, 13, 28, 36, 42, 48, 42];
resultWidths.forEach((width, i) => results.getRange(`${letter(i)}:${letter(i)}`).format.columnWidth = width);
results.getRange(`A4:AA${4 + resultRows.length}`).format.rowHeight = 32;

const idToResultRow = new Map(data.catalog.map((row, index) => [row.id, 5 + index]));
function accFormula(id) {
  const row = idToResultRow.get(id);
  if (!row) throw new Error(`ID ausente no catálogo: ${id}`);
  return `='Resultados'!$T$${row}`;
}
function deltaFormula(idA, idB) {
  return `=${accFormula(idA).slice(1)}-${accFormula(idB).slice(1)}`;
}

title(summary, "Fonte única dos resultados few-shot", "Use esta aba antes de citar qualquer acurácia..");
summary.mergeCells("A5:G5");
summary.getRange("A5").values = [["Regra: nenhum número deve ser citado sem protocolo, escopo, treino, dimensão, inicialização, método e fonte.."]];
summary.getRange("A5:G5").format = {
  fill: amber,
  font: { name: font, size: 11, bold: true, color: "#6B4E00" },
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: "#D6B656" },
};
summary.getRange("A5:G5").format.rowHeight = 34;

summary.getRange("A7:G7").values = [["Número", "Escopo", "Protocolo", "Condição", "Acurácia (%)", "Grupo", "Leitura correta"]];
header(summary.getRange("A7:G7"));
const headline = [
  ["DGCG 4–6 histórico geral", "10 bases e 27 protocolos", "way 2/3/5; shot 1/5/10; query 1/5/10", "Treino padrão, 256, aleatória", "HIST-ALL27-dgcg-4:6", "HIST_ALL27_TRAINED", "Este é o 76,05% do relatório.."],
  ["DGCG 4–6 histórico no recorte", "10 bases e um protocolo", "5-way, 5-shot, 10-query", "Treino padrão, 256, aleatória", "HIST-5W5S10Q-dgcg-4:6", "HIST_5W5S10Q_TRAINED", "Recorte correspondente na GPU.."],
  ["DGCG 4–6 reproduzido", "10 bases e um protocolo", "5-way, 5-shot, 10-query", "Treino padrão, 256, aleatória", "CLAUDE-E1-5S-dgcg-TR256", "CLAUDE_RAW_5W5S10Q_TR256", "Este é o 75,14% da CPU.."],
  ["DGCG 4–6 sem treino", "10 bases e um protocolo", "5-way, 5-shot, 10-query", "Sem treino, 512, identidade", "CLAUDE-E1-5S-dgcg-TF512", "CLAUDE_RAW_5W5S10Q_TF512", "Controle sem treino.."],
  ["Sem grafo histórico geral", "10 bases e 27 protocolos", "way 2/3/5; shot 1/5/10; query 1/5/10", "Treino padrão, 256, aleatória", "HIST-ALL27-no-graph--", "HIST_ALL27_TRAINED", "Não comparar com 5w5s10q.."],
  ["Sem grafo histórico no recorte", "10 bases e um protocolo", "5-way, 5-shot, 10-query", "Treino padrão, 256, aleatória", "HIST-5W5S10Q-no-graph--", "HIST_5W5S10Q_TRAINED", "Recorte correspondente na GPU.."],
  ["Sem grafo sem treino", "10 bases e um protocolo", "5-way, 5-shot, 10-query", "Sem treino, 512, identidade", "CLAUDE-E1-5S-no-graph-TF512", "CLAUDE_RAW_5W5S10Q_TF512", "Este é o 76,10%.."],
  ["Sem grafo identidade + treino", "10 bases e um protocolo", "5-way, 5-shot, 10-query", "Treino, 512, identidade", "CLAUDE-IDINIT-5S-no-graph", "CLAUDE_IDINIT512_5W5S10Q", "Controle de inicialização.."],
];
summary.getRange("A8:G15").values = headline.map(row => [row[0], row[1], row[2], row[3], null, row[5], row[6]]);
headline.forEach((row, index) => summary.getRange(`E${8 + index}`).formulas = [[accFormula(row[4])]]);
summary.getRange("E8:E15").format.numberFormat = "0.0000";
summary.getRange("A8:G15").format.wrapText = true;
summary.getRange("A8:G15").format.rowHeight = 42;
summary.getRange("A8:G15").format.borders = { preset: "inside", style: "thin", color: "#D8E0E6" };

summary.getRange("A17:F17").values = [["Comparação", "Resultado A (%)", "Resultado B (%)", "Diferença (pp)", "Pode comparar?", "Motivo"]];
header(summary.getRange("A17:F17"));
const comparisons = [
  ["DGCG 4–6 vs sem grafo na média dos 27 protocolos", "HIST-ALL27-dgcg-4:6", "HIST-ALL27-no-graph--", "SIM", "Mesmo grupo histórico e mesma regra de agregação.."],
  ["DGCG 4–6 vs sem grafo no histórico 5w5s10q", "HIST-5W5S10Q-dgcg-4:6", "HIST-5W5S10Q-no-graph--", "SIM", "Mesmo protocolo, seed, treino, dimensão e dispositivo.."],
  ["DGCG 4–6 vs sem grafo na reprodução treinada", "CLAUDE-E1-5S-dgcg-TR256", "CLAUDE-E1-5S-no-graph-TR256", "SIM", "Mesmo protocolo e condição no Claude E1.."],
  ["DGCG 4–6 vs sem grafo sem treino", "CLAUDE-E1-5S-dgcg-TF512", "CLAUDE-E1-5S-no-graph-TF512", "SIM", "Mesmo protocolo e condição sem treino.."],
  ["DGCG treinado vs DGCG sem treino", "CLAUDE-E1-5S-dgcg-TR256", "CLAUDE-E1-5S-dgcg-TF512", "COM RESSALVA", "Também mudam dimensão e inicialização.."],
  ["76,05% histórico geral vs 75,14% Claude", "HIST-ALL27-dgcg-4:6", "CLAUDE-E1-5S-dgcg-TR256", "NÃO", "27 protocolos contra um único protocolo.."],
  ["Sem grafo identidade + treino vs sem treino", "CLAUDE-IDINIT-5S-no-graph", "CLAUDE-E1-5S-no-graph-TF512", "SIM, COM RESSALVA", "Mesma dimensão e inicialização; resultados vêm de arquivos separados.."],
];
summary.getRange("A18:F24").values = comparisons.map(row => [row[0], null, null, null, row[3], row[4]]);
comparisons.forEach((row, index) => {
  const excelRow = 18 + index;
  summary.getRange(`B${excelRow}`).formulas = [[accFormula(row[1])]];
  summary.getRange(`C${excelRow}`).formulas = [[accFormula(row[2])]];
  summary.getRange(`D${excelRow}`).formulas = [[deltaFormula(row[1], row[2])]];
});
summary.getRange("B18:D24").format.numberFormat = "0.0000";
summary.getRange("A18:F24").format.wrapText = true;
summary.getRange("A18:F24").format.rowHeight = 38;
summary.getRange("A18:F24").format.borders = { preset: "inside", style: "thin", color: "#D8E0E6" };
summary.getRange("E18:E24").conditionalFormats.add("containsText", { text: "NÃO", format: { fill: red, font: { bold: true, color: "#A61B1B" } } });
summary.getRange("E18:E24").conditionalFormats.add("containsText", { text: "SIM", format: { fill: lightTeal, font: { bold: true, color: teal } } });
summary.getRange("E18:E24").conditionalFormats.add("containsText", { text: "RESSALVA", format: { fill: amber, font: { bold: true, color: "#6B4E00" } } });

summary.getRange("A26:B26").values = [["Estado da consolidação", "Valor"]];
header(summary.getRange("A26:B26"));
summary.getRange("A27:B30").values = [
  ["Auditoria", data.audit.status],
  ["Linhas canônicas", data.audit.canonical_rows],
  ["Médias da bateria de extras", data.audit.extras_rows],
  ["Checagens com falha", data.audit.failures.length],
];
summary.getRange("A27:B30").format.borders = { preset: "inside", style: "thin", color: "#D8E0E6" };
summary.getRange("B27:B30").format.font = { name: font, bold: true, color: teal };
summary.mergeCells("A32:G32");
summary.getRange("A32").values = [["Para apresentar ao professor: cite o grupo de comparação junto com o número.. O PDF continua útil para narrativa; esta planilha é a referência numérica.."]];
summary.getRange("A32:G32").format = { fill: lightBlue, font: { name: font, bold: true, color: navy }, wrapText: true };
summary.getRange("A32:G32").format.rowHeight = 32;
summary.getRange("A:G").format.verticalAlignment = "center";
const summaryWidths = [42, 29, 38, 31, 18, 32, 40];
summaryWidths.forEach((width, i) => summary.getRange(`${letter(i)}:${letter(i)}`).format.columnWidth = width);
summary.freezePanes.freezeRows(7);

const extraKeys = Object.keys(data.extras[0]);
const extraHeaders = extraKeys.map(key => ({
  grupo_comparacao: "Grupo de comparação", escopo: "Escopo", dispositivo: "Dispositivo", treino: "Treino", alerta: "Alerta",
  n_way: "Way", n_shot: "Shot", n_query: "Query", metodo: "Método", alvo: "Alvo", extra_mode: "Modo dos extras",
  extra_phase: "Fase", extras_per_class: "Extras/classe", concluidos: "Concluídos", planejados: "Planejados",
  val_accuracy: "Validação (%)", test_accuracy: "Teste (%)", test_macro_f1: "F1 macro (%)",
  test_balanced_accuracy: "Acurácia balanceada (%)", ganho_accuracy_vs_sem_extras_pp: "Ganho vs controle (pp)",
}[key] || key));
const extraRows = data.extras.map(row => extraKeys.map(key => row[key]));
title(extras, "Bateria de imagens extras", "474 médias de 4.740 execuções; não misturar com o histórico sem banco reservado..");
extras.getRange("A4").write([extraHeaders, ...extraRows]);
header(extras.getRange(`A4:${letter(extraHeaders.length - 1)}4`));
const extrasTable = extras.tables.add(`A4:${letter(extraHeaders.length - 1)}${4 + extraRows.length}`, true, "MediasExtras");
extrasTable.style = "TableStyleMedium2";
extras.freezePanes.freezeRows(4);
extras.freezePanes.freezeColumns(5);
extras.getRange(`M5:O${4 + extraRows.length}`).format.numberFormat = "0";
extras.getRange(`P5:T${4 + extraRows.length}`).format.numberFormat = "0.0000";
extras.getRange(`A4:T${4 + extraRows.length}`).format.rowHeight = 30;
extras.getRange(`A5:E${4 + extraRows.length}`).format.wrapText = true;
const extraWidths = [34, 32, 26, 24, 44, 8, 8, 8, 18, 10, 18, 16, 13, 13, 13, 14, 14, 14, 20, 20];
extraWidths.forEach((width, i) => extras.getRange(`${letter(i)}:${letter(i)}`).format.columnWidth = width);

title(sources, "Inventário das fontes", "Hashes e contagens identificam exatamente os arquivos usados na consolidação..");
const inventoryHeaders = ["Fonte", "Arquivo", "Linhas", "SHA-256", "Status"];
const inventoryRows = data.inventory.map(row => [row.fonte, row.arquivo, row.linhas, row.sha256, row.status]);
sources.getRange("A5").write([inventoryHeaders, ...inventoryRows]);
header(sources.getRange("A5:E5"));
sources.tables.add(`A5:E${5 + inventoryRows.length}`, true, "InventarioFontes").style = "TableStyleMedium2";
sources.getRange(`A6:E${5 + inventoryRows.length}`).format.rowHeight = 30;
sources.getRange(`A6:B${5 + inventoryRows.length}`).format.wrapText = true;
sources.getRange("A:A").format.columnWidth = 32;
sources.getRange("B:B").format.columnWidth = 70;
sources.getRange("C:C").format.columnWidth = 12;
sources.getRange("D:D").format.columnWidth = 68;
sources.getRange("E:E").format.columnWidth = 12;
sources.freezePanes.freezeRows(5);

title(audit, "Auditoria da consolidação", "Contagens, chaves únicas, recálculos e reprodução CPU/GPU..");
audit.getRange("A5:C5").values = [["Checagem", "Status", "Detalhe"]];
header(audit.getRange("A5:C5"));
const checkRows = data.checks.map(row => [row.checagem, row.status, row.detalhe]);
audit.getRange("A6").write(checkRows);
audit.getRange(`A6:C${5 + checkRows.length}`).format.rowHeight = 27;
audit.getRange(`A6:C${5 + checkRows.length}`).format.wrapText = true;
audit.getRange(`B6:B${5 + checkRows.length}`).conditionalFormats.add("containsText", { text: "FALHA", format: { fill: red, font: { bold: true, color: "#A61B1B" } } });
audit.getRange(`B6:B${5 + checkRows.length}`).conditionalFormats.add("containsText", { text: "OK", format: { fill: lightTeal, font: { bold: true, color: teal } } });
const repStart = 8 + checkRows.length;
audit.getRange(`A${repStart}:F${repStart}`).values = [["Método", "Pares", "Claude CPU (%)", "Histórico GPU (%)", "Diferença média (pp)", "Maior diferença absoluta (pp)"]];
header(audit.getRange(`A${repStart}:F${repStart}`));
const repRows = data.reproduction.map(row => [row.method, row.pares, row.cpu_mean, row.gpu_mean, row.delta_mean, row.delta_abs_max]);
audit.getRange(`A${repStart + 1}`).write(repRows);
audit.getRange(`C${repStart + 1}:F${repStart + repRows.length}`).format.numberFormat = "0.0000";
audit.getRange("A:A").format.columnWidth = 54;
audit.getRange("B:B").format.columnWidth = 14;
audit.getRange("C:F").format.columnWidth = 24;
audit.freezePanes.freezeRows(5);

title(glossary, "Glossário e regras de leitura", "Definições usadas na fonte única..");
glossary.getRange("A5:C5").values = [["Termo", "Definição", "Implicação para comparação"]];
header(glossary.getRange("A5:C5"));
const glossaryRows = [
  ["Média dos 27 protocolos", "Média de 270 execuções: dez bases vezes 27 combinações de way, shot e query..", "Não equivale ao desempenho de um protocolo específico.."],
  ["Recorte 5w5s10q", "Média de dez execuções: uma por base, com 5-way, 5-shot e 10-query..", "Pode ser comparado somente a outro resultado com o mesmo protocolo e escopo.."],
  ["Treino padrão", "Projeção aleatória para 256 dimensões, 2.000 episódios e checkpoint escolhido por validação..", "Diferente da identidade 512 sem treino.."],
  ["Sem treino", "Θ identidade em 512 dimensões; LayerNorm e normalização L2 continuam presentes..", "Não significa feature ResNet completamente intocada.."],
  ["Idinit", "Θ identidade em 512 dimensões antes do treino..", "Isola melhor o efeito do aprendizado, mas o DGCG desta bateria usa grau 2–4.."],
  ["DGCG 4–6 / 2–4", "Faixa alvo de grau do grafo..", "São configurações diferentes e devem aparecer no rótulo do resultado.."],
  ["Grupo de comparação", "Identificador que reúne somente linhas com protocolo e condição compatíveis..", "Comparações fora do grupo exigem justificativa explícita.."],
  ["Desvio entre seeds", "Desvio amostral das seeds 7, 17 e 27 na bateria de escalas..", "Não é o intervalo de confiança dos episódios.."],
  ["Extras", "Bateria com banco reservado e diferentes fases de inclusão das imagens extras..", "Usar o controle sem extras da mesma bateria.."],
  ["CPU/GPU", "O dispositivo é parte da proveniência; os arquivos históricos não gravam esse campo diretamente..", "Atribuição causal ao dispositivo exige experimento controlado.."],
];
glossary.getRange("A6").write(glossaryRows);
glossary.getRange(`A6:C${5 + glossaryRows.length}`).format.wrapText = true;
glossary.getRange(`A6:C${5 + glossaryRows.length}`).format.rowHeight = 52;
glossary.getRange(`A5:C${5 + glossaryRows.length}`).format.borders = { preset: "inside", style: "thin", color: "#D8E0E6" };
glossary.getRange("A:A").format.columnWidth = 30;
glossary.getRange("B:B").format.columnWidth = 67;
glossary.getRange("C:C").format.columnWidth = 58;

wb.recalculate();

const summaryCheck = await wb.inspect({
  kind: "table",
  range: "Leia primeiro!A7:G24",
  include: "values,formulas",
  tableMaxRows: 25,
  tableMaxCols: 8,
  maxChars: 9000,
});
console.log(summaryCheck.ndjson);
const errors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
  maxChars: 5000,
});
console.log(errors.ndjson);

const previewRanges = {
  "Leia primeiro": "A1:G32",
  "Resultados": "A1:AA14",
  "Extras": "A1:T14",
  "Fontes": "A1:E22",
  "Auditoria": "A1:F50",
  "Glossário": "A1:C15",
};
for (const sheetName of Object.keys(previewRanges)) {
  const preview = await wb.render({ sheetName, range: previewRanges[sheetName], scale: 0.8, format: "png" });
  const safe = sheetName.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/\s+/g, "_").toLowerCase();
  await fs.writeFile(`${previewDir}/${safe}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, previewDir, sheets: 6, canonicalRows: data.catalog.length, extrasRows: data.extras.length }));
