"use strict";

// A interface apenas desenha os tensores recebidos. O aprendizado acontece em engine.py.
const $ = (id) => document.getElementById(id);
const form = $("configuration");
const COLORS = ["#66dec9", "#77a8ff", "#ffbb71", "#ee94d8", "#b7a0ff"];
const STAGES = [
  {id: "episode", label: "Episódio", title: "Embeddings fixos do episódio", code: "episodes.py · FeatureEpisodeSampler.sample"},
  {id: "graph", label: "Grafo", title: "Quem agrega informação de quem", code: "model.py · normalized_graph_adjacency"},
  {id: "propagation", label: "Propagação", title: "A informação atravessa o grafo", code: "model.py · encode_episode"},
  {id: "embedding", label: "Representação", title: "Projeção, LayerNorm e normalização L2", code: "model.py · encode_episode"},
  {id: "prototype", label: "Protótipos", title: "Um centroide por classe do suporte", code: "model.py · forward"},
  {id: "prediction", label: "Predição", title: "Consultas comparadas aos protótipos", code: "model.py · forward → cross_entropy"},
  {id: "update", label: "Atualização", title: "Gradiente e passo do AdamW", code: "training.py · loss.backward → optimizer.step"},
];
let data = null;
let step = 1;
let stage = "episode";
let phase = "before";
let selected = 0;
let hop = 0;
let timer = null;
let running = false;
let currentTab = "simulation";
let source = "synthetic";
const plotDomains = new WeakMap();
let snippets = {};
let codeError = "";
let selectedSource = "";
const openCodeSections = new Set();

const esc = (value) => String(value).replace(/[&<>"']/g, (character) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[character]));
const num = (value, digits = 4) => Number.isFinite(value) ? Number(value).toLocaleString("pt-BR", {maximumFractionDigits: digits, minimumFractionDigits: digits}) : "—";
const pct = (value) => `${num(100 * value, 1)}%`;
const tiny = (value) => Number(value).toExponential(3).replace(".", ",");
const sum = (values) => values.reduce((total, value) => total + value, 0);

function snapshot() { return data.steps[step - 1]; }

function observed() {
  const item = snapshot();
  if ($("episode-view").value === "probe") {
    return {trace: phase === "after" ? item.probe : (step === 1 ? data.initial_probe : data.steps[step - 2].probe), episode: data.probe_episode};
  }
  return {trace: item[phase], episode: item.episode};
}

function stop() {
  if (timer) clearInterval(timer);
  timer = null;
  $("play").textContent = "▶ Reproduzir";
}

function chooseStep(value) {
  if (!data) return;
  step = Math.max(1, Math.min(data.steps.length, Number(value)));
  render();
}

function setTab(value) {
  currentTab = value;
  for (const tab of ["simulation", "math"]) {
    $(`tab-${tab}`).setAttribute("aria-selected", String(tab === value));
    $(`${tab}-panel`).hidden = tab !== value;
  }
  if (data) render();
}

function enabledFields() {
  const baseline = form.elements.model_name.value === "protonet";
  const dgcg = form.elements.graph_type.value.startsWith("dgcg");
  if (baseline) {
    form.elements.grande_metric.value = "off";
    form.elements.cosine_rbf_weight.checked = false;
  }
  for (const name of ["graph_type", "sgc_hops", "grande_metric", "cosine_rbf_weight", "graph_temperature"]) form.elements[name].disabled = baseline;
  form.elements.knn.disabled = baseline || dgcg;
  form.elements.grande_sigma.disabled = baseline || form.elements.grande_metric.value === "off";
  form.elements.grande_rbo_p.disabled = baseline || form.elements.grande_metric.value !== "rbo";
  for (const name of ["dgcg_metric", "dgcg_top_k"]) form.elements[name].disabled = baseline || (!dgcg && form.elements.grande_metric.value !== "rbo");
  for (const name of ["dgcg_correlation", "dgcg_threshold"]) form.elements[name].disabled = baseline || !dgcg;
  form.elements.noise.disabled = source === "cache";
}

function configuration() {
  const values = {};
  for (const input of form.elements) {
    if (!input.name) continue;
    if (input.type === "checkbox") values[input.name] = input.checked;
    else if (input.type === "number" || input.type === "range") values[input.name] = input.value === "" ? null : Number(input.value);
    else values[input.name] = input.value;
  }
  return values;
}

async function run(event) {
  if (event) event.preventDefault();
  if (running || !form.reportValidity()) return;
  stop();
  running = true;
  $("error").hidden = true;
  $("run").disabled = true;
  $("run").textContent = "Calculando no PyTorch…";
  const config = configuration();
  for (const input of form.elements) input.disabled = true;
  $("status").textContent = `Executando ${config.steps} episódios, validação e teste…`;
  const start = performance.now();
  try {
    const response = await fetch("/api/simulate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(config)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Não foi possível executar a simulação.");
    data = result;
    source = result.source;
    step = 1;
    selected = 0;
    hop = 0;
    $("status").textContent = `${result.steps.length} passos calculados em ${num((performance.now() - start) / 1000, 1)} s. Altere os parâmetros e execute novamente para comparar.`;
    $("source-description").textContent = result.source === "cache"
      ? `Cache real da ResNet: ${result.total_images} imagens, ${result.input_dim} dimensões. Os vetores ficam congelados durante o treino. A dispersão sintética não se aplica.`
      : `Embeddings sintéticos fixos: ${result.total_images} exemplos, ${result.input_dim} dimensões. Substituem a ResNet congelada; o modelo, os grafos e o AdamW são os do projeto.`;
    render();
  } catch (error) {
    $("error").hidden = false;
    $("error").textContent = error.message;
    $("status").textContent = data ? "O resultado anterior continua disponível." : "Ajuste os parâmetros e tente novamente. O servidor precisa continuar aberto no terminal.";
  } finally {
    running = false;
    for (const input of form.elements) input.disabled = false;
    enabledFields();
    $("run").innerHTML = 'Executar experimento <span aria-hidden="true">↗</span>';
  }
}

function svgElement(tag, attributes = {}, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

function bounds(values) {
  const lo = Math.min(...values), hi = Math.max(...values);
  const padding = Math.max((hi - lo) * 0.16, 0.15);
  return [lo - padding, hi + padding];
}

function stagePoints(trace) {
  if (stage === "episode" || stage === "graph") return trace.x;
  if (stage === "propagation") return trace.hops[Math.min(hop, trace.hops.length - 1)];
  return trace.embeddings;
}

function stablePlotDomain() {
  // Mesmos limites ao comparar pesos e passos: o movimento não vem de um
  // reenquadramento automático. Todos os saltos compartilham o mesmo plano.
  const mode = $("episode-view").value;
  let cache = plotDomains.get(data);
  if (!cache) {cache = new Map(); plotDomains.set(data, cache);}
  const key = `${mode}:${stage}`;
  if (cache.has(key)) return cache.get(key);
  const traces = mode === "probe" ? [data.initial_probe, ...data.steps.map(s => s.probe)] : data.steps.flatMap(s => [s.before, s.after]);
  const points = traces.flatMap(trace => stage === "propagation" ? trace.hops.flat() : stage === "prototype" || stage === "prediction" ? [...trace.embeddings, ...trace.prototypes] : stagePoints(trace));
  const domain = [bounds(points.map(p => p[0])), bounds(points.map(p => p[1]))];
  cache.set(key, domain);
  return domain;
}

function drawPlot(trace, episode) {
  const container = $("plot");
  container.replaceChildren();
  if (stage === "update") { drawUpdate(); return; }
  const points = stagePoints(trace);
  const showPrototypes = stage === "prototype" || stage === "prediction";
  const width = Math.max(240, container.clientWidth);
  const height = width < 420 ? 310 : 350;
  const margin = {l: 48, r: 20, t: 22, b: 43};
  const [[xmin, xmax], [ymin, ymax]] = stablePlotDomain();
  const x = (v) => margin.l + (v - xmin) / (xmax - xmin) * (width - margin.l - margin.r);
  const y = (v) => height - margin.b - (v - ymin) / (ymax - ymin) * (height - margin.t - margin.b);
  const svg = svgElement("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${STAGES.find(s => s.id === stage).title}. ${points.length} nós; círculos preenchidos são suporte, vazios são consultas.`});
  container.append(svg);
  const defs = svgElement("defs");
  const marker = svgElement("marker", {id: "arrow", markerWidth: 7, markerHeight: 7, refX: 6, refY: 3.5, orient: "auto", markerUnits: "userSpaceOnUse"});
  marker.append(svgElement("path", {d: "M0,0 L7,3.5 L0,7 Z", fill: "#66dec9"}));
  defs.append(marker); svg.append(defs);
  const ticks = width < 420 ? 3 : 5;
  for (let i = 0; i < ticks; i++) {
    const a = xmin + (xmax - xmin) * i / (ticks - 1), b = ymin + (ymax - ymin) * i / (ticks - 1);
    svg.append(svgElement("line", {x1:x(a), x2:x(a), y1:margin.t, y2:height-margin.b, class:"grid"}));
    svg.append(svgElement("line", {x1:margin.l, x2:width-margin.r, y1:y(b), y2:y(b), class:"grid"}));
    svg.append(svgElement("text", {x:x(a), y:height-margin.b+20, "text-anchor":"middle"}, num(a, 1)));
    svg.append(svgElement("text", {x:margin.l-8, y:y(b)+4, "text-anchor":"end"}, num(b, 1)));
  }
  svg.append(svgElement("rect", {x:margin.l, y:margin.t, width:width-margin.l-margin.r, height:height-margin.t-margin.b, class:"frame"}));
  const symbol = stage === "episode" || stage === "graph" ? "X" : stage === "propagation" ? `H⁽${hop}⁾` : "Z";
  svg.append(svgElement("text", {x:(margin.l+width-margin.r)/2, y:height-3, "text-anchor":"middle", class:"axis-title"}, `${symbol} · coordenada 1`));
  svg.append(svgElement("text", {transform:`translate(13 ${(margin.t+height-margin.b)/2}) rotate(-90)`, "text-anchor":"middle", class:"axis-title"}, `${symbol} · coordenada 2`));
  if ((stage === "graph" || stage === "propagation") && data.config.model_name === "proto-sgc") {
    for (let target = 0; target < points.length; target++) for (let sourceIndex = 0; sourceIndex < points.length; sourceIndex++) {
      if (target === sourceIndex || !trace.adjacency[target][sourceIndex]) continue;
      const chosen = target === selected;
      const from = points[sourceIndex], to = points[target];
      const dx = x(to[0])-x(from[0]), dy = y(to[1])-y(from[1]), length = Math.hypot(dx,dy);
      if (length < 1) continue;
      svg.append(svgElement("line", {x1:x(from[0])+dx/length*9, y1:y(from[1])+dy/length*9, x2:x(to[0])-dx/length*11, y2:y(to[1])-dy/length*11,
        stroke:chosen?COLORS[0]:"#75869f", "stroke-width":chosen?1.8:0.7, opacity:chosen?0.85:0.18,
        ...(chosen ? {"marker-end":"url(#arrow)"} : {})}));
    }
  }
  points.forEach((p, index) => {
    const node = episode.nodes[index], query = node.role === "query";
    const prediction = query ? trace.predictions[index - episode.support_count] : node.label;
    const color = COLORS[stage === "prediction" ? prediction : node.label];
    const g = svgElement("g", {class:"node", role:"button", tabindex:"0", "aria-label":`${query?"Consulta":"Suporte"} ${index}, classe ${node.class_name}, ${node.id}`});
    g.append(svgElement("title", {}, `${index} · ${node.id} · ${query?"consulta":"suporte"} · classe local ${node.label}`));
    g.append(svgElement("circle", {cx:x(p[0]), cy:y(p[1]), r:17, fill:"transparent"}));
    if (index === selected) g.append(svgElement("circle", {cx:x(p[0]), cy:y(p[1]), r:12, fill:"none", stroke:"#e6edf7", "stroke-width":1.2}));
    g.append(svgElement("circle", {cx:x(p[0]), cy:y(p[1]), r:7, fill:query?"#111a27":color, stroke:color, "stroke-width":2.2}));
    if (stage === "prediction" && query && prediction !== node.label) g.append(svgElement("path", {d:`M${x(p[0])-3},${y(p[1])-3} l6,6 m0,-6 l-6,6`, stroke:"#ff8897", "stroke-width":1.5}));
    if (points.length <= 25 || index === selected) g.append(svgElement("text", {x:x(p[0])+11, y:y(p[1])-10, class:"node-label"}, index));
    const selectNode = () => {selected = index; renderSimulation();};
    g.addEventListener("click", selectNode);
    g.addEventListener("keydown", (event) => {if (event.key === "Enter" || event.key === " ") {event.preventDefault(); selectNode();}});
    svg.append(g);
  });
  if (showPrototypes) trace.prototypes.forEach((p, index) => {
    const g = svgElement("g");
    g.append(svgElement("path", {d:`M${x(p[0])},${y(p[1])-10} l10,10 l-10,10 l-10,-10 Z`, fill:"#111a27", stroke:COLORS[index], "stroke-width":2.5}));
    g.append(svgElement("text", {x:x(p[0])+13, y:y(p[1])+4, fill:COLORS[index]}, `P${index}`));
    g.append(svgElement("title", {}, `Protótipo da classe ${episode.classes[index]}`));
    svg.append(g);
  });
}

function drawUpdate() {
  const s = snapshot();
  const a = s.parameters_before.theta[0][0], b = s.parameters_after.theta[0][0];
  $("plot").innerHTML = `<div class="weight-map"><div><span class="eyebrow">COEFICIENTE Θ₁₁ · UM ENTRE TODOS OS PARÂMETROS</span><div class="equation">${num(a,6)} <span class="numeric">→</span> ${num(b,6)}<br>ΔΘ₁₁ = <span class="numeric">${num(b-a,6)}</span></div></div><div><h3>O que participa da atualização</h3><div class="equation">∂ℒ/∂Θ₁₁ = ${num(s.gradient[0][0],6)}<br>Após clipping = ${num(s.clipped_gradient[0][0],6)}<br>‖∇ℒ‖₂ = ${num(s.gradient_norm,5)}<br>‖Θ<sub>depois</sub> − Θ<sub>antes</sub>‖<sub>F</sub> = ${num(s.weight_delta_norm,6)}</div></div><p class="muted">Theta, γ e β da LayerNorm e a escala dos logits são aprendidos. X fica fixo. A aba Matemática detalha o AdamW deste passo.</p></div>`;
}

function table(matrix, title, options = {}) {
  if (!matrix.length) return "";
  const maxRows = options.maxRows || 12, maxCols = options.maxCols || 8;
  const columns = Math.min(matrix[0].length, maxCols);
  const rows = matrix.slice(0, maxRows);
  return `<h3>${esc(title)}</h3><div class="table-wrap"><table><thead><tr><th>Índice</th>${Array.from({length: columns}, (_,i)=>`<th>${i}</th>`).join("")}</tr></thead><tbody>${rows.map((row,i)=>`<tr><th>${i}</th>${row.slice(0,columns).map(v=>`<td>${num(v,4)}</td>`).join("")}</tr>`).join("")}</tbody></table></div><p class="muted">Exibição: ${rows.length} de ${matrix.length} linhas; ${columns} de ${matrix[0].length} colunas recebidas. Todos os atributos participam do cálculo no PyTorch.</p>`;
}

function graphSources() {
  const c = data.config;
  if (c.model_name === "protonet") return ["encode"];
  const ids = c.graph_type.startsWith("dgcg")
    ? ["dgcg", "rankings", "correlation", ...(c.dgcg_threshold === null ? ["threshold"] : []), ...(c.graph_type === "dgcg-plus" && !c.cosine_rbf_weight ? ["mutual"] : [])]
    : ["knn"];
  if (c.cosine_rbf_weight) ids.push("rbf");
  return [...ids, "graph_dispatch"];
}

function degreeSources() {
  if (data.config.model_name === "protonet") return ["encode"];
  return ["normalize_graph", ...(data.config.grande_metric !== "off" ? ["grande_degree", "grande_distances"] : [])];
}

function stageSources() {
  return {
    episode: ["sample", "concatenate", "split", data.source === "cache" ? "cache" : "synthetic"],
    graph: [...graphSources(), ...degreeSources()].filter((id, index, all) => all.indexOf(id) === index),
    propagation: ["encode", ...(data.config.model_name === "proto-sgc" ? ["normalize_graph"] : [])],
    embedding: ["normalization", "parameters", "encode"],
    prototype: ["prototypes", "concatenate"],
    prediction: ["prediction", "loss", "prototypes"],
    update: ["train_step", "optimizer", "observed_step", "after_update"],
  }[stage];
}

function pythonLine(line) {
  // Colore tokens sem inserir o código como HTML. Todo texto é escapado.
  const pattern = /#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:def|class|return|if|elif|else|for|in|not|and|or|with|as|from|import|try|except|raise|None|True|False)\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b/g;
  let result = "", offset = 0;
  for (const match of line.matchAll(pattern)) {
    result += esc(line.slice(offset, match.index));
    const token = match[0];
    const kind = token.startsWith("#") ? "comment" : /^["']/.test(token) ? "string" : /^\d/.test(token) ? "number-token" : "keyword";
    result += `<span class="source-${kind}">${esc(token)}</span>`;
    offset = match.index + token.length;
  }
  return result + esc(line.slice(offset));
}

function importantLine(id, line) {
  const patterns = {
    sample: /selected =|indices = rng.choice|support_indices =|query_indices =/,
    concatenate: /self.encode_episode|support_embeddings =|query_embeddings =/,
    graph_dispatch: /adjacency = self\.|return adjacency/,
    knn: /\.topk\(|\.scatter_|return torch\.|return adjacency_out/,
    dgcg: /ranked_lists =|correlations =|selected =|candidate_weights =|return adjacency_out/,
    normalize_graph: /adjacency = adjacency|out_degree =|in_degree =|inv_sqrt/,
    grande_degree: /edge_mask =|normalized_distances =|gaussian_similarities =|return degree/,
    encode: /propagated =|for _ in range|embeddings = self.normalization|return F.normalize/,
    normalization: /embeddings =|return F.normalize/,
    prototypes: /class_support =|prototypes.append|prototypes_tensor =/,
    prediction: /squared_distances =|scale =|return -scale/,
    train_step: /cross_entropy|loss.backward|clip_grad_norm_|optimizer.step/,
    observed_step: /cross_entropy|loss.backward|clip_grad_norm_|optimizer.step/,
    optimizer: /AdamW|lr=|weight_decay=/,
  };
  return patterns[id]?.test(line) || false;
}

function sourceHTML(id) {
  const item = snippets[id];
  if (!item) return `<p class="muted">${esc(codeError || "Este trecho de código não está disponível.")}</p>`;
  const nonempty = item.lines.filter(line => line.trim());
  const indent = nonempty.length ? Math.min(...nonempty.map(line => line.match(/^\s*/)[0].length)) : 0;
  const lines = item.lines.map((line, index) => `<span class="source-line${importantLine(id, line) ? " source-emphasis" : ""}" data-line="${item.start_line + index}"><span class="source-number" aria-hidden="true">${item.start_line + index}</span><span>${pythonLine(line.slice(indent)) || " "}</span></span>`).join("\n");
  return `<div class="source-location"><strong>${esc(item.path)}</strong><span>Linhas ${item.start_line}–${item.end_line} · ${esc(item.symbol)}</span></div><pre class="source-code" aria-label="${esc(item.title)}"><code>${lines}</code></pre><p class="muted source-caption">Trecho do arquivo original. As operações principais estão destacadas.</p>`;
}

function renderSource() {
  const ids = stageSources();
  if (!ids.includes(selectedSource)) selectedSource = ids[0];
  $("source-selector").innerHTML = ids.map(id => `<option value="${id}">${esc(snippets[id]?.title || id)}</option>`).join("");
  $("source-selector").value = selectedSource;
  $("source-content").innerHTML = sourceHTML(selectedSource);
}

function mathSource(index) {
  const groups = {
    1: ["sample", "split", data.source === "cache" ? "cache" : "synthetic"],
    2: graphSources(),
    3: degreeSources(),
    4: ["encode", "normalization", "parameters"],
    5: ["prototypes", "prediction", "loss"],
    6: ["train_step", "observed_step"],
    7: ["optimizer", "train_step", "observed_step"],
    8: ["after_update", "validation", "restore", "evaluate"],
  };
  const ids = groups[index], key = `math-${index}`;
  const extra = ids.slice(1).map(id => {
    const nestedKey = `${key}-${id}`;
    return `<details class="source-related" data-code-key="${nestedKey}" ${openCodeSections.has(nestedKey) ? "open" : ""}><summary>${esc(snippets[id]?.title || id)}</summary>${sourceHTML(id)}</details>`;
  }).join("");
  return `<details class="math-source" data-code-key="${key}" ${openCodeSections.has(key) ? "open" : ""}><summary>Ver o código desta operação</summary>${sourceHTML(ids[0])}${extra}</details>`;
}

function renderSimulation() {
  const {trace, episode} = observed(), s = snapshot(), cfg = data.config;
  selected = Math.min(selected, episode.nodes.length - 1);
  const definition = STAGES.find(value => value.id === stage);
  $("stage-title").textContent = definition.title;
  $("stage-code").textContent = definition.code;
  for (const button of $("stage-nav").querySelectorAll("button")) button.setAttribute("aria-pressed", String(button.dataset.stage === stage));
  $("state-before").setAttribute("aria-pressed", String(phase === "before"));
  $("state-after").setAttribute("aria-pressed", String(phase === "after"));
  const probe = $("episode-view").value === "probe";
  $("view-hint").textContent = probe
    ? "Mesmas imagens, mesmas classes e mesmos papéis em todos os passos. Este episódio não gera atualizações; serve para observar os pesos mudando."
    : "Cada passo amostra outras imagens. Antes/depois mostra o mesmo episódio com os pesos dos dois instantes.";
  $("hop-control").hidden = stage !== "propagation" || cfg.model_name === "protonet";
  $("hop-slider").max = trace.hops.length - 1;
  hop = Math.min(hop, trace.hops.length - 1);
  $("hop-slider").value = hop;
  $("hop-value").textContent = `${hop} / ${trace.hops.length-1}`;
  drawPlot(trace, episode);
  $("legend").innerHTML = stage === "update" ? "" : episode.classes.map((name, i) => `<span><i class="swatch" style="background:${COLORS[i]}"></i>${esc(name)} · rótulo ${i}</span>`).join("") + '<span>● suporte · ○ consulta</span>' + (stage === "prototype" || stage === "prediction" ? '<span>◇ protótipo</span>' : "");
  const displayedDim = stage === "episode" || stage === "graph" || (stage === "propagation" && cfg.grande_metric === "off" && cfg.model_name !== "protonet") ? data.input_dim : cfg.output_dim;
  $("projection-note").textContent = stage === "update" ? "" : `Mostrando as coordenadas 1 e 2 de vetores ${displayedDim}D. Distâncias e vizinhos são calculados em todas as dimensões. Cores indicam rótulos para leitura; os rótulos das consultas não entram no grafo.`;
  const node = episode.nodes[selected];
  let detail = `Nó ${selected} · ${node.id} · ${node.role === "support" ? "suporte" : "consulta"} · classe local ${node.label}.`;
  if (stage === "graph" || stage === "propagation") {
    const contributors = trace.normalized_adjacency[selected].map((weight, index) => ({weight, index})).filter(item => item.weight !== 0).sort((a,b) => b.weight-a.weight);
    detail += ` Agrega ${contributors.map(item=>`${item.index === selected ? `${item.index} (autolaço)` : item.index}: ${num(item.weight,3)}`).join("; ")}. Pesos de Â, linha ${selected}.`;
    if (stage === "propagation" && hop > 0) {
      const previous = trace.hops[hop - 1];
      const terms = contributors.slice(0, 5).map(item => `${num(item.weight,3)} × ${num(previous[item.index][0],3)}`);
      detail += ` Coordenada 1: ${terms.join(" + ")}${contributors.length > 5 ? " + …" : ""} = ${num(trace.hops[hop][selected][0],4)} (soma com precisão completa).`;
    }
  }
  if ((stage === "prediction" || stage === "prototype") && node.role === "query") {
    const index = selected - episode.support_count;
    detail += ` Predição: ${episode.classes[trace.predictions[index]]}. P(classe real) = ${pct(trace.probabilities[index][node.label])}.`;
  }
  if (stage === "update") detail = `Esta atualização foi calculada com o episódio de treino do passo ${step}, mesmo quando você observa a referência fixa.`;
  $("selected-node").textContent = detail;
  const explanations = {
    episode:`O amostrador escolheu ${cfg.n_way} classes, ${episode.support_count} suportes e ${episode.nodes.length-episode.support_count} consultas. X é concatenado nessa ordem e não é um parâmetro treinável.`,
    graph:cfg.model_name === "protonet" ? "O baseline ProtoNet não constrói grafo. Cada imagem será projetada independentemente." : `A topologia ${cfg.graph_type} é construída a partir de X. As setas destacadas mostram mensagens chegando ao nó selecionado: Â[i,j] multiplica o vetor do nó j para atualizar i. ${cfg.grande_metric !== "off" ? "GRaNDe altera a normalização usando XΘ." : "A normalização usa os graus ponderados de saída e entrada."}`,
    propagation:cfg.model_name === "protonet" ? "No ProtoNet não há saltos. Cada nó passa apenas por XΘ, LayerNorm e normalização L2." : `H⁽${hop}⁾ ${hop ? "resulta da multiplicação Â @ H do salto anterior" : cfg.grande_metric === "off" ? "começa em X; a projeção Θ acontece depois dos saltos" : "começa em XΘ; a projeção já aconteceu antes do GRaNDe"}. Use o controle de salto para acompanhar cada multiplicação. Não há ReLU nem outro peso entre os saltos.`,
    embedding:"As representações passam pela projeção Θ, LayerNorm com γ e β treináveis e normalização L2. Os pontos mostram Z; o modelo compara os vetores completos, e não apenas o plano visível.",
    prototype:`Cada losango é a média das representações dos ${cfg.n_shot} suportes daquela classe. Consultas participam do grafo, mas não entram nessa média. O protótipo não recebe uma normalização L2 adicional.`,
    prediction:"A classe prevista tem a menor distância Euclidiana ao quadrado até um protótipo. A cor das consultas passa a representar a previsão; × indica erro. A escala positiva dos logits também é aprendida.",
    update:`A loss usa os rótulos das consultas do episódio de treino. O backward calcula os gradientes; clipping limita a norma global; AdamW atualiza os parâmetros. Neste passo a loss do mesmo episódio foi de ${num(s.before.loss)} para ${num(s.after.loss)}. Uma queda em cada passo não é garantida.`,
  };
  $("stage-explanation").textContent = explanations[stage];
  renderSource();
  $("metric-loss").textContent = num(trace.loss);
  $("metric-accuracy").textContent = pct(trace.accuracy);
  $("metric-update").textContent = tiny(s.weight_delta_norm);
  let tensors;
  if (stage === "episode") tensors = table(trace.x, `X · dimensão completa: ${episode.nodes.length} × ${data.input_dim}`);
  else if (stage === "graph") tensors = table(trace.adjacency, "A · sem autolaços") + table(trace.normalized_adjacency, "Â · normalizada, com autolaços");
  else if (stage === "propagation") tensors = table(trace.hops[hop], `H do salto ${hop}`);
  else if (stage === "embedding") tensors = table(trace.projected_propagated, "Após projeção e propagação") + table(trace.layernorm, "Após LayerNorm") + table(trace.embeddings, "Z · após normalização L2");
  else if (stage === "prototype") tensors = table(trace.prototypes, "Protótipos · linhas correspondem às classes locais");
  else if (stage === "prediction") tensors = table(trace.distances, "Distâncias ao quadrado") + table(trace.logits, "Logits") + table(trace.probabilities, "Probabilidades · softmax");
  else tensors = table(s.parameters_before.theta, "Θ antes · convenção X @ Θ") + table(s.gradient, "∂ℒ/∂Θ antes do clipping") + table(s.parameters_after.theta, "Θ depois");
  $("tensor-content").innerHTML = tensors;
  drawCurve();
  const currentValidation = s.validation;
  $("evaluation-content").innerHTML = `<p>Classes disjuntas: <strong>${data.splits.train.length} treino / ${data.splits.val.length} validação / ${data.splits.test.length} teste</strong>. ${currentValidation ? `Validação deste passo: <strong>${pct(currentValidation.accuracy)}</strong>, em 10 episódios.` : "Neste passo não houve validação; ela acontece a cada 5 passos e no último."}</p><p>Resultado ao fim da execução: melhor estado no <strong>passo ${data.best_step}</strong>, escolhido pela acurácia de validação. Teste com esse estado: <strong>${pct(data.test.accuracy)} ± ${pct(data.test.ci95)}</strong> (IC 95%, 20 episódios). A navegação acima mostra os estados históricos, sem substituir os pesos pelo melhor estado.</p>`;
}

function drawCurve() {
  const container = $("curve");
  container.replaceChildren();
  const width = Math.max(240, container.clientWidth), height = 195;
  const m = {l:46,r:15,t:14,b:38};
  const maximum = Math.max(0.05, data.initial_probe.loss, ...data.steps.flatMap(s=>[s.before.loss,s.probe.loss])) * 1.12;
  const x = (value) => m.l + value / data.steps.length * (width-m.l-m.r);
  const y = (value) => height-m.b - value/maximum*(height-m.t-m.b);
  const svg = svgElement("svg", {viewBox:`0 0 ${width} ${height}`, role:"img", "aria-label":"Loss ao longo dos passos, do episódio de treino e da referência fixa."});
  container.append(svg);
  for(let i=0;i<4;i++) {
    const value=maximum*i/3;
    svg.append(svgElement("line",{x1:m.l,x2:width-m.r,y1:y(value),y2:y(value),class:"grid"}));
    svg.append(svgElement("text",{x:m.l-8,y:y(value)+4,"text-anchor":"end"},num(value,2)));
  }
  const ticks = [...new Set([0,Math.round(data.steps.length/2),data.steps.length])];
  ticks.forEach(value=>svg.append(svgElement("text",{x:x(value),y:height-m.b+19,"text-anchor":"middle"},value)));
  svg.append(svgElement("text",{x:(width+m.l-m.r)/2,y:height-2,"text-anchor":"middle",class:"axis-title"},"Passo / episódio"));
  svg.append(svgElement("text",{transform:`translate(12 ${height/2}) rotate(-90)`,"text-anchor":"middle",class:"axis-title"},"Entropia cruzada"));
  svg.append(svgElement("rect",{x:m.l,y:m.t,width:width-m.l-m.r,height:height-m.t-m.b,class:"frame"}));
  const series = [
    {color:COLORS[1], points:data.steps.map(s=>[s.step,s.before.loss])},
    {color:COLORS[0], points:[[0,data.initial_probe.loss],...data.steps.map(s=>[s.step,s.probe.loss])]},
  ];
  for(const line of series) {
    const d=line.points.map((point,index)=>`${index?"L":"M"}${x(point[0])},${y(point[1])}`).join(" ");
    svg.append(svgElement("path",{d,stroke:line.color,fill:"none","stroke-width":1.8}));
    const point=line.points.find(p=>p[0]===step);
    if(point) svg.append(svgElement("circle",{cx:x(point[0]),cy:y(point[1]),r:4,fill:line.color}));
  }
  svg.append(svgElement("line",{x1:x(step),x2:x(step),y1:m.t,y2:height-m.b,stroke:"#e6edf7","stroke-dasharray":"3 5",opacity:.55}));
  const hit=svgElement("rect",{x:m.l,y:m.t,width:width-m.l-m.r,height:height-m.t-m.b,fill:"transparent",style:"cursor:crosshair"});
  hit.append(svgElement("title",{},`Passo ${step}: treino ${num(snapshot().before.loss)}, referência ${num(snapshot().probe.loss)}. Clique para mudar de passo.`));
  hit.addEventListener("click",event=>{const box=svg.getBoundingClientRect();stop();chooseStep(Math.round(((event.clientX-box.left)*width/box.width-m.l)/(width-m.l-m.r)*data.steps.length));});
  svg.append(hit);
}

function mathBlock(index, title, body, equation, reference) {
  return `<section class="math-block"><h3><span>${String(index).padStart(2,"0")}</span>${title}</h3>${body}<div class="equation">${equation}</div><p class="code-ref"><code>${reference}</code></p>${mathSource(index)}</section>`;
}

function renderMath() {
  const s=snapshot(), c=data.config, before=s.before, after=s.after, t=step;
  const n=c.n_way*(c.n_shot+c.n_query), ns=c.n_way*c.n_shot, nq=c.n_way*c.n_query;
  const a=s.parameters_before.theta[0][0], b=s.parameters_after.theta[0][0];
  const mhat=s.adam_first_moment/(1-Math.pow(.9,t)), vhat=s.adam_second_moment/(1-Math.pow(.999,t));
  const predicted=(1-c.learning_rate*c.weight_decay)*a-c.learning_rate*mhat/(Math.sqrt(vhat)+1e-8);
  const supports=s.episode.classes.map((name,index)=>`${esc(name)} ↔ ${index}`).join("; ");
  let html=`<div class="math-intro"><span class="eyebrow">PASSO ${String(t).padStart(2,"0")} / EPISÓDIO DO TREINO</span><h2>Do episódio ao novo estado dos pesos</h2><p>Esta aba acompanha o episódio que gerou a atualização ${t}, independentemente do modo de observação na outra aba. θ<sub>t−1</sub> representa todos os parâmetros antes do passo; Θ é especificamente a matriz de projeção. Os valores abaixo vêm do PyTorch.</p></div>`;
  html+=mathBlock(1,"Amostragem e entradas fixas",`<p>O amostrador seleciona ${c.n_way} classes de treino, sem reposição: <strong>${supports}</strong>. Cada classe contribui com ${c.n_shot} suportes e ${c.n_query} consultas. Os dois conjuntos são disjuntos.</p><p>A divisão por classes precede o treino. X é ${data.source==="synthetic"?"sintético neste experimento didático":"o cache real da ResNet congelada"}; não recebe gradiente nem atualização do AdamW.</p>`,
    `S<sub>t</sub> = {(xᵢ,yᵢ)}<sub>i=1</sub><sup>${ns}</sup> &nbsp; Q<sub>t</sub> = {(xⱼ,yⱼ)}<sub>j=1</sub><sup>${nq}</sup><br>X<sub>t</sub> = [X<sub>S</sub>; X<sub>Q</sub>] ∈ ℝ<sup>${n}×${data.input_dim}</sup><br>Θ<sub>t−1</sub> ∈ ℝ<sup>${data.input_dim}×${c.output_dim}</sup>`,"episodes.py · split_classes / FeatureEpisodeSampler.sample");
  let graphEquation, graphText;
  if(c.model_name==="protonet") {
    graphEquation="Sem construção de A. A propagação é omitida: U = XΘ.";
    graphText="O baseline ignora a construção do grafo e projeta cada vetor independentemente.";
  } else if(c.graph_type.startsWith("knn")) {
    graphEquation=`dᵢⱼ = ‖xᵢ−xⱼ‖₂<br>Bᵢⱼ = 𝟙[j está entre os k menores dᵢⱼ, j ≠ i]<br>A = ${c.graph_type==="knn-in"?"Bᵀ":c.graph_type==="knn-union"?"max(B,Bᵀ)":c.graph_type==="knn-reciprocal"?"min(B,Bᵀ)":"B"}<br>k efetivo = ${Math.min(c.knn,n-1)}; arestas padrão ∈ {0,1}`;
    graphText="Os vizinhos vêm da distância Euclidiana em X, como no BallTree original. A diagonal é excluída nessa etapa e as arestas valem 1. Cada linha i de A informa quais nós j fornecem suas representações para i em A @ X.";
  } else {
    graphText=`As listas ranqueadas por ${c.dgcg_metric==="cosine"?"cosseno":"distância Euclidiana"} incluem o próprio nó em primeiro lugar. A correlação é ${esc(c.dgcg_correlation)}; o próprio nó é retirado dos candidatos a aresta. ${c.dgcg_threshold===null?"O limiar automático procura o grau médio mais próximo do intervalo [4,6], limitado ao número de candidatos, preserva empates e acrescenta o melhor candidato de um nó que ficaria isolado.":`O limiar manual aceita somente correlações estritamente maiores que ${num(c.dgcg_threshold,2)}; não aplica o fallback para nós isolados.`}`;
    graphEquation=c.dgcg_correlation==="rbo"?`corr(i,j) = (1−p) Σ<sub>d=1</sub><sup>k</sup> p<sup>d−1</sup> |Rᵢ[:d] ∩ Rⱼ[:d]| / d<br>p = 0,9; k efetivo = ${Math.min(c.dgcg_top_k,n)}`:`J<sub>d</sub>(i,j) = |Rᵢ[:d] ∩ Rⱼ[:d]| / (2d − |Rᵢ[:d] ∩ Rⱼ[:d]|)<br>corr(i,j) = ${c.dgcg_correlation==="jaccardk"?"média":c.dgcg_correlation==="jaccard-max"?"máximo":"mediana"}<sub>d=1…k</sub> J<sub>d</sub>(i,j)`;
    if(c.graph_type==="dgcg-plus") graphEquation+="<br>sᵢ = (1/k²) Σ<sub>j∈Rᵢ[:k]</sub> Σ<sub>v∈Rᵢ[:k]∩Rⱼ[:k]</sub> 1/rankᵢ(v)<br>Peso de (i,j) = (sᵢ+sⱼ)/2";
    graphEquation+="<br>A = Bᵀ, para converter a convenção origem→destino em agregação por linha.";
  }
  if(c.cosine_rbf_weight) graphEquation+="<br>Ablação opcional: sᵢⱼ = ⟨xᵢ/‖xᵢ‖₂, xⱼ/‖xⱼ‖₂⟩<br>Aᵢⱼ = 𝟙[Aᵢⱼ ≠ 0] · exp(−(1−sᵢⱼ)²/(2σ²))";
  html+=mathBlock(2,"Topologia e pesos das arestas",`<p>${graphText}</p><p>Os rótulos y<sub>Q</sub> não são recebidos por nenhum construtor de grafo. Cores na visualização são apenas anotações.</p>`,graphEquation,"model.py · graph_adjacency → graphs/knn.py ou graphs/dgcg.py");
  let normEquation="Ã = A + I<br>dᵢ<sup>out</sup> = Σⱼ Ãᵢⱼ; &nbsp; dⱼ<sup>in</sup> = Σᵢ Ãᵢⱼ<br>Âᵢⱼ = Ãᵢⱼ / √(dᵢ<sup>out</sup> dⱼ<sup>in</sup>)";
  let normText="Os graus são somas dos pesos. Para grafos simétricos, os graus de entrada e saída coincidem. Autolaços são adicionados depois da ponderação das arestas.";
  if(c.grande_metric!=="off") {
    normText=`GRaNDe mede as distâncias em XΘ, com métrica ${esc(c.grande_metric)}. Conta a existência de cada aresta, incluindo autolaços; os pesos de A continuam na propagação. O min-max é global sobre as arestas de cada orientação. O código destaca (detach) mínimo e amplitude, preservando o gradiente apenas pelo numerador da distância.`;
    normEquation=`Ã = A + I; &nbsp; H⁽⁰⁾ = XΘ<br>δᵢⱼ = max(dist(Hᵢ⁽⁰⁾,Hⱼ⁽⁰⁾),10⁻¹²)<br>δ̃ᵢⱼ = (δᵢⱼ − stopgrad(min δ)) / max(stopgrad(max δ − min δ),10⁻¹²)<br>gᵢ = |Nᵢ| + (1/|Nᵢ|) Σ<sub>j∈Nᵢ</sub> 1/max(exp(−δ̃ᵢⱼ²/σ),10⁻¹²)<br>Âᵢⱼ = Ãᵢⱼ / √(gᵢ<sup>out</sup> gⱼ<sup>in</sup>); &nbsp; σ = ${num(c.grande_sigma,2)}`;
    normText+=" Se a amplitude das distâncias é ≤10⁻¹², a distância normalizada é zero. Graus de entrada usam a adjacência transposta.";
    if(c.grande_metric==="rbo") normText+=" Aqui dist = 1−RBO, com p = "+num(c.grande_rbo_p,2)+". Rankings são discretos e calculados sem gradiente; neste modo o grau não fornece uma derivada através da ordenação.";
  }
  if(c.model_name==="protonet") {normText="Esta etapa não é executada pelo baseline. A identidade que a interface usa para representar a ausência de mistura é apenas uma convenção de visualização.";normEquation="Sem autolaços ou normalização de adjacência no forward do ProtoNet.";}
  else normEquation+=`<br>Nó 0, antes do passo: grau de saída = <span class="numeric">${num(before.out_degree[0])}</span>; Â₀₀ = <span class="numeric">${num(before.normalized_adjacency[0][0])}</span>`;
  html+=mathBlock(3,"Normalização da adjacência",`<p>${normText}</p>`,normEquation,"model.py · normalized_graph_adjacency / graphs/grande.py");
  const propagation=c.model_name==="protonet"?"U = XΘ":c.grande_metric==="off"?`H⁽⁰⁾ = X; &nbsp; H⁽ʳ⁺¹⁾ = ÂH⁽ʳ⁾; &nbsp; U = H⁽${c.sgc_hops}⁾Θ`:`H⁽⁰⁾ = XΘ; &nbsp; H⁽ʳ⁺¹⁾ = ÂH⁽ʳ⁾; &nbsp; U = H⁽${c.sgc_hops}⁾`;
  html+=mathBlock(4,"Propagação e representação",`<p>${c.grande_metric==="off"?"Sem GRaNDe, a adjacência independe de Θ para um episódio fixo; no Proto-SGC a projeção ocorre depois da propagação.":"Com GRaNDe, a projeção ocorre antes da normalização e dos saltos. A topologia continua baseada em X."} Não há ativações nem pesos diferentes entre os saltos do SGC.</p><p>A LayerNorm calcula média e variância por nó nas ${c.output_dim} coordenadas, com variância populacional. Seus γ e β também são treináveis. A normalização L2 usa proteção numérica de 10⁻¹².</p>`,
    `${propagation}<br>μᵢ = (1/D)Σₐ Uᵢₐ; &nbsp; vᵢ = (1/D)Σₐ(Uᵢₐ−μᵢ)²<br>Lᵢ = γ ⊙ (Uᵢ−μᵢ)/√(vᵢ+10⁻⁵) + β<br>Zᵢ = Lᵢ/max(‖Lᵢ‖₂,10⁻¹²)<br>Z₀₁: <span class="numeric">${num(before.embeddings[0][0],6)} → ${num(after.embeddings[0][0],6)}</span> no mesmo episódio`,"model.py · encode_episode");
  html+=mathBlock(5,"Protótipos, logits e loss",`<p>O protótipo é a média somente dos suportes da classe. Ele não é renormalizado após a média. Cada consulta produz uma distribuição de probabilidade sobre os ${c.n_way} protótipos; os rótulos das consultas entram agora para calcular a loss.</p>`,
    `p<sub>c</sub> = (1/${c.n_shot}) Σ<sub>i∈S, yᵢ=c</sub> Zᵢ<br>α = min(exp(a),100); &nbsp; ℓⱼc = −α‖Zⱼ−p<sub>c</sub>‖₂²<br>P(y=c|xⱼ) = exp(ℓⱼc)/Σ<sub>r</sub>exp(ℓⱼr)<br>ℒ<sub>t</sub> = −(1/${nq})Σ<sub>j∈Q</sub>log P(y=yⱼ|xⱼ)<br>α = ${num(before.scale)}; ℒ antes = <span class="numeric">${num(before.loss,6)}</span><br>Primeira consulta: y = ${s.episode.query_labels[0]}, previsão = ${before.predictions[0]}, P(y real) = ${num(before.probabilities[0][s.episode.query_labels[0]],6)}`,
    "model.py · forward / training.py · F.cross_entropy");
  html+=mathBlock(6,"Backward e clipping global",`<p>O gradiente inclui Θ, γ, β e a escala a. A escolha de vizinhos e as ordenações são discretas. Com GRaNDe Euclidiano ou cosseno, a dependência diferenciável da normalização em XΘ também participa do backward.</p><p>O clipping usa uma única norma sobre todos os parâmetros, e não uma norma separada para Θ. O gradiente negativo aponta uma direção local; o passo final também depende dos momentos do AdamW.</p>`,
    `g = ∇<sub>θ</sub> ℒ<sub>t</sub>(θ<sub>t−1</sub>)<br>c = min(1, ${num(c.grad_clip,2)}/(‖g‖₂+10⁻⁶)); &nbsp; g̃ = c·g<br>‖g‖₂ = <span class="numeric">${num(s.gradient_norm,6)}</span>; c = ${num(s.clip_factor,6)}<br>∂ℒ/∂Θ₁₁ = ${num(s.gradient[0][0],6)} → após clipping = ${num(s.clipped_gradient[0][0],6)}`,
    "training.py · loss.backward / nn.utils.clip_grad_norm_");
  html+=mathBlock(7,"AdamW: a atualização deste passo",`<p>O otimizador mantém média móvel do gradiente e do seu quadrado. A correção de viés usa o número do passo ${t}. O weight decay é desacoplado do gradiente, como em <code>torch.optim.AdamW</code>.</p><p>Exemplo numérico do coeficiente Θ₁₁. A matriz exibida usa a convenção X @ Θ; <code>nn.Linear.weight</code> armazena a transposta.</p>`,
    `m<sub>t</sub> = 0,9m<sub>t−1</sub> + 0,1g̃<sub>t</sub><br>v<sub>t</sub> = 0,999v<sub>t−1</sub> + 0,001g̃<sub>t</sub>²<br>m̂<sub>t</sub> = m<sub>t</sub>/(1−0,9<sup>t</sup>); &nbsp; v̂<sub>t</sub> = v<sub>t</sub>/(1−0,999<sup>t</sup>)<br>θ<sub>t</sub> = (1−ηλ)θ<sub>t−1</sub> − ηm̂<sub>t</sub>/(√v̂<sub>t</sub>+10⁻⁸)<br>η = ${num(c.learning_rate,4)}; λ = ${num(c.weight_decay,4)}<br>m̂₁₁ = ${num(mhat,7)}; v̂₁₁ = ${tiny(vhat)}<br>Θ₁₁: <span class="numeric">${num(a,7)} → ${num(b,7)}</span><br>Fórmula com os momentos registrados: ${num(predicted,7)} (diferenças apenas de arredondamento)`,
    "training.py · torch.optim.AdamW / optimizer.step");
  html+=mathBlock(8,"Reavaliação e seleção por validação",`<p>Após atualizar os pesos, o simulador repete o mesmo episódio sem gradientes para mostrar o efeito isolado da atualização. Essa reavaliação e a referência fixa são instrumentos de observação; não geram novos passos do otimizador.</p><p>A validação usa sempre a mesma sequência de 10 episódios, em outras classes. O melhor estado da execução completa é o passo ${data.best_step}; só ele é usado nos 20 episódios de teste, de classes novamente disjuntas. A seleção usa apenas a acurácia de validação, mantendo o primeiro estado em caso de empate.</p>`,
    `ℒ(S<sub>t</sub>,Q<sub>t</sub>;θ<sub>t−1</sub>) = ${num(before.loss,6)}<br>ℒ(S<sub>t</sub>,Q<sub>t</sub>;θ<sub>t</sub>) = ${num(after.loss,6)}<br>Acurácia no mesmo episódio: ${pct(before.accuracy)} → ${pct(after.accuracy)}<br>${s.validation?`Validação neste passo: ${pct(s.validation.accuracy)}`:"Sem validação neste passo."}<br>IC95% = 1,96 · desvio-padrão amostral das acurácias / √n<sub>episódios</sub>`,
    "training.py · evaluate / seleção do melhor state_dict");
  $("math-content").innerHTML=html;
}

function render() {
  if(!data) return;
  $("step-number").textContent=String(step).padStart(2,"0");
  $("step-total").textContent=` / ${data.steps.length}`;
  $("step-slider").disabled=false;
  $("step-slider").max=data.steps.length;
  $("step-slider").value=step;
  $("previous").disabled=step===1;
  $("next").disabled=step===data.steps.length;
  $("play").disabled=false;
  $("checkpoint-label").textContent=snapshot().validation?`Validação neste passo: ${pct(snapshot().validation.accuracy)}`:"Validação a cada 5 passos e no último";
  if(currentTab==="simulation") renderSimulation();
  else renderMath();
}

STAGES.forEach((definition,index)=>{
  const button=document.createElement("button");
  button.type="button";
  button.dataset.stage=definition.id;
  button.setAttribute("aria-pressed",String(index===0));
  button.innerHTML=`<span>${String(index+1).padStart(2,"0")}</span>${definition.label}`;
  button.addEventListener("click",()=>{stage=definition.id;if(data)renderSimulation();});
  $("stage-nav").append(button);
});
form.addEventListener("submit",run);
form.addEventListener("change",()=>{enabledFields();if(data&&!running)$("status").textContent="Parâmetros alterados. Execute novamente para aplicar; a visualização ainda mostra a execução anterior.";});
form.elements.noise.addEventListener("input",()=>$("noise-value").textContent=num(Number(form.elements.noise.value),2));
$("previous").addEventListener("click",()=>{stop();chooseStep(step-1);});
$("next").addEventListener("click",()=>{stop();chooseStep(step+1);});
$("step-slider").addEventListener("input",event=>{stop();chooseStep(event.target.value);});
$("play").addEventListener("click",()=>{
  if(timer){stop();return;}
  if(step===data.steps.length)chooseStep(1);
  $("play").textContent="Ⅱ Pausar";
  timer=setInterval(()=>{if(step>=data.steps.length){stop();return;}chooseStep(step+1);},900);
});
$("episode-view").addEventListener("change",()=>{selected=0;if(data)renderSimulation();});
$("state-before").addEventListener("click",()=>{phase="before";if(data)renderSimulation();});
$("state-after").addEventListener("click",()=>{phase="after";if(data)renderSimulation();});
$("hop-slider").addEventListener("input",event=>{hop=Number(event.target.value);if(data)renderSimulation();});
$("source-selector").addEventListener("change",event=>{selectedSource=event.target.value;if(data)renderSource();});
$("math-content").addEventListener("toggle",event=>{
  const key=event.target.dataset.codeKey;
  if(key){if(event.target.open)openCodeSections.add(key);else openCodeSections.delete(key);}
},true);
for(const name of ["simulation","math"]) {
  $(`tab-${name}`).addEventListener("click",()=>setTab(name));
  $(`tab-${name}`).addEventListener("keydown",event=>{if(event.key==="ArrowLeft"||event.key==="ArrowRight"){event.preventDefault();const next=name==="math"?"simulation":"math";setTab(next);$(`tab-${next}`).focus();}});
}
let resizeTimer;
new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(data&&currentTab==="simulation")renderSimulation();},100);}).observe($("plot"));
async function initialize(){
  try{
    const response=await fetch("/api/config");
    if(!response.ok)throw new Error("Não foi possível acessar o servidor local.");
    const info=await response.json();source=info.source;
    for(const [name,value] of Object.entries(info.defaults)){
      const input=form.elements[name];if(!input)continue;
      if(input.type==="checkbox")input.checked=value;else input.value=value===null?"":value;
    }
    try {
      const response=await fetch("/api/code");
      if(!response.ok)throw new Error("Trechos indisponíveis. Reinicie o servidor do simulador e atualize a página.");
      snippets=await response.json();
    } catch(error) {codeError=error.message;}
    enabledFields();
    await run();
  }catch(error){$("error").hidden=false;$("error").textContent=error.message;$("status").textContent="Inicie o aplicativo com python -m simulador e abra o endereço mostrado no terminal.";}
}
initialize();
