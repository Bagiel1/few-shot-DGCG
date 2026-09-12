/* Smoke test do JavaScript com DOM mínimo, sem abrir um navegador.
 * Exercita os controles usando snapshots reais; não verifica layout visual.
 * Execute na raiz: node tests/check_simulator_ui.cjs
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const {execFileSync} = require("node:child_process");

const python = process.env.SIMULATOR_PYTHON || (fs.existsSync("venv/bin/python") ? "venv/bin/python" : "python3");
const fixtureData = JSON.parse(execFileSync(python, ["-c", `
import json
import torch
from simulador.engine import parse_config, simulate
from simulador.source_code import code_catalog
torch.set_num_threads(1)
configs = [
    {'steps': 3, 'n_way': 2, 'n_shot': 1, 'n_query': 2},
    {'steps': 2, 'n_way': 2, 'model_name': 'protonet'},
    {'steps': 2, 'n_way': 2, 'graph_type': 'dgcg-plus', 'grande_metric': 'rbo', 'cosine_rbf_weight': True},
]
print(json.dumps({'runs': [simulate(parse_config(cfg)) for cfg in configs], 'code': code_catalog()}, allow_nan=False))
`], {encoding: "utf8", maxBuffer: 20 * 1024 * 1024}));
const fixture = fixtureData.runs;

class Element {
  constructor(tag = "div", attrs = {}) {
    this.tagName = tag;
    this.attrs = {...attrs};
    this.children = [];
    this.events = {};
    this.dataset = {};
    this.clientWidth = 740;
    this.value = attrs.value || "";
    this.name = attrs.name || "";
    this.type = attrs.type || (tag === "select" ? "select-one" : "");
    this.disabled = false;
    this.checked = false;
    this.textContent = "";
    this.innerHTML = "";
  }
  append(...children) {this.children.push(...children);}
  replaceChildren(...children) {this.children = children; this.innerHTML = "";}
  setAttribute(key, value) {this.attrs[key] = String(value);}
  addEventListener(type, callback) {this.events[type] = callback;}
  querySelectorAll(tag) {return this.children.filter(child => child.tagName === tag);}
  getBoundingClientRect() {return {left:0, top:0, width:this.clientWidth, height:350};}
  reportValidity() {return true;}
  focus() {}
  async dispatch(type, extra = {}) {
    if (this.events[type]) await this.events[type]({target:this, preventDefault(){}, ...extra});
  }
}

function attrs(text) {
  return Object.fromEntries([...text.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => [match[1], match[2]]));
}

const html = fs.readFileSync("simulador/static/index.html", "utf8");
const ids = new Map();
for (const match of html.matchAll(/<([\w-]+)\b([^>]*)>/g)) {
  const attributes = attrs(match[2]);
  if (attributes.id) ids.set(attributes.id, new Element(match[1], attributes));
}
const elements = [];
for (const match of html.matchAll(/<(input)\b([^>]*)>|<(select)\b([^>]*)>([\s\S]*?)<\/select>/g)) {
  const tag = match[1] || match[3];
  const attributes = attrs(match[2] || match[4]);
  const element = attributes.id ? ids.get(attributes.id) : new Element(tag, attributes);
  if (tag === "select") {
    const option = (match[5] || "").match(/<option(?:\s+value="([^"]*)")?[^>]*>([^<]*)/);
    element.value = option ? option[1] === undefined ? option[2] : option[1] : "";
  }
  if (attributes.name) {elements.push(element); elements[attributes.name] = element;}
}
ids.get("configuration").elements = elements;
elements.push(ids.get("run"));

let requested, failNext = false;
const context = vm.createContext({
  document: {
    getElementById(id) {assert(ids.has(id), `ID inexistente: ${id}`); return ids.get(id);},
    createElement(tag) {return new Element(tag);},
    createElementNS(namespace, tag) {return new Element(tag);},
  },
  performance: {now: () => Date.now()},
  ResizeObserver: class {observe() {}},
  setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
  async fetch(url, options) {
    if (url === "/api/config") return {ok:true, json:async()=>({defaults:fixture[0].config, source:"synthetic"})};
    if (url === "/api/code") return {ok:true, json:async()=>fixtureData.code};
    assert.equal(url, "/api/simulate");
    requested = JSON.parse(options.body);
    if (failNext) {failNext=false; return {ok:false, json:async()=>({error:"Falha de teste"})};}
    return {ok:true, json:async()=>fixture[0]};
  },
});
const source = fs.readFileSync("simulador/static/app.js", "utf8").replace(/initialize\(\);\s*$/, "globalThis.boot = initialize();");
vm.runInContext(source, context);

(async () => {
  await context.boot;
  assert.equal(ids.get("step-number").textContent, "01");
  assert(ids.get("plot").children.length > 0, "Nenhum SVG desenhado");
  assert.equal(requested.steps, 3);
  assert(ids.get("source-content").innerHTML.includes("proto_sgc/episodes.py"));
  assert(ids.get("source-content").innerHTML.includes("data-line="));
  await ids.get("next").dispatch("click");
  assert.equal(ids.get("step-number").textContent, "02");
  await ids.get("previous").dispatch("click");
  await ids.get("state-after").dispatch("click");
  assert.equal(ids.get("state-after").attrs["aria-pressed"], "true");
  ids.get("episode-view").value = "probe";
  await ids.get("episode-view").dispatch("change");
  assert(ids.get("view-hint").textContent.includes("Mesmas imagens"));

  for (const result of fixture) {
    context.result = result;
    vm.runInContext("data = result; step = 1; selected = 0;", context);
    for (const width of [360, 740]) {
      ids.get("plot").clientWidth = width;
      ids.get("curve").clientWidth = width;
      for (const mode of ["train", "probe"]) {
        ids.get("episode-view").value = mode;
        for (const moment of ["before", "after"]) {
          await ids.get(`state-${moment}`).dispatch("click");
          for (const button of ids.get("stage-nav").children) {
            await button.dispatch("click");
            assert(!ids.get("selected-node").textContent.includes("undefined"));
            assert(!ids.get("tensor-content").innerHTML.includes("NaN"));
            assert(ids.get("source-content").innerHTML.includes("source-code"));
            assert(!ids.get("source-content").innerHTML.includes("não está disponível"));
            if (button.dataset.stage === "graph") {
              const expected = result.config.model_name === "protonet" ? "encode" : result.config.graph_type.startsWith("dgcg") ? "dgcg" : "knn";
              assert.equal(ids.get("source-selector").value, expected);
            }
          }
        }
      }
    }
    await ids.get("tab-math").dispatch("click");
    assert(ids.get("math-content").innerHTML.includes("AdamW"));
    assert(!ids.get("math-content").innerHTML.includes("NaN"));
    assert(!ids.get("math-content").innerHTML.includes("undefined"));
    assert.equal((ids.get("math-content").innerHTML.match(/class="math-source"/g) || []).length, 8);
    assert(!ids.get("math-content").innerHTML.includes("não está disponível"));
    ids.get("step-slider").value = result.steps.length;
    await ids.get("step-slider").dispatch("input");
    assert(ids.get("math-content").innerHTML.includes(`PASSO ${String(result.steps.length).padStart(2,"0")}`));
    await ids.get("tab-simulation").dispatch("click");
  }
  elements.model_name.value = "protonet";
  await ids.get("configuration").dispatch("change");
  assert(elements.graph_type.disabled);
  assert.equal(elements.grande_metric.value, "off");
  assert.equal(elements.cosine_rbf_weight.checked, false);
  const sourcePicker = ids.get("source-selector");
  sourcePicker.value = "observed_step";
  await sourcePicker.dispatch("change");
  assert(ids.get("source-content").innerHTML.includes("simulador/engine.py"));
  assert(vm.runInContext('pythonLine("<script>alert(1)</script>").includes("&lt;script&gt;")', context));
  failNext = true;
  await ids.get("configuration").dispatch("submit");
  assert.equal(ids.get("error").textContent, "Falha de teste");
  assert.equal(ids.get("run").disabled, false);
  console.log("Interface: navegação, 7 etapas, antes/depois, referência fixa, fórmulas e erros verificados com snapshots reais (DOM de teste, sem QA visual).");
})().catch(error => {console.error(error); process.exitCode = 1;});
