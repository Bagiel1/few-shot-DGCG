"""Audita a planilha e as fontes antes de planejar a bateria de backbones.

Somente leitura: nao reconstroi a planilha nem modifica resultados historicos.
"""

import csv
import hashlib
import json
import math
import posixpath
import statistics
from xml.etree import ElementTree as ET
from zipfile import ZipFile


WORKBOOK = "resultados/consolidacao_confiavel_20261103/resultados_fewshot_fonte_unica.xlsx"
INVENTORY = "resultados/consolidacao_confiavel_20261103/inventario_fontes.csv"
SCALES = "resultados/consolidado_fewshot_20261102/fontes/resultados_comparacao.csv"
CLAUDE = "resultados/testes_claude_20261102/resultados/res_e1.jsonl"
HISTORICAL_METRICS = "resultados/analise_experimentos_20261026/fontes/resultados_metricas.csv"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workbook_results(path):
    """Le valores da aba Resultados; localiza a aba e o cabecalho pelo nome."""
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.itertext()) for t in ET.fromstring(
                archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next(s for s in workbook.findall("s:sheets/s:sheet", NS) if s.get("name") == "Resultados")
        relationship = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in rels if r.get("Id") == relationship)
        target = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        rows = ET.fromstring(archive.read(target)).findall("s:sheetData/s:row", NS)
        header = None
        records = []
        for row in rows:
            cells = {}
            for cell in row.findall("s:c", NS):
                column = "".join(c for c in cell.get("r") if c.isalpha())
                value = cell.find("s:v", NS)
                value = value.text if value is not None else ""
                if cell.get("t") == "s":
                    value = strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(cell.find("s:is", NS).itertext())
                cells[column] = value
            if cells.get("A") == "ID":
                header = cells
            elif header is not None and cells.get("A"):
                records.append({**{title: cells.get(col, "") for col, title in header.items() if title},
                                "sheet_row": int(row.get("r"))})
        if not records:
            raise ValueError("Planilha sem a tabela de resultados esperada.")
        return records


def _verified_source(root, relative, inventory):
    path = root / relative
    entries = [r for r in inventory if r["arquivo"] == relative and r["status"] == "OK"]
    if len(entries) != 1 or sha256(path) != entries[0]["sha256"]:
        raise ValueError(f"Fonte ausente, alterada ou nao auditada: {relative}")
    return path


def audit_results(root):
    """Confere as fontes de escalas, graus e referencias CPU contra a planilha."""
    with (root / INVENTORY).open(newline="") as stream:
        inventory = list(csv.DictReader(stream))
    scale_path = _verified_source(root, SCALES, inventory)
    claude_path = _verified_source(root, CLAUDE, inventory)
    degree_path = _verified_source(root, HISTORICAL_METRICS, inventory)
    workbook = _workbook_results(root / WORKBOOK)
    with scale_path.open(newline="") as stream:
        raw = [(line, row) for line, row in enumerate(csv.DictReader(stream), 2)
               if row["dataset"] in ("flowers_micro", "cars_micro")]
    table = [r for r in workbook if r["ID"].startswith("ESCALAS-") and r["Versão"] == "Micro"]
    if len(table) != 12 or len(raw) != 36:
        raise ValueError("Esperava 12 medias Micro na planilha e 36 execucoes nas fontes.")
    covered = set()
    records = []
    method_names = {"dgcg": "DGCG", "knn-union": "kNN-union", "no-graph": "Sem grafo"}
    for line, row in raw:
        key = (row["dataset"], int(row["n_shot"]), row["metodo"], int(row["seed"]))
        if key in covered:
            raise ValueError(f"Resultado historico duplicado: {key}")
        covered.add(key)
        if not (row["version"] == "micro" and row["n_way"] == "5" and row["n_query"] == "10"
                and json.loads(row["class_split"]) == [10, 5, 5]
                and row["val_episodes"] == "200" and row["test_episodes"] == "1000"):
            raise ValueError(f"Protocolo historico incompativel na linha {line}.")
        candidates = [t for t in table if t["Escopo das bases"] == row["dataset"]
                      and int(t["Shot"]) == int(row["n_shot"]) and t["Método"] == method_names[row["metodo"]]]
        if len(candidates) != 1:
            raise ValueError(f"Fonte sem correspondencia unica na planilha: {key}")
        canonical = candidates[0]
        if not (canonical["Way"] == "5" and canonical["Query"] == "10"
                and canonical["Seeds"] == "7;17;27" and canonical["Dimensão"] == "256"
                and canonical["Inicialização"] == "Aleatória" and canonical["Pré-processamento"] == "raw"
                and canonical["Treino"] == "Padrão, 2.000 episódios"
                and canonical["Dispositivo"] == "GPU VM (informado; não consta no CSV)"):
            raise ValueError(f"Metadados da planilha mudaram: {canonical['ID']}")
        group = [r for _, r in raw if (r["dataset"], r["n_shot"], r["metodo"]) ==
                 (row["dataset"], row["n_shot"], row["metodo"])]
        if sorted(int(r["seed"]) for r in group) != [7, 17, 27]:
            raise ValueError(f"Seeds historicas incompletas: {key}")
        for field, column in (("val_accuracy", "Validação (%)"), ("test_accuracy", "Teste (%)"),
                              ("test_macro_f1", "F1 macro (%)")):
            if not math.isclose(statistics.mean(float(r[field]) for r in group),
                                float(canonical[column]), abs_tol=1e-8, rel_tol=0):
                raise ValueError(f"Planilha diverge da fonte: {canonical['ID']}, {field}")
        expected = "densidade " + row["alvo"] if row["metodo"] == "dgcg" else (
            "k=5" if row["metodo"] == "knn-union" else "sem grafo")
        if canonical["Configuração"] != expected:
            raise ValueError(f"Configuracao historica divergente: {canonical['ID']}")
        records.append(dict(dataset=row["dataset"], n_shot=int(row["n_shot"]), metodo=row["metodo"],
                            seed=int(row["seed"]), source=SCALES, source_line=line,
                            workbook_id=canonical["ID"], workbook_row=canonical["sheet_row"],
                            device=canonical["Dispositivo"], source_row=row))
    degree_records = _degree_records(degree_path, workbook)
    claude_rows = [(line, json.loads(text)) for line, text in enumerate(claude_path.read_text().splitlines(), 1)]
    # A tabela agrega dez bases: conferir essa media antes de listar referencias
    # individuais de Flowers/Cars. Uma media de dez bases nao substitui uma base.
    for shot in (1, 5):
        for method in ("no-graph", "union", "dgcg"):
            for train_free in (False, True):
                identifier = f"CLAUDE-E1-{shot}S-{method}-{'TF512' if train_free else 'TR256'}"
                canonical = [r for r in workbook if r["ID"] == identifier]
                group = [r for _, r in claude_rows if r["shot"] == shot and r["method"] == method
                         and r["trainfree"] == train_free and r["feat"] == "raw" and r["tag"] == ""]
                if len(canonical) != 1 or len(group) != 10 or canonical[0]["Dispositivo"] != "CPU local":
                    raise ValueError(f"Referencia CPU sem correspondencia na planilha: {identifier}")
                for field, column in (("val", "Validação (%)"), ("test", "Teste (%)"), ("f1", "F1 macro (%)")):
                    if not math.isclose(statistics.mean(r[field] for r in group), float(canonical[0][column]), abs_tol=1e-8, rel_tol=0):
                        raise ValueError(f"Referencia CPU diverge da planilha: {identifier}")
    cpu_references = []
    for line, row in claude_rows:
        if row["dataset"] in ("flowers_micro", "cars_micro") and row["feat"] == "raw" and row["tag"] == "":
            cpu_references.append(dict(source=CLAUDE, source_line=line, device="cpu", result=row,
                                       exclusion="CPU local, seed7; nao substitui a grade CUDA. DGCG deste arquivo usa grau4-6."))
    return dict(workbook=WORKBOOK, workbook_sha256=sha256(root / WORKBOOK),
                sources_sha256={name: sha256(root / name) for name in (INVENTORY, SCALES, CLAUDE, HISTORICAL_METRICS)},
                validated_workbook_groups=len(table), gpu_trained_records=records,
                gpu_degree_records=degree_records,
                cpu_references=cpu_references,
                limitation="GPU VM informada na planilha; modelo da GPU e versoes historicas nao constam no CSV. Sem checkpoint importado.")


def _degree_records(path, workbook):
    """Confere as medias historicas e recupera oito execucoes por base/protocolo.

    A media ALL27 serve para auditar a fonte, nunca como resultado individual.
    """
    with path.open(newline="") as stream:
        raw = list(enumerate(csv.DictReader(stream), 2))
    records = []
    for target, method in (("4:6", "dgcg-4-6"), ("2:4", "dgcg-2-4")):
        identifier = "HIST-ALL27-dgcg-" + target
        canonical = [r for r in workbook if r["ID"] == identifier]
        group = [r for _, r in raw if r["metodo"] == "dgcg" and r["alvo"] == target]
        if len(canonical) != 1 or len(group) != 270:
            raise ValueError(f"Fonte historica de graus incompleta: {identifier}")
        canonical = canonical[0]
        if not (canonical["Configuração"] == "grau " + target and canonical["Dimensão"] == "256"
                and canonical["Treino"] == "Padrão, 2.000 episódios"
                and canonical["Dispositivo"] == "GPU VM (informado; não consta no CSV)"):
            raise ValueError(f"Protocolo historico de graus divergente: {identifier}")
        for field, column in (("val_accuracy", "Validação (%)"), ("test_accuracy", "Teste (%)"),
                              ("test_macro_f1", "F1 macro (%)")):
            if not math.isclose(statistics.mean(float(r[field]) for r in group),
                                float(canonical[column]), abs_tol=1e-8, rel_tol=0):
                raise ValueError(f"Planilha diverge da fonte de graus: {identifier}")
        selected = [(line, row) for line, row in raw if row["metodo"] == "dgcg" and row["alvo"] == target
                    and row["dataset"] in ("flowers_micro", "cars_micro") and row["n_way"] == "5"
                    and row["n_shot"] in ("1", "5") and row["n_query"] == "10" and row["seed"] == "7"]
        if len(selected) != 4 or len({(r["dataset"], r["n_shot"]) for _, r in selected}) != 4:
            raise ValueError(f"Execucoes por base/shot ausentes ou duplicadas: {identifier}")
        for line, row in selected:
            if row["val_episodes"] != "200" or row["test_episodes"] != "1000":
                raise ValueError(f"Orcamento historico de graus divergente: {line}")
            records.append(dict(dataset=row["dataset"], n_shot=int(row["n_shot"]), metodo=method, seed=7,
                                source=HISTORICAL_METRICS, source_line=line, workbook_id=identifier,
                                workbook_row=canonical["sheet_row"], device=canonical["Dispositivo"],
                                source_row=dict(row, target_kind="degree")))
    return records


def reusable_result(job, args, audit):
    """Nao reaproveita CPU como CUDA, grau como densidade ou quick como completo."""
    if args.quick or args.weights != "imagenet" or args.device != "cuda":
        return None
    if job["backbone"] != "resnet18" or job["image_size"] != 128 or job["mode"] != "trained":
        return None
    candidates = audit["gpu_trained_records"] + audit["gpu_degree_records"]
    found = [r for r in candidates if all(r[k] == job[k] for k in ("dataset", "n_shot", "metodo", "seed"))]
    if len(found) > 1:
        raise ValueError("Mais de um resultado historico para a mesma condicao.")
    if not found:
        return None
    record = found[0]
    row = record["source_row"]
    if row["target_kind"] != job["target_kind"] or row["alvo"] != job["alvo"]:
        return None
    return record


def historical_row(job, record):
    """Mantem as metricas originais em porcentagem; nao inventa tempos/estados."""
    row = {k: v for k, v in job.items() if k not in ("directory", "reuse")}
    for key, value in record["source_row"].items():
        if key.startswith(("val_", "test_")):
            if value in ("True", "False"):
                row[key] = value == "True"
            else:
                row[key] = float(value) if value != "" else ""
    row.update(origin="reused", source_device=record["device"], source=record["source"],
               source_line=record["source_line"], workbook_id=record["workbook_id"],
               workbook_row=record["workbook_row"], elapsed_seconds="")
    return row
