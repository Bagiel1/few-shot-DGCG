"""Flowers/Cars Micro: ResNet/ViT, com e sem treino, em episodios pareados."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time

from experimentos_extras import atomic_json, source_digest
from experimentos_metricas_graus import ROOT, flatten, save_csv
from proto_sgc.result_inventory import audit_results, historical_row, reusable_result


DATASETS = {"flowers_micro": 44239, "cars_micro": 44245}
# Selecao historica por validacao na ResNet/128 treinada, seed 7.
# Transferimos estes alvos; nao selecionamos hiperparametros olhando o teste.
DENSITIES = {"flowers_micro": {1: 0.125, 5: 0.05}, "cars_micro": {1: 0.075, 5: 0.025}}
DEGREES = {"dgcg-4-6": (4, 6), "dgcg-2-4": (2, 4)}
METHODS = ("no-graph", "knn-union", "dgcg", *DEGREES)
GROUP = ("dataset", "backbone", "mode", "n_shot", "metodo")
TIMING_GROUP = ("backbone", "mode", "n_shot", "metodo")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--backbones", nargs="+", choices=["resnet18", "vit-b-16"], default=["resnet18", "vit-b-16"])
    p.add_argument("--modes", nargs="+", choices=["trained", "train-free"], default=["trained", "train-free"])
    p.add_argument("--shots", nargs="+", type=int, choices=[1, 5], default=[1, 5])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                   help="dgcg transfere a densidade historica; dgcg-4-6 e dgcg-2-4 usam grau medio fixo.")
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cuda")
    p.add_argument("--weights", choices=["imagenet", "random"], default="imagenet",
                   help="random somente para verificacao offline, sem interpretacao cientifica.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu-threads", type=int, default=1,
                   help="Threads CPU por processo; matrizes episodicas sao pequenas.")
    p.add_argument("--output-root", type=Path, default=Path("runs/comparacao_backbones"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/openml"))
    p.add_argument("--max-runs", type=int, default=None,
                   help="Executa no maximo N condicoes pendentes; mesma pasta pode ser retomada.")
    p.add_argument("--quick", action="store_true", help="Verificacao reduzida; usar pasta separada da bateria completa.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p


def jobs(args):
    # O plano inclui os historicos; main identifica quais nao precisam executar.
    for seed, dataset, shot, backbone, mode, method in itertools.product(
        args.seeds, args.datasets, args.shots, args.backbones, args.modes, args.methods,
    ):
        size = 224 if backbone == "vit-b-16" else 128
        folder = (args.output_root / dataset / backbone / mode
                  / f"5way_{shot}shot_10query" / f"seed{seed}" / method)
        target_kind = "degree" if method in DEGREES else "density" if method == "dgcg" else "none"
        target = (":".join(map(str, DEGREES[method])) if method in DEGREES
                  else str(DENSITIES[dataset][shot]) if method == "dgcg" else "-")
        yield dict(dataset=dataset, openml_id=DATASETS[dataset], backbone=backbone,
                   image_size=size, mode=mode, n_way=5, n_shot=shot, n_query=10,
                   seed=seed, metodo=method, target_kind=target_kind, alvo=target,
                   directory=str(folder))


def command(job, args):
    # Um cache por base/extrator, compartilhado por modos, metodos, shots e seeds.
    work = args.output_root / "_work" / job["dataset"] / job["backbone"]
    cmd = [sys.executable, str(ROOT / "protosgc_fewshot.py"),
           "--openml-id", str(job["openml_id"]), "--cache-dir", str(args.cache_dir),
           "--work-dir", str(work), "--backbone", job["backbone"], "--weights", args.weights,
           "--image-size", str(job["image_size"]), "--batch-size", str(args.batch_size),
           "--class-split", "10", "5", "5", "--n-way", "5", "--n-shot", str(job["n_shot"]),
           "--n-query", "10", "--seed", str(job["seed"]), "--device", args.device,
           "--feature-dim", "256", "--knn", "5", "--knn-metric", "euclidean",
           "--sgc-hops", "2", "--no-grande",
           "--train-episodes", "2000", "--val-every", "200",
           "--val-episodes", "200", "--test-episodes", "1000"]
    cache = args.cache_dir / "org/openml/www/datasets" / str(job["openml_id"])
    labels = list(cache.glob("labels.csv")) + list(cache.glob("*/labels.csv"))
    if len(labels) == 1:
        cmd += ["--data-root", str(labels[0].parent)]
    cmd += ["--model", "protonet" if job["metodo"] == "no-graph" else "proto-sgc",
            "--graph-type", "dgcg" if job["metodo"] in DEGREES else
            "knn-union" if job["metodo"] == "no-graph" else job["metodo"]]
    if job["metodo"] == "dgcg":
        cmd += ["--dgcg-target-density", job["alvo"], job["alvo"]]
    elif job["metodo"] in DEGREES:
        cmd += ["--dgcg-target-degree", *job["alvo"].split(":")]
    if job["mode"] == "train-free":
        cmd += ["--train-free"]
    if args.quick:
        cmd += ["--quick"]
    return cmd


def matches(job, result, args):
    a = result.get("args", {})
    train_free = job["mode"] == "train-free"
    dim = 768 if job["backbone"] == "vit-b-16" else 512
    expected = {k: job[k] for k in ("openml_id", "backbone", "image_size", "n_way", "n_shot", "n_query", "seed")}
    expected.update(weights=args.weights, class_split=[10, 5, 5], train_free=train_free,
                    feature_dim=dim if train_free else 256, batch_size=args.batch_size,
                    model="protonet" if job["metodo"] == "no-graph" else "proto-sgc",
                    graph_type="dgcg" if job["metodo"] in DEGREES else
                    "knn-union" if job["metodo"] == "no-graph" else job["metodo"],
                    knn=5, knn_metric="euclidean", sgc_hops=2, grande=False,
                    cosine_rbf_weight=False, extra_images=None, dgcg_extra_references=None,
                    dgcg_train_global_rankings=False, dgcg_reference_train_rankings=False,
                    train_episodes=200 if args.quick else 2000, val_every=50 if args.quick else 200,
                    val_episodes=50 if args.quick else 200, test_episodes=200 if args.quick else 1000,
                    quick=args.quick)
    if job["target_kind"] in ("density", "degree"):
        expected.update(dgcg_target_density=[float(job["alvo"])] * 2 if job["target_kind"] == "density" else None,
                        dgcg_metric="euclidean", dgcg_correlation="rbo", dgcg_threshold=None)
        if job["target_kind"] == "degree":
            expected["dgcg_target_degree"] = [float(v) for v in job["alvo"].split(":")]
    return all(a.get(k) == v for k, v in expected.items())


def summarize(root, plan, args):
    rows, statuses = [], []
    for job in plan:
        folder = Path(job["directory"])
        path = folder / "metrics.json"
        status = json.loads((folder / "status.json").read_text()) if (folder / "status.json").exists() else {"status": "pending"}
        if job.get("reuse"):
            rows.append(historical_row(job, job["reuse"]))
            status = {"status": "reused", "source": job["reuse"]["source"],
                      "source_line": job["reuse"]["source_line"]}
        elif path.exists():
            result = json.loads(path.read_text())
            if not matches(job, result, args):
                raise ValueError(f"Resultado incompativel: {path}")
            row = flatten(job, result)
            row["elapsed_seconds"] = result["execution"]["elapsed_seconds"]
            row.update(origin="new", source_device=result["args"].get("device", args.device),
                       source=str(path), source_line="", workbook_id="", workbook_row="")
            rows.append(row)
            status["status"] = "complete"
        statuses.append(dict({k: v for k, v in job.items() if k != "reuse"}, **status))
    controls = {tuple(r[k] for k in ("dataset", "backbone", "mode", "n_shot", "seed")): r
                for r in rows if r["metodo"] == "no-graph"}
    for row in rows:
        baseline = controls.get(tuple(row[k] for k in ("dataset", "backbone", "mode", "n_shot", "seed")))
        row["ganho_vs_sem_grafo_pp"] = row["test_accuracy"] - baseline["test_accuracy"] if baseline else ""
    means = []
    for key in sorted({tuple(j[k] for k in GROUP) for j in plan}):
        group = [r for r in rows if tuple(r[k] for k in GROUP) == key]
        row = dict(zip(GROUP, key), concluidos=len(group),
                   planejados=sum(tuple(j[k] for k in GROUP) == key for j in plan),
                   seeds=";".join(str(r["seed"]) for r in group))
        for metric in ("val_accuracy", "test_accuracy", "test_macro_f1", "test_balanced_accuracy", "ganho_vs_sem_grafo_pp"):
            values = [r[metric] for r in group if r[metric] != ""]
            row[metric + "_mean"] = statistics.mean(values) if values else ""
            row[metric + "_std_seed"] = statistics.stdev(values) if len(values) > 1 else ""
        means.append(row)
    for name, values in (("resultados", rows), ("status", statuses), ("medias", means)):
        save_csv(root / (name + ".csv"), values)
    timing = {}
    for row in rows:
        if row["elapsed_seconds"] != "":
            timing.setdefault(tuple(row[k] for k in TIMING_GROUP), []).append(row["elapsed_seconds"])
    pending = [s for s in statuses if s["status"] not in ("complete", "reused")]
    covered = [j for j in pending if tuple(j[k] for k in TIMING_GROUP) in timing]
    estimate = sum(statistics.median(timing[tuple(j[k] for k in TIMING_GROUP)]) for j in covered)
    timing_result = dict(completed=len(rows), planned=len(plan), pending=len(pending),
                         pending_with_timing=len(covered), pending_without_timing=len(pending)-len(covered),
                         estimated_seconds_for_covered_pending=estimate,
                         reused=sum(s["status"] == "reused" for s in statuses),
                         measured_seconds=sum(r["elapsed_seconds"] for r in rows if r["elapsed_seconds"] != ""),
                         note="Medianas por backbone/modo/shot/metodo; inclui inicializacao e eventual extracao. Nao e garantia.")
    atomic_json(root / "tempo_estimado.json", timing_result)
    print(f"Concluidos: {len(rows)}/{len(plan)}", flush=True)
    if pending and covered:
        print(f"Tempo estimado para {len(covered)}/{len(pending)} pendentes com referencia medida: {estimate / 3600:.2f} h", flush=True)
    elif pending:
        print("Tempo: ainda faltam medicoes das condicoes pendentes.", flush=True)


def hardware_info(device):
    import torch
    import torchvision
    info = dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                cuda_runtime=torch.version.cuda, device=str(device), gpu=None)
    if device.type == "cuda":
        # Confere que o build executa kernels nesta GPU, nao apenas que a detecta.
        value = torch.ones((8, 8), device=device)
        (value @ value).sum().item()
        torch.cuda.synchronize(device)
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def run_job(job, args):
    folder = Path(job["directory"])
    folder.mkdir(parents=True, exist_ok=True)
    cmd = command(job, args)
    atomic_json(folder / "command.json", cmd)
    atomic_json(folder / "status.json", {"status": "running"})
    work = args.output_root / "_work" / job["dataset"] / job["backbone"]
    before = {p: p.stat().st_mtime_ns for p in work.rglob("*.metrics.json")}
    started = time.perf_counter()
    try:
        with (folder / "run.log").open("w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, PYTHONUNBUFFERED="1",
                                           OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads)))
        elapsed = time.perf_counter() - started
        fresh = [p for p in work.rglob("*.metrics.json") if before.get(p) != p.stat().st_mtime_ns]
        result = json.loads(fresh[0].read_text()) if proc.returncode == 0 and len(fresh) == 1 else None
        if result is None or not matches(job, result, args):
            raise RuntimeError(f"Execucao falhou ou protocolo incompativel; veja {folder / 'run.log'}")
        shutil.copy2(result["checkpoint"], folder / "checkpoint.pt")
        result["checkpoint"] = str(folder / "checkpoint.pt")
        result["execution"] = {"elapsed_seconds": elapsed, "cpu_threads": args.cpu_threads}
        atomic_json(folder / "metrics.json", result)
        atomic_json(folder / "status.json", {"status": "complete", "elapsed_seconds": elapsed})
    except KeyboardInterrupt:
        atomic_json(folder / "status.json", {"status": "interrupted"})
        raise
    except Exception as exc:
        atomic_json(folder / "status.json", {"status": "failed", "error": str(exc)})
        raise


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for key in ("datasets", "backbones", "modes", "shots", "methods", "seeds"):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            p.error(f"Valores repetidos: {key}")
    if min(args.seeds) < 0 or min(args.batch_size, args.cpu_threads) < 1 or (args.max_runs is not None and args.max_runs < 1):
        p.error("Seeds >= 0; batch-size, cpu-threads e max-runs positivos.")
    args.output_root = args.output_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    # Falha antes de executar se a tabela e suas fontes nao puderem ser conferidas.
    audit = audit_results(ROOT)
    manifest_path = args.output_root / "manifest.json"
    if args.summarize_only:
        if not manifest_path.exists():
            p.error("Manifesto nao encontrado.")
        manifest = json.loads(manifest_path.read_text())
        if manifest["configuration"].get("result_inventory") != {
            "workbook_sha256": audit["workbook_sha256"], "sources_sha256": audit["sources_sha256"],
        }:
            p.error("Planilha/fontes mudaram desde o manifesto; confira o reaproveitamento antes de consolidar.")
        saved = argparse.Namespace(**manifest["configuration"]["arguments"])
        summarize(args.output_root, manifest["runs"], saved)
        return 0
    plan = list(jobs(args))
    for job in plan:
        reused = reusable_result(job, args, audit)
        if reused:
            job["reuse"] = reused
    reused_count = sum(bool(j.get("reuse")) for j in plan)
    print(f"{len(plan)} condicoes: {reused_count} reaproveitadas, {len(plan)-reused_count} novas; "
          f"5-way/10-query; Micro; sem extras; pesos={args.weights}", flush=True)
    if args.dry_run:
        for job in plan:
            if job.get("reuse"):
                print(f"# REAPROVEITAR {job['directory']}: {job['reuse']['source']}:{job['reuse']['source_line']}")
            else:
                print(shlex.join(command(job, args)))
        return 0
    configuration = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                      if k not in ("dry_run", "summarize_only", "max_runs")},
        "code_sha256": source_digest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dgcg_selection": "historical trained ResNet densities transferred without new search; fixed degree4-6 and degree2-4 compared separately",
        "densities": DENSITIES,
        "degrees": DEGREES,
        "result_inventory": {"workbook_sha256": audit["workbook_sha256"], "sources_sha256": audit["sources_sha256"]},
    }
    # JSON normaliza chaves numericas das densidades antes da comparacao.
    configuration = json.loads(json.dumps(configuration))
    if manifest_path.exists() and json.loads(manifest_path.read_text())["configuration"] != configuration:
        p.error("Codigo/configuracao diferente nesta pasta; use outro --output-root.")
    if reused_count == len(plan):
        hardware = {"device": "historical_only", "gpu": None}
    else:
        from proto_sgc.runtime import select_device
        hardware = hardware_info(select_device(args.device))
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        atomic_json(manifest_path, {"configuration": configuration, "hardware": hardware, "runs": plan})
    elif json.loads(manifest_path.read_text())["hardware"] != hardware:
        p.error("Ambiente/GPU diferente do manifesto; use outro --output-root.")
    print("Ambiente:", hardware, flush=True)
    atomic_json(args.output_root / "auditoria_reaproveitamento.json",
                dict(audit, planned=len(plan), reused=reused_count, new=len(plan)-reused_count))
    failure = 0
    executed = 0
    try:
        for index, job in enumerate(plan, 1):
            if job.get("reuse"):
                continue
            path = Path(job["directory"]) / "metrics.json"
            if path.exists():
                if not matches(job, json.loads(path.read_text()), args):
                    raise ValueError(f"Resultado incompativel: {path}")
                continue
            if args.max_runs is not None and executed >= args.max_runs:
                break
            print(f"[{index}/{len(plan)}] {Path(job['directory']).relative_to(args.output_root)}", flush=True)
            run_job(job, args)
            executed += 1
            summarize(args.output_root, plan, args)
    except KeyboardInterrupt:
        failure = 130
    except Exception as exc:
        print(str(exc), flush=True)
        failure = 1
    finally:
        summarize(args.output_root, plan, args)
    return failure


if __name__ == "__main__":
    raise SystemExit(main())
