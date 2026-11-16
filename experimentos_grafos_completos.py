"""Dez bases Micro/Mini: grafos completos RBO/RBF e extras de toda a particao."""

import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

from experimentos_extras import atomic_json
from experimentos_metricas_graus import DATASETS, ROOT, flatten, save_csv
from proto_sgc.result_inventory import audit_results, sha256


PROTOCOLS = {"2w5s5q": (2, 5, 5), "2w10s10q": (2, 10, 10),
             "5w5s5q": (5, 5, 5), "5w10s10q": (5, 10, 10)}
FAMILIES = [name.removesuffix("_micro") for name in DATASETS]
GRAPHS = ("complete-rbo", "complete-rbf")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=FAMILIES, default=FAMILIES)
    p.add_argument("--versions", nargs="+", choices=("micro", "mini"), default=["micro", "mini"])
    p.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    p.add_argument("--graphs", nargs="+", choices=GRAPHS, default=list(GRAPHS))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--extras-step", type=int, choices=(5, 10), default=10)
    p.add_argument("--rbf-sigma-mode", choices=("episode-median", "fixed"), default="episode-median")
    p.add_argument("--rbf-sigma", type=float, default=32.)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--cache-dir", type=Path, default=Path("data/openml"))
    p.add_argument("--output-root", type=Path, default=Path("runs/grafos_completos_extras_mediana_seed7"))
    p.add_argument("--feature-root", type=Path, default=None,
                   help="Cache de vetores por base, compartilhado entre todas as condicoes.")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--train-episodes", type=int, default=2000)
    p.add_argument("--val-episodes", type=int, default=200)
    p.add_argument("--test-episodes", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--limit", type=int, default=None, help="Maximo de condicoes novas nesta chamada.")
    p.add_argument("--dry-run", action="store_true", help="Audita e mostra o plano, sem gravar nem baixar dados.")
    p.add_argument("--summarize-only", action="store_true", help="Reconstroi CSV e graficos dos resultados salvos.")
    return p


def jobs(args):
    for family, version in itertools.product(args.datasets, args.versions):
        openml_id = DATASETS[family + "_micro"] + (44 if version == "mini" else 0)
        maximum = 100 if version == "micro" else 200
        conditions = [("no-graph", 0)] + list(itertools.product(
            args.graphs, range(0, maximum + 1, args.extras_step)))
        for protocol, (graph, extras) in itertools.product(args.protocols, conditions):
            way, shot, query = PROTOCOLS[protocol]
            folder = (args.output_root / f"{family}_{version}" / protocol
                      / f"seed{args.seed}" / graph / f"extras{extras}")
            yield dict(dataset=f"{family}_{version}", family=family, version=version,
                       openml_id=openml_id, protocol=protocol, n_way=way, n_shot=shot,
                       n_query=query, seed=args.seed, metodo=graph, extras_total=extras,
                       rbf_sigma_mode=args.rbf_sigma_mode if graph == "complete-rbf" else None,
                       rbf_sigma=args.rbf_sigma if graph == "complete-rbf" and args.rbf_sigma_mode == "fixed" else None,
                       directory=str(folder))


def code_digest():
    digest = hashlib.sha256()
    paths = [Path(__file__), ROOT / "experimentos_metricas_graus.py", ROOT / "experimentos_extras.py"]
    paths.extend(sorted((ROOT / "proto_sgc").rglob("*.py")))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def arguments(args, job, split_sizes):
    from proto_sgc.config import parse_args, validate_hyperparameters

    no_graph = job["metodo"] == "no-graph"
    argv = ["--openml-id", str(job["openml_id"]), "--backbone", "resnet18",
            "--weights", "imagenet", "--image-size", "128", "--feature-dim", "256",
            "--model", "protonet" if no_graph else "proto-sgc",
            "--graph-type", "knn-union" if no_graph else job["metodo"], "--sgc-hops", "2",
            "--class-split", *map(str, split_sizes), "--n-way", str(job["n_way"]),
            "--n-shot", str(job["n_shot"]), "--n-query", str(job["n_query"]),
            "--seed", str(job["seed"]), "--device", args.device,
            "--work-dir", job["directory"], "--cache-dir", str(args.cache_dir),
            "--extra-images", str(job["extras_total"]), "--extra-source", "split-pool",
            "--extra-mode", "propagation", "--extra-phases", "train", "val", "test",
            "--graph-temperature", str(args.rbf_sigma), "--no-grande",
            "--rbf-sigma-mode", args.rbf_sigma_mode,
            "--dgcg-metric", "euclidean", "--dgcg-correlation", "rbo",
            "--dgcg-rbo-p", "0.9", "--dgcg-top-k", "40", "--dgcg-list-size", "200",
            "--train-episodes", str(args.train_episodes), "--val-episodes", str(args.val_episodes),
            "--test-episodes", str(args.test_episodes), "--val-every", str(args.val_every)]
    result = parse_args(argv)
    validate_hyperparameters(result)
    return result


def load_features(args, job, device):
    from proto_sgc.data import download_meta_album, load_local_meta_album
    from proto_sgc.features import extract_or_load_features, feature_spec, records_fingerprint
    from proto_sgc.runtime import _safe_torch_load
    import shutil

    cache = args.cache_dir / "org/openml/www/datasets" / str(job["openml_id"])
    labels = list(cache.glob("labels.csv")) + list(cache.glob("*/labels.csv"))
    records = (load_local_meta_album(labels[0].parent) if len(labels) == 1
               else download_meta_album(job["openml_id"], args.cache_dir))
    class_count = len({r.label for r in records})
    split_sizes = [class_count // 2, class_count // 4,
                   class_count - class_count // 2 - class_count // 4]
    config = arguments(args, job, split_sizes)
    config.work_dir = args.feature_root / job["dataset"]
    target = config.work_dir / "features_resnet18_imagenet_128px.pt"
    fingerprint = records_fingerprint(records)
    spec = feature_spec(config)
    if not target.exists():
        # Somente copia caches cujo conjunto de imagens e extrator conferem.
        for candidate in sorted((ROOT / "runs").rglob(target.name)):
            if candidate == target:
                continue
            cached = _safe_torch_load(candidate)
            if (isinstance(cached, dict) and cached.get("fingerprint") == fingerprint
                    and cached.get("feature_spec", spec) == spec and cached.get("input_dim") == 512):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, target)
                print(f"Cache compativel reaproveitado: {candidate}", flush=True)
                break
    features, dim = extract_or_load_features(records, config, device)
    if dim != 512:
        raise ValueError(f"Dimensao inesperada para ResNet-18: {dim}")
    provenance = dict(records_fingerprint=fingerprint, feature_cache_sha256=sha256(target),
                      images=len(records), classes=class_count)
    return features, split_sizes, spec, provenance


def resume_key(args, job, digest):
    return dict(job=job, code_sha256=digest, device=args.device,
                train_episodes=args.train_episodes, val_episodes=args.val_episodes,
                test_episodes=args.test_episodes, val_every=args.val_every,
                feature_root=str(args.feature_root), cache_dir=str(args.cache_dir))


def read_result(args, job, digest):
    path = Path(job["directory"]) / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result["battery"]["resume_key"] != resume_key(args, job, digest):
        raise ValueError(f"Resultado incompativel com codigo/configuracao atuais: {path}. Use outra pasta.")
    return result


def run_job(args, job, features, split_sizes, spec, provenance, device, digest):
    import torch
    from proto_sgc.episodes import FeatureEpisodeSampler, split_classes
    from proto_sgc.model import ProtoSGC
    from proto_sgc.runtime import seed_everything
    from proto_sgc.training import train_model

    folder = Path(job["directory"])
    folder.mkdir(parents=True, exist_ok=True)
    config = arguments(args, job, split_sizes)
    config.feature_spec = spec
    config.feature_cache_dir = str(args.feature_root / job["dataset"])
    config.feature_provenance = provenance
    sampler = FeatureEpisodeSampler(features, job["n_way"], job["n_shot"], job["n_query"])
    splits = split_classes(sorted(features), job["n_way"], split_sizes, args.seed)
    seed_everything(args.seed)
    model = ProtoSGC(512, 256, config.model, 5, config.graph_type, 2, args.rbf_sigma,
                     rbf_sigma_mode=config.rbf_sigma_mode)
    atomic_json(folder / "status.json", {"status": "running", "job": job})
    started = time.perf_counter()
    try:
        with (folder / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            _, _, checkpoint = train_model(model, sampler, splits, config, device)
        result = json.loads(checkpoint.with_suffix(".metrics.json").read_text())
        result["battery"] = dict(resume_key=resume_key(args, job, digest),
                                 elapsed_seconds=time.perf_counter() - started,
                                 checkpoint_sha256=sha256(checkpoint),
                                 hardware=dict(torch=torch.__version__, python=platform.python_version(),
                                               cuda_runtime=torch.version.cuda,
                                               gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
        atomic_json(folder / "result.json", result)
        atomic_json(folder / "status.json", {"status": "complete"})
        return result
    except Exception as error:
        atomic_json(folder / "status.json", {"status": "failed", "error": str(error)})
        raise


def summarize(args, plan, digest):
    records = []
    pairing = defaultdict(dict)
    for job in plan:
        result = read_result(args, job, digest)
        if result is None:
            continue
        row = flatten(job, result)
        row["elapsed_seconds"] = result["battery"]["elapsed_seconds"]
        row["source_device"] = result["args"]["device"]
        row["result_source"] = str(Path(job["directory"]) / "result.json")
        row["code_sha256"] = digest
        row["class_split"] = json.dumps(result["args"]["class_split"])
        scope = job["dataset"], job["protocol"], job["seed"]
        for phase, bank in result["extra_context"]["banks"].items():
            key = phase, bank["episodes_sampled"]
            expected = pairing[scope].setdefault(key, bank["episode_ids_sha256"])
            if bank["episode_ids_sha256"] != expected:
                raise ValueError(f"Episodios nao pareados entre condicoes: {scope}, {phase}")
            row[f"{phase}_episode_ids_sha256"] = bank["episode_ids_sha256"]
        records.append(row)
    baselines = {(r["dataset"], r["protocol"], r["seed"]): r["test_accuracy"]
                 for r in records if r["metodo"] == "no-graph"}
    for row in records:
        baseline = baselines.get((row["dataset"], row["protocol"], row["seed"]))
        row["gain_vs_no_graph_pp"] = row["test_accuracy"] - baseline if baseline is not None else ""
    atomic_json(args.output_root / "progress.json", dict(complete=len(records), planned=len(plan)))
    save_csv(args.output_root / "resultados.csv", records)
    return records


def plot(args, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    folder = args.output_root / "graficos"
    folder.mkdir(exist_ok=True)
    for version, protocol, graph in itertools.product(args.versions, args.protocols, args.graphs):
        fig, ax = plt.subplots(figsize=(10, 6))
        has_baseline = False
        for index, family in enumerate(args.datasets):
            color = plt.get_cmap("tab10")(index % 10)
            rows = sorted([r for r in records if (r["family"], r["version"], r["protocol"], r["metodo"])
                           == (family, version, protocol, graph)], key=lambda r: r["extras_total"])
            if rows:
                ax.plot([r["extras_total"] for r in rows], [r["test_accuracy"] for r in rows],
                        marker="o", markersize=3, linewidth=1.2, color=color,
                        label=family.replace("_", " "))
            baseline = [r for r in records if (r["family"], r["version"], r["protocol"], r["metodo"])
                        == (family, version, protocol, "no-graph")]
            if baseline:
                has_baseline = True
                ax.axhline(baseline[0]["test_accuracy"], color=color, linestyle="--", linewidth=.9, alpha=.65,
                           label="_nolegend_" if rows else family.replace("_", " ") + " (sem grafo)")
        way, shot, query = PROTOCOLS[protocol]
        sigma_label = "mediana por episodio" if args.rbf_sigma_mode == "episode-median" else f"{args.rbf_sigma:g}"
        graph_label = "RBO (p=0,9)" if graph == "complete-rbo" else f"RBF euclidiano (sigma={sigma_label})"
        ax.set_title(f"{version.title()} | {way}-way, {shot}-shot, {query}-query | {graph_label}")
        ax.set_xlabel("Imagens extras por episodio (treino, validacao e teste)")
        ax.set_ylabel("Acuracia de teste (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 100 if version == "micro" else 200)
        ax.grid(alpha=.25)
        if ax.lines:
            handles, labels = ax.get_legend_handles_labels()
            if has_baseline:
                handles.append(Line2D([], [], color="gray", linestyle="--"))
                labels.append("Sem grafo (mesma cor por base)")
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
        fig.text(.08, .018, "ResNet-18 congelada | Projecao 512 para 256 treinada | Seed " + str(args.seed), fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(folder / f"{version}_{protocol}_{graph}.png", dpi=160)
        plt.close(fig)


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < args.rbf_sigma < float("inf"):
        raise ValueError("Sigma deve ser finito e positivo.")
    if min(args.train_episodes, args.val_episodes, args.test_episodes, args.val_every, args.cpu_threads) < 1:
        raise ValueError("Episodios, intervalo e threads devem ser positivos.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit deve ser positivo.")
    for values in (args.datasets, args.versions, args.protocols, args.graphs):
        if len(set(values)) != len(values):
            raise ValueError("Selecoes nao podem conter duplicatas.")
    args.output_root = args.output_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.feature_root = (args.feature_root or args.output_root / "_features").expanduser().resolve()
    audit = audit_results(ROOT)
    plan = list(jobs(args))
    digest = code_digest()
    print(f"{len(plan)} condicoes; {len(args.versions)*len(args.protocols)*len(args.graphs)} graficos; seed {args.seed}.", flush=True)
    print("Auditoria das fontes historicas concluida. Grafos completos com extras split-pool requerem resultados proprios.", flush=True)
    if args.dry_run:
        for version in args.versions:
            subset = [j for j in plan if j["version"] == version]
            print(f"{version}: {len(subset)} condicoes, extras de 0 a {max(j['extras_total'] for j in subset)}, passo {args.extras_step}.")
        return 0
    # Falha antes de treinar caso a dependencia dos graficos esteja ausente.
    import matplotlib
    import torch
    from proto_sgc.runtime import select_device

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = args.output_root / "manifest.json"
        manifest = dict(schema=1, runs=plan, code_sha256=digest,
                        configuration={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
                                       if k not in ("limit", "dry_run", "summarize_only")},
                        audit=audit,
                        dataset_sources=[f"https://meta-album.github.io/datasets/{code}.html"
                                         for code in ("BCT","PLK","FLW","SPT","BRD","PLT_VIL","MD_MIX","TEX","CRS","RESISC")])
        if config_path.exists():
            saved = json.loads(config_path.read_text())
            if saved != manifest:
                raise ValueError("Manifesto diferente. Retome com a mesma configuracao ou use outra pasta.")
        else:
            atomic_json(config_path, manifest)
        if args.summarize_only:
            records = summarize(args, plan, digest)
            plot(args, records)
            return 0
        torch.set_num_threads(args.cpu_threads)
        device = select_device(args.device)
        current_dataset, loaded = None, None
        executed = 0
        try:
            for index, job in enumerate(plan, 1):
                if read_result(args, job, digest) is not None:
                    continue
                if args.limit is not None and executed >= args.limit:
                    break
                print(f"[{index}/{len(plan)}] {job['dataset']} {job['protocol']} {job['metodo']} extras={job['extras_total']}", flush=True)
                if current_dataset != job["dataset"]:
                    loaded = load_features(args, job, device)
                    current_dataset = job["dataset"]
                features, split_sizes, spec, provenance = loaded
                run_job(args, job, features, split_sizes, spec, provenance, device, digest)
                executed += 1
                records = summarize(args, plan, digest)
                print(f"Concluidas {len(records)}/{len(plan)}.", flush=True)
        finally:
            records = summarize(args, plan, digest)
            if records:
                plot(args, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
