"""Repete o diagnostico do PDF com ResNet congelada e classificador treinado."""

import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

import torch

from experimentos_extras import atomic_json, source_digest
from experimentos_metricas_graus import ROOT, DATASETS, save_csv
from proto_sgc.config import parse_args, validate_hyperparameters
from proto_sgc.data import load_local_meta_album, download_meta_album
from proto_sgc.episodes import FeatureEpisodeSampler, split_classes
from proto_sgc.features import extract_or_load_features
from proto_sgc.homophily import measure_homophily, pearson
from proto_sgc.model import ProtoSGC
from proto_sgc.result_inventory import audit_results, CLAUDE, INVENTORY, sha256
from proto_sgc.runtime import seed_everything, select_device
from proto_sgc.training import train_model, evaluate
from proto_sgc.training_curve import paired_curves, train_with_curve

METHODS = ("no-graph", "knn-union", "dgcg-4-6")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--shots", nargs="+", type=int, choices=[1, 5], default=[1, 5])
    p.add_argument("--seeds", nargs="+", type=int, default=[7])
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--cache-dir", type=Path, default=Path("data/openml"))
    p.add_argument("--output-root", type=Path, default=Path("runs/homofilia_treinada_resnet"))
    p.add_argument("--checkpoint-roots", nargs="+", type=Path,
                   default=[Path("resultados/execucoes"), Path("resultados/testes_claude_20261102/scripts/work")])
    p.add_argument("--train-missing", action="store_true",
                   help="Autoriza treinar somente os checkpoints compativeis que faltarem.")
    p.add_argument("--evaluate-models", action="store_true",
                   help="Reavalia modelos/checkpoints. Padrao: cruza resultados treinados ja existentes com a homofilia das mesmas arestas fixas.")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--quick", action="store_true", help="Apenas funcionamento: treino200/val50/test200/hom20, pasta separada.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--training-curve", action="store_true",
                   help="Treina do início e registra homofilia e acurácia ao longo dos episódios.")
    return p


def core_args(args, dataset, shot, seed, method, work):
    argv = ["homophily", "--openml-id", str(DATASETS[dataset]), "--backbone", "resnet18",
            "--weights", "imagenet", "--image-size", "128", "--feature-dim", "256",
            "--class-split", "10", "5", "5", "--n-way", "5", "--n-shot", str(shot),
            "--n-query", "10", "--seed", str(seed), "--device", args.device,
            "--model", "protonet" if method == "no-graph" else "proto-sgc",
            "--graph-type", "dgcg" if method == "dgcg-4-6" else "knn-union",
            "--dgcg-target-degree", "4", "6", "--work-dir", str(work),
            "--cache-dir", str(args.cache_dir)]
    if args.quick:
        argv += ["--quick"]
    a = parse_args(argv[1:])
    validate_hyperparameters(a)
    return a


def build_model(a):
    return ProtoSGC(512, 256, a.model, a.knn, a.graph_type, a.sgc_hops, a.graph_temperature,
                    dgcg_target_degree=a.dgcg_target_degree)


def compatible(saved, expected, dataset, location):
    """Rejeita extras, ViT, identidade, ponderacoes e outros protocolos."""
    defaults = {"backbone": "resnet18", "train_free": False, "extra_images": None,
                "dgcg_extra_references": None, "dgcg_train_global_rankings": False,
                "dgcg_reference_train_rankings": False, "cosine_rbf_weight": False,
                "grande": False, "dgcg_target_density": None, "dgcg_threshold": None,
                "knn_metric": "euclidean"}
    fields = ("weights", "image_size", "feature_dim", "n_way", "n_shot", "n_query", "seed",
              "model", "graph_type", "sgc_hops", "knn", "train_episodes", "val_episodes",
              "test_episodes", "val_every", "learning_rate", "weight_decay", "grad_clip",
              "dgcg_metric", "dgcg_correlation", "dgcg_list_size", "dgcg_candidate_k",
              "dgcg_top_k", "dgcg_rbo_p", *defaults)
    if any(saved.get(k, defaults.get(k)) != getattr(expected, k) for k in fields):
        return False
    if saved.get("device") != expected.device:
        return False  # Nao mistura treino CPU com a bateria CUDA, ou vice-versa.
    if saved.get("class_split") not in (None, [10, 5, 5], (10, 5, 5)):
        return False
    if expected.graph_type == "dgcg" and tuple(saved.get("dgcg_target_degree", ())) != (4., 6.):
        return False
    text = str(location)
    if any(token in text for token in ("cl2n", "pt-cl2n", "idinit", "topo:", "alpha", "adapt")):
        return False
    markers = [name for name in DATASETS if name in text or name in str(saved.get("work_dir", ""))]
    if markers:
        return set(markers) == {dataset}
    data_root = str(saved.get("data_root") or "")
    if data_root:
        return str(DATASETS[dataset]) in data_root
    return saved.get("openml_id") == DATASETS[dataset]


def checkpoint_index(roots):
    found = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(set(root.rglob("*.metrics.json")) | set(root.rglob("metrics.json"))):
            try:
                result = json.loads(path.read_text())
                saved = result.get("args", {})
                candidates = [path.parent / "checkpoint.pt", path.with_name(path.name.replace(".metrics.json", ".pt"))]
                if result.get("checkpoint"):
                    candidates.insert(0, Path(result["checkpoint"]))
                checkpoint = next((c for c in candidates if c.is_file() and c.suffix == ".pt"), None)
                if checkpoint:
                    found.append((path, checkpoint, saved))
            except (ValueError, OSError):
                continue
    return found


def plan(args, index):
    jobs = []
    for dataset, shot, seed, method in itertools.product(args.datasets, args.shots, args.seeds, METHODS):
        folder = args.output_root / dataset / f"5way_{shot}shot_10query" / f"seed{seed}" / method
        a = core_args(args, dataset, shot, seed, method, folder / "model")
        # Prefere um checkpoint da propria pasta antes de reutilizar o catalogo.
        candidates = sorted(index, key=lambda c: (not c[1].is_relative_to(args.output_root), str(c[0])))
        match = next((c for c in candidates if compatible(c[2], a, dataset, c[0])), None)
        jobs.append(dict(dataset=dataset, shot=shot, seed=seed, method=method, folder=str(folder),
                         checkpoint=str(match[1]) if match else None))
    return jobs


def pdf_reference():
    """Recalcula a referencia sem treino, com h_query, a partir das fontes."""
    topo = ROOT / "resultados/testes_claude_20261102/resultados/res_topo_tf.jsonl"
    rows = [json.loads(s) for s in topo.read_text().splitlines()]
    import csv
    with (ROOT / INVENTORY).open() as stream:
        expected = next(r["sha256"] for r in csv.DictReader(stream)
                        if r["arquivo"] == str(topo.relative_to(ROOT)))
    if sha256(topo) != expected:
        raise ValueError("Fonte de homofilia mudou desde a auditoria.")
    controls = [json.loads(s) for s in (ROOT / CLAUDE).read_text().splitlines()]
    output = {}
    for shot in (1, 5):
        group = [r for r in rows if r["shot"] == shot and r["tag"] == "" and r["method"] in ("union", "dgcg")]
        gain = [r["test"] - next(c["test"] for c in controls if c["dataset"] == r["dataset"]
                and c["shot"] == shot and c["method"] == "no-graph" and c["trainfree"] and c["feat"] == "raw") for r in group]
        output[str(shot)] = {"seed": 7, "n": len(group),
                            "pearson_query": pearson([r["hom_query"] for r in group], gain)}
    return output


def summarize(args, jobs, reference):
    rows, statuses = [], []
    for job in jobs:
        path = Path(job["folder"]) / "diagnostic.json"
        statuses.append(dict(job, status="complete" if path.exists() else "pending"))
        if path.exists():
            rows.append(json.loads(path.read_text()))
    rows = [{k: v for k, v in row.items() if k != "curve"} for row in rows]
    controls = {(r["dataset"], r["shot"], r["seed"]): r for r in rows if r["method"] == "no-graph"}
    points = []
    for r in rows:
        if r["method"] == "no-graph":
            continue
        baseline = controls.get((r["dataset"], r["shot"], r["seed"]))
        if baseline:
            points.append(dict(r, baseline_test_percent=baseline["test_percent"],
                               gain_pp=r["test_percent"] - baseline["test_percent"]))
    correlations = []
    for seed, shot in itertools.product(args.seeds, args.shots):
        group = [r for r in points if r["seed"] == seed and r["shot"] == shot]
        correlations.append(dict(seed=seed, shot=shot, points=len(group), expected_points=2*len(args.datasets),
                                 complete=len(group)==2*len(args.datasets),
                                 pearson_query=pearson([r["homophily_query_percent"] for r in group], [r["gain_pp"] for r in group])
                                 if group and all(r["homophily_query_percent"] is not None for r in group) else None,
                                 pdf_seed7_pearson_query=reference[str(shot)]["pearson_query"]))
    means=[]
    for seed,shot,method in itertools.product(args.seeds,args.shots,METHODS[1:]):
        group=[r for r in points if (r['seed'],r['shot'],r['method'])==(seed,shot,method)]
        if group:
            mean=dict(seed=seed,shot=shot,method=method,bases=len(group))
            for key in ('homophily_query_percent','test_percent','baseline_test_percent','gain_pp'):
                values=[r[key] for r in group if r[key] is not None]
                mean[key]=statistics.mean(values) if values else None
            means.append(mean)
    for name, values in (("resultados", rows), ("status", statuses), ("pontos_homofilia", points), ("correlacoes", correlations), ("medias",means)):
        save_csv(args.output_root / (name + ".csv"), values)
    print(f"Concluidos {len(rows)}/{len(jobs)} registros; {len(points)} pontos com controle sem grafo.", flush=True)
    return points, correlations


def plot(args, points, correlations):
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_root / "_matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for seed in args.seeds:
        fig, axes = plt.subplots(1, len(args.shots), figsize=(6*len(args.shots), 4.5), squeeze=False)
        for ax, shot in zip(axes[0], args.shots):
            for method, color in (("knn-union", "#4477AA"), ("dgcg-4-6", "#EE8833")):
                group = [r for r in points if (r["seed"], r["shot"], r["method"]) == (seed, shot, method)
                         and r["homophily_query_percent"] is not None]
                ax.scatter([r["homophily_query_percent"] for r in group], [r["gain_pp"] for r in group], label="kNN-union" if method=='knn-union' else "DGCG 4-6", color=color)
                for r in group:
                    if r['dataset'] in ('cars_micro','flowers_micro'):
                        ax.annotate(r["dataset"].replace("_micro", "").title(), (r["homophily_query_percent"], r["gain_pp"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
            c = next(r for r in correlations if r["seed"] == seed and r["shot"] == shot)
            ax.set_title(f"{shot}-shot; r = {c['pearson_query']:.2f}; n={c['points']}" if c["pearson_query"] is not None else f"{shot}-shot; correlacao indefinida")
            ax.axhline(0, color="gray", lw=1); ax.grid(alpha=.25)
            ax.set_xlabel("Homofilia nas linhas de query (%)"); ax.set_ylabel("Grafo - sem grafo (pp)"); ax.legend()
        fig.suptitle(f"ResNet-18 congelada; projecao treinada; seed {seed}")
        fig.tight_layout(); fig.savefig(args.output_root / f"homofilia_treinada_seed{seed}.png", dpi=180); plt.close(fig)


def from_existing_results(args, p, audit):
    """Mesmo X/sorteio/grafo: reaproveita h(X,A), substitui somente as acuracias.

    Nao mede homofilia de X Theta e nao apresenta a topologia como aprendida.
    """
    if args.seeds != [7] or args.quick or args.train_missing:
        p.error("Reaproveitamento usa seed7 e protocolo completo. Para outros testes, use --evaluate-models.")
    reference=pdf_reference()
    trained=[json.loads(s) for s in (ROOT/CLAUDE).read_text().splitlines()]
    topology=ROOT/'resultados/testes_claude_20261102/resultados/res_topo_tf.jsonl'
    hom=[json.loads(s) for s in topology.read_text().splitlines()]
    jobs=plan(args,[])
    records=[]
    for j in jobs:
        method={'knn-union':'union','dgcg-4-6':'dgcg'}.get(j['method'],j['method'])
        candidates=[r for r in trained if r['dataset']==j['dataset'] and r['shot']==j['shot']
                    and r['method']==method and r['seed']==7 and r['way']==5 and r['query']==10
                    and not r['trainfree'] and r['feat']=='raw' and r['tag']=='' and r['backbone']=='resnet18@128']
        if len(candidates)!=1:
            raise ValueError(f"Resultado treinado ausente/duplicado: {j}")
        r=candidates[0]
        result=dict(dataset=j['dataset'],shot=j['shot'],seed=7,method=j['method'],
                    test_percent=r['test'],test_ci95_pp=r['ci'],test_episodes=1000,
                    device='cpu',backbone='resnet18',image_size=128,feature_dim=256,
                    training_episodes=2000,classification_reused=True,
                    classification_source=CLAUDE,homophily_reused=method!='no-graph',
                    graph_space='frozen_resnet512_before_theta')
        if method!='no-graph':
            candidates=[h for h in hom if h['dataset']==j['dataset'] and h['shot']==j['shot']
                        and h['method']==method and h['tag']=='' and h['seed']==7 and h['feat']=='raw'
                        and h['backbone']=='resnet18@128' and h['trainfree']]
            if len(candidates)!=1:
                raise ValueError(f"Homofilia ausente/duplicada: {j}")
            h=candidates[0]
            result.update(homophily_all_percent=h['hom_all'],homophily_query_percent=h['hom_query'],
                          diagnostic_episodes=200,homophily_source=str(topology.relative_to(ROOT)))
        records.append(result)
    print(f"{len(records)} resultados treinados em CPU + homofilia das arestas fixas; zero treinos novos.",flush=True)
    if args.dry_run:
        return 0
    args.output_root.mkdir(parents=True,exist_ok=True)
    manifest=args.output_root/'manifest.json'
    configuration=dict(mode='existing_results',datasets=args.datasets,shots=args.shots,seeds=args.seeds,
                       source_sha256=audit['sources_sha256'],topology_sha256=sha256(topology),
                       code_sha256=source_digest(),runner_sha256=sha256(Path(__file__)))
    if manifest.exists() and json.loads(manifest.read_text())['configuration']!=configuration:
        p.error("Outra configuracao/codigo nesta pasta; use outro --output-root.")
    atomic_json(manifest,dict(configuration=configuration,pdf_reference=reference,
                definition='Homofilia de X congelado, linhas de query, sem autolacos; ganho usa modelos treinados256',
                justification='Arestas independem de Theta; homofilia identica no mesmo episodio antes/depois do treino.'))
    for job,result in zip(jobs,records):
        folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'diagnostic.json',result)
    points,correlations=summarize(args,jobs,reference)
    plot(args,points,correlations)
    return 0


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.training_curve:
        args.evaluate_models = True
        args.train_missing = True
        if args.output_root == Path("runs/homofilia_treinada_resnet"):
            args.output_root = Path("runs/homofilia_curva_treino")
    for key in ("datasets", "shots", "seeds"):
        if len(set(getattr(args,key))) != len(getattr(args,key)):
            p.error(f"Valores repetidos: {key}")
    if min(args.seeds) < 0 or args.cpu_threads < 1:
        p.error("Seeds >=0 e cpu-threads >=1.")
    args.output_root = args.output_root.resolve(); args.cache_dir = args.cache_dir.resolve()
    audit = audit_results(ROOT)
    if not args.evaluate_models:
        return from_existing_results(args,p,audit)
    index = [] if args.training_curve else checkpoint_index([args.output_root, *args.checkpoint_roots])
    jobs = plan(args, index)
    print(f"{len(jobs)} modelos; {sum(j['checkpoint'] is not None for j in jobs)} checkpoints encontrados; ResNet/128, projecao256; device={args.device}", flush=True)
    if args.dry_run:
        for j in jobs:
            print(j["dataset"], j["shot"], j["seed"], j["method"], j["checkpoint"] or "TREINO FALTANTE")
        return 0
    if any(j["checkpoint"] is None and not (Path(j["folder"])/"diagnostic.json").exists() for j in jobs) and not args.train_missing:
        p.error("Faltam checkpoints treinados. Confira --dry-run; forneca --checkpoint-roots ou autorize --train-missing.")
    device = select_device(args.device); torch.set_num_threads(args.cpu_threads)
    reference = pdf_reference()
    config = {k: str(v) if isinstance(v,Path) else [str(x) for x in v] if k=="checkpoint_roots" else v
              for k,v in vars(args).items() if k not in ("dry_run", "train_missing")}
    config.update(code_sha256=source_digest(), runner_sha256=sha256(Path(__file__)),
                  workbook_sha256=audit["workbook_sha256"], sources_sha256=audit["sources_sha256"],
                  reference_topology_sha256=sha256(ROOT/"resultados/testes_claude_20261102/resultados/res_topo_tf.jsonl"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["configuration"] != config:
        p.error("Outra configuracao/codigo nesta pasta; escolha outro --output-root.")
    if not manifest.exists():
        atomic_json(manifest, dict(configuration=config, pdf_reference=reference,
                    definition="sum weights between same labels / sum weights; query rows; exclude diagonal",
                    graph_space="frozen_resnet512_before_theta; topology independent of learned theta"))
    try:
        for dataset in args.datasets:
            pending = [j for j in jobs if j["dataset"] == dataset and not (Path(j["folder"])/"diagnostic.json").exists()]
            if not pending:
                continue
            a = core_args(args,dataset,args.shots[0],args.seeds[0],"no-graph",args.output_root/"_features"/dataset)
            seed_everything(a.seed)
            cached = args.cache_dir / "org/openml/www/datasets" / str(DATASETS[dataset])
            labels = list(cached.glob("labels.csv"))+list(cached.glob("*/labels.csv"))
            records = load_local_meta_album(labels[0].parent) if len(labels)==1 else download_meta_album(DATASETS[dataset],args.cache_dir)
            features,dim = extract_or_load_features(records,a,device)
            if dim != 512:
                raise ValueError("Este experimento requer ResNet-18 com 512 atributos.")
            for job in pending:
                folder=Path(job["folder"]);folder.mkdir(parents=True,exist_ok=True)
                a=core_args(args,dataset,job["shot"],job["seed"],job["method"],folder/"model")
                splits=split_classes(sorted(features),5,[10,5,5],a.seed)
                sampler=FeatureEpisodeSampler(features,5,a.n_shot,10)
                seed_everything(a.seed);model=build_model(a)
                checkpoint=job["checkpoint"]
                curve = None
                if checkpoint:
                    # Apenas checkpoints locais do projeto, com metadados validados.
                    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
                    if not compatible(state["args"],a,dataset,checkpoint) or state["class_splits"]!=splits:
                        raise ValueError(f"Checkpoint incompatível: {checkpoint}")
                    model.load_state_dict(state["model_state_dict"]);model.to(device)
                    metrics=evaluate(model,sampler,splits["test"],a.test_episodes,5,device,seed=a.seed+303)
                else:
                    a.work_dir.mkdir(parents=True,exist_ok=True)
                    if args.training_curve:
                        model,metrics,path,curve=train_with_curve(model,sampler,splits,a,device)
                    else:
                        model,metrics,path=train_model(model,sampler,splits,a,device)
                    checkpoint=str(path)
                result=dict(dataset=dataset,shot=a.n_shot,seed=a.seed,method=job["method"],
                            test_percent=100*metrics.accuracy,test_ci95_pp=100*metrics.ci95,
                            test_episodes=a.test_episodes,device=str(device),backbone="resnet18",image_size=128,
                            feature_dim=256,training_episodes=a.train_episodes,
                            checkpoint=checkpoint,checkpoint_sha256=sha256(Path(checkpoint)),
                            checkpoint_reused=job["checkpoint"] is not None)
                if curve is not None:
                    result["curve"] = curve
                if model.model_name=="proto-sgc":
                    result.update(measure_homophily(model,sampler,splits["test"],5,20 if args.quick else 200,device,a.seed+909))
                atomic_json(folder/"diagnostic.json",result)
                summarize(args,jobs,reference)
    finally:
        points,correlations=summarize(args,jobs,reference)
        if args.training_curve:
            curves = paired_curves(args.output_root, jobs)
            for name, rows in zip(
                ("curva_episodios", "curva_blocos", "primeira_vantagem", "correlacoes_entre_bases"),
                curves,
                strict=True,
            ):
                save_csv(args.output_root / f"{name}.csv", rows)
    if points:
        plot(args,points,correlations)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
