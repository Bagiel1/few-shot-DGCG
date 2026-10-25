"""Bateria reproduzivel e separada; nao modifica CSVs/resultados anteriores."""

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASETS = dict(zip([
    "bacteria_micro", "plankton_micro", "flowers_micro", "sports_micro", "birds_micro",
    "plant_village_micro", "textures_micro", "manufacturing_micro", "cars_micro", "resisc_micro",
], range(44237, 44247)))
METHODS = ("knn-in", "knn-out", "knn-mutual", "knn-union", "no-graph", "dgcg")
METRICS = ("accuracy", "ci95", "macro_f1", "macro_f1_ci95", "balanced_accuracy",
           "balanced_accuracy_ci95", "loss")


def parse_degree(value):
    parts = value.split(":")
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use MIN:MAX ou um valor de grau.")
    try:
        low, high = map(float, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Graus devem ser numeros.") from exc
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise argparse.ArgumentTypeError("Use 0 < MIN <= MAX, valores finitos.")
    return low, high


def degree_name(degree):
    return f"{degree[0]:g}:{degree[1]:g}"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--ways", nargs="+", type=int, default=[2, 3, 5])
    p.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--queries", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--degrees", nargs="+", type=parse_degree,
                   default=[(2., 4.), (4., 6.), (6., 8.), (8., 12.)])
    p.add_argument("--seeds", nargs="+", type=int, default=[7])
    p.add_argument("--output-root", type=Path, default=Path("runs/busca_metricas_graus"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/openml"))
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p


def jobs(args):
    for dataset, way, shot, query, seed, method in itertools.product(
        args.datasets, args.ways, args.shots, args.queries, args.seeds, args.methods,
    ):
        for degree in (args.degrees if method == "dgcg" else [None]):
            target = degree_name(degree) if degree else "-"
            label = method + ("-deg" + target.replace(":", "-") if degree else "")
            folder = args.output_root / dataset / f"{way}way_{shot}shot_{query}query" / f"seed{seed}" / label
            yield dict(dataset=dataset, n_way=way, n_shot=shot, n_query=query,
                       seed=seed, metodo=method, alvo=target, directory=str(folder))


def command(job, args):
    # Mesma pasta de embeddings por base; checkpoint/resultados copiados por execucao.
    work = args.output_root / "_work" / job["dataset"]
    cmd = [sys.executable, str(ROOT / "protosgc_fewshot.py"),
           "--work-dir", str(work), "--class-split", "10", "5", "5",
           "--n-way", str(job["n_way"]), "--n-shot", str(job["n_shot"]),
           "--n-query", str(job["n_query"]), "--seed", str(job["seed"]),
           "--device", args.device]
    cache = args.cache_dir / "org/openml/www/datasets" / str(DATASETS[job["dataset"]])
    labels = list(cache.glob("labels.csv")) + list(cache.glob("*/labels.csv"))
    if len(labels) == 1:
        cmd += ["--data-root", str(labels[0].parent)]
    else:
        cmd += ["--openml-id", str(DATASETS[job["dataset"]]), "--cache-dir", str(args.cache_dir)]
    if job["metodo"] == "no-graph":
        cmd += ["--model", "protonet"]
    else:
        cmd += ["--graph-type", job["metodo"]]
    if job["metodo"] == "dgcg":
        cmd += ["--dgcg-target-degree", *[f"{v:g}" for v in parse_degree(job["alvo"])]]
    if args.quick:
        cmd += ["--quick"]
    return cmd


def flatten(job, result):
    row = {key: value for key, value in job.items() if key != "directory"}
    row["episode_nodes"] = job["n_way"] * (job["n_shot"] + job["n_query"])
    for split, prefix in [("validation", "val"), ("test", "test")]:
        for metric in METRICS:
            value = result[split][metric]
            row[f"{prefix}_{metric}"] = value if metric == "loss" else value * 100
        for key, value in result[split]["graph"].items():
            row[f"{prefix}_{key}"] = value
    return row


def save_csv(path, rows):
    if not rows:
        # Uma nova sumarizacao parcial nao deve manter um CSV antigo enganoso.
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_degrees(rows, plan):
    """Seleciona so por validacao e so se todos os alvos planejados terminaram."""
    keys = ("dataset", "n_way", "n_shot", "n_query", "seed")
    groups, expected = {}, {}
    for job in plan:
        if job["metodo"] == "dgcg":
            expected.setdefault(tuple(job[k] for k in keys), set()).add(job["alvo"])
    for row in rows:
        if row["metodo"] == "dgcg":
            groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    selected = []
    for key, options in groups.items():
        if {r["alvo"] for r in options} != expected[key]:
            continue
        best_value = max(r["val_accuracy"] for r in options)
        ties = sorted([r for r in options if r["val_accuracy"] == best_value],
                      key=lambda r: parse_degree(r["alvo"]))
        # Empate: prefere o default 4:6, depois menor intervalo; nunca usa teste.
        winner = next((r for r in ties if r["alvo"] == "4:6"), ties[0])
        entry = dict(winner, alvos_empatados=";".join(r["alvo"] for r in ties),
                     numero_alvos_testados=len(options))
        baseline = next((r for r in options if r["alvo"] == "4:6"), None)
        entry["ganho_teste_vs_4_6_pp"] = (winner["test_accuracy"] - baseline["test_accuracy"]
                                          if baseline else "")
        selected.append(entry)
    return selected


def degree_variation(selected):
    """Isola cada eixo dentro da mesma base/seed e demais eixos constantes."""
    results = []
    for axis in ("n_way", "n_shot", "n_query"):
        fixed = [k for k in ("dataset", "seed", "n_way", "n_shot", "n_query") if k != axis]
        groups = {}
        for row in selected:
            groups.setdefault(tuple(row[k] for k in fixed), []).append(row)
        for key, group in groups.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda r: r[axis])
            lows, highs = zip(*(parse_degree(r["alvo"]) for r in group))
            mids = [(a+b)/2 for a,b in zip(lows, highs)]
            results.append(dict(zip(fixed, key), eixo_variado=axis,
                escolhas=";".join(f"{r[axis]}->{r['alvo']}" for r in group),
                intervalos_distintos=len({r['alvo'] for r in group}),
                amplitude_centro=max(mids)-min(mids),
                desvio_centro=statistics.pstdev(mids),
                amplitude_limite_inferior=max(lows)-min(lows),
                amplitude_limite_superior=max(highs)-min(highs)))
    return results


def plot_results(rows, selected, root):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'_matplotlib_cache'))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    charts = root / "graficos"
    charts.mkdir(exist_ok=True)
    for dataset in sorted({r["dataset"] for r in rows}):
        base = [r for r in rows if r["dataset"] == dataset]
        for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
            labels = sorted({r['metodo'] + (f" {r['alvo']}" if r['metodo']=='dgcg' else '') for r in base})
            values = [statistics.mean(r[f"test_{metric}"] for r in base
                      if r['metodo'] + (f" {r['alvo']}" if r['metodo']=='dgcg' else '') == label) for label in labels]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.barh(labels, values)
            ax.set(xlim=(0,100), xlabel=f"Teste: {metric} (%)",
                   title=f"{dataset} - media descritiva das execucoes concluidas")
            fig.tight_layout();fig.savefig(charts/f"{dataset}_{metric}.png", dpi=160);plt.close(fig)
        chosen = [r for r in selected if r['dataset'] == dataset]
        if not chosen:
            continue
        targets = sorted({r['alvo'] for r in base if r['metodo']=='dgcg'}, key=parse_degree)
        ways = sorted({r['n_way'] for r in chosen})
        seeds = sorted({r['seed'] for r in chosen})
        for seed in seeds:
            fig, axes = plt.subplots(1,len(ways),figsize=(5*len(ways),4), squeeze=False)
            palette = plt.get_cmap('viridis',len(targets))
            norm = BoundaryNorm([v-.5 for v in range(len(targets)+1)],len(targets))
            for ax, way in zip(axes[0], ways):
                shot = sorted({r['n_shot'] for r in chosen})
                query = sorted({r['n_query'] for r in chosen})
                # Cinza = configuracao ausente, nunca um resultado inferido.
                ax.set_facecolor('#dddddd')
                for row in chosen:
                    if row['seed'] != seed or row['n_way'] != way:
                        continue
                    x,y = query.index(row['n_query']),shot.index(row['n_shot'])
                    ax.add_patch(plt.Rectangle((x-.5,y-.5),1,1,
                                 color=palette(norm(targets.index(row['alvo'])))))
                    ax.text(x,y,row['alvo'],ha='center',va='center',color='white',
                            bbox=dict(facecolor='black',alpha=.45,edgecolor='none'))
                ax.set(xticks=range(len(query)),xticklabels=query,yticks=range(len(shot)),
                       yticklabels=shot,xlim=(-.5,len(query)-.5),ylim=(len(shot)-.5,-.5),
                       xlabel='n-query',ylabel='n-shot',title=f'{way}-way')
            fig.suptitle(f'{dataset} - intervalo escolhido pela validacao - seed {seed}')
            fig.tight_layout();fig.savefig(charts/f'{dataset}_intervalos_seed{seed}.png',dpi=160);plt.close(fig)
        dgcg = [r for r in base if r['metodo']=='dgcg']
        fig, ax = plt.subplots(figsize=(9,5))
        for target in targets:
            subset=[r for r in dgcg if r['alvo']==target]
            ax.scatter([r['episode_nodes'] for r in subset],
                       [r['val_degree_mean'] for r in subset],label=target,alpha=.6)
        ax.set(xlabel='Nos = way x (shot + query)',ylabel='Grau medio realizado na validacao',title=dataset)
        ax.legend(title='Intervalo-alvo');fig.tight_layout()
        fig.savefig(charts/f'{dataset}_grau_realizado.png',dpi=160);plt.close(fig)


def summarize(root, plan):
    rows = []
    for job in plan:
        path = Path(job['directory']) / 'metrics.json'
        if path.is_file():
            result = json.loads(path.read_text())
            rows.append(flatten(job, result))
    selected = select_degrees(rows, plan)
    save_csv(root/'resultados.csv', rows)
    save_csv(root/'melhores_graus_por_protocolo.csv', selected)
    save_csv(root/'variacao_intervalos.csv', degree_variation(selected))
    print(f'Resultados: {len(rows)}/{len(plan)}; protocolos DGCG com grade completa: {len(selected)}', flush=True)
    if rows:
        plot_results(rows, selected, root)
    return len(rows)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if any(w < 2 or w > 5 for w in args.ways):
        p.error('n-way deve estar entre 2 e 5, para split fixo 10/5/5.')
    if min(args.shots + args.queries) < 1 or max(args.shots)+max(args.queries)>40:
        p.error('shot/query positivos e shot+query <= 40 nos datasets Micro.')
    if min(args.seeds) < 0:
        p.error('Seeds devem ser nao negativas.')
    for name in ['datasets','methods','ways','shots','queries','seeds','degrees']:
        if len(set(getattr(args,name))) != len(getattr(args,name)):
            p.error(f'Valores repetidos em --{name}.')
    args.output_root = args.output_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.summarize_only:
        manifest = args.output_root / 'manifest.json'
        if not manifest.is_file():
            p.error('Nao existe manifesto nessa pasta.')
        summarize(args.output_root,json.loads(manifest.read_text())['runs'])
        return 0
    plan = list(jobs(args))
    print(f'{len(plan)} execucoes | {len(args.datasets)} bases | seeds={args.seeds}',flush=True)
    print('Sem extras; rankings episodicos; kNN k=5; defaults de treino preservados.',flush=True)
    if args.dry_run:
        for job in plan:
            print(shlex.join(command(job,args)))
        return 0
    args.output_root.mkdir(parents=True,exist_ok=True)
    manifest = args.output_root/'manifest.json'
    configuration = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                     if k not in ['dry_run','summarize_only','no_resume','fail_fast']}
    configuration=json.loads(json.dumps(configuration))
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['configuration'] != configuration:
            p.error('Esta pasta pertence a outra configuracao. Use outro --output-root para nao misturar resultados.')
    else:
        manifest.write_text(json.dumps(dict(configuration=configuration,runs=plan),indent=2)+'\n')
    failures=0
    try:
        for index,job in enumerate(plan,1):
            folder=Path(job['directory']);result_path=folder/'metrics.json'
            print(f"[{index}/{len(plan)}] {job['dataset']} {job['metodo']} {job['alvo']} "
                  f"{job['n_way']}w/{job['n_shot']}s/{job['n_query']}q seed={job['seed']}",flush=True)
            if result_path.is_file() and not args.no_resume:
                print('  concluido; pulando.',flush=True);continue
            folder.mkdir(parents=True,exist_ok=True)
            work=args.output_root/'_work'/job['dataset']
            # Somente JSONs novos/modificados por esta execucao podem ser copiados.
            before={path:path.stat().st_mtime_ns for path in work.glob('*.metrics.json')}
            env=dict(os.environ,PYTHONUNBUFFERED='1')
            with (folder/'run.log').open('w') as log:
                completed=subprocess.run(command(job,args),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            fresh=[path for path in work.glob('*.metrics.json')
                   if before.get(path)!=path.stat().st_mtime_ns]
            if completed.returncode or len(fresh)!=1:
                failures+=1
                print(f'  FALHOU: veja {folder / "run.log"}',flush=True)
                if args.fail_fast:break
                continue
            result=json.loads(fresh[0].read_text())
            shutil.copy2(result['checkpoint'],folder/'checkpoint.pt')
            result['checkpoint']=str(folder/'checkpoint.pt')
            result_path.write_text(json.dumps(result,indent=2)+'\n')
            print(f"  teste={100*result['test']['accuracy']:.2f}% | "
                  f"F1={100*result['test']['macro_f1']:.2f}% | "
                  f"BA={100*result['test']['balanced_accuracy']:.2f}%",flush=True)
    finally:
        summarize(args.output_root,plan)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
