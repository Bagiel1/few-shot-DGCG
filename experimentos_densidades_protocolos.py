"""Densidade DGCG x protocolos; nao modifica a bateria de graus existente."""

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess

import experimentos_metricas_graus as base


DEFAULT_DENSITIES = [.025, .05, .075, .10, .125, .15, .175, .20,
                     .25, .30, .40, .50, .60, .75, .90]
GROUP_KEYS = ('dataset', 'n_way', 'n_shot', 'n_query', 'seed')


def density(value):
    try:
        value = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Densidade deve ser um numero.') from exc
    if not math.isfinite(value) or not 0 < value < 1:
        raise argparse.ArgumentTypeError('Densidade deve satisfazer 0 < rho < 1.')
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=base.DATASETS, default=list(base.DATASETS))
    p.add_argument('--ways', nargs='+', type=int, default=[2, 3, 5])
    p.add_argument('--shots', nargs='+', type=int, default=[1, 5, 10])
    p.add_argument('--queries', nargs='+', type=int, default=[1, 5, 10])
    p.add_argument('--densities', nargs='+', type=density, default=DEFAULT_DENSITIES)
    p.add_argument('--seeds', nargs='+', type=int, default=[7])
    p.add_argument('--output-root', type=Path, default=Path('runs/busca_densidades_protocolos'))
    p.add_argument('--cache-dir', type=Path, default=Path('data/openml'))
    p.add_argument('--baseline-root', type=Path, default=Path('runs/busca_metricas_graus'),
                   help='Resultados 4-6 ja existentes, somente leitura; nunca reexecutados.')
    p.add_argument('--feature-cache-root', type=Path, default=Path('runs/busca_metricas_graus/_work'),
                   help='Copia caches de embeddings existentes para a nova area de trabalho.')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    for flag in ['quick', 'dry-run', 'summarize-only', 'no-resume', 'fail-fast']:
        p.add_argument('--'+flag, action='store_true')
    return p


def jobs(args):
    for dataset, way, shot, query, seed, rho in itertools.product(
        args.datasets, args.ways, args.shots, args.queries, args.seeds, args.densities,
    ):
        # repr(float) nao arredonda alvos customizados diferentes para a mesma pasta.
        target = str(rho)
        folder = (args.output_root / dataset / f'{way}way_{shot}shot_{query}query'
                  / f'seed{seed}' / ('dgcg-rho'+target.replace('.', 'p')))
        yield dict(dataset=dataset, n_way=way, n_shot=shot, n_query=query, seed=seed,
                   metodo='dgcg', modo='densidade', alvo=target, directory=str(folder))


def command(job, args):
    # Reusa defaults, cache OpenML e split do runner anterior, sem altera-lo.
    cmd = base.command(job, args)
    index = cmd.index('--dgcg-target-degree')
    del cmd[index:index+3]
    cmd += ['--dgcg-target-density', job['alvo'], job['alvo']]
    return cmd


def seed_feature_cache(args, dataset):
    """Copia apenas embeddings; nunca compartilha checkpoints/JSONs mutaveis."""
    source = args.feature_cache_root / dataset
    work = args.output_root / '_work' / dataset
    work.mkdir(parents=True, exist_ok=True)
    for cached in source.glob('features_resnet18_*.pt'):
        destination = work / cached.name
        if not destination.exists():
            shutil.copy2(cached, destination)
    return work


def flatten(job, result):
    row = base.flatten(job, result)
    candidates = min(result['args']['dgcg_candidate_k'], result['args']['dgcg_list_size']-1,
                     row['episode_nodes']-1)
    rho = float(job['alvo'])
    row.update(densidade_alvo=rho, candidatos_por_no=candidates,
               grau_alvo_equivalente=rho*candidates)
    for prefix in ['val', 'test']:
        row[f'{prefix}_desvio_densidade'] = row[f'{prefix}_density_mean'] - rho
    return row


def baseline_comparison(job, result, baseline_root):
    path = (baseline_root / job['dataset']
            / f"{job['n_way']}way_{job['n_shot']}shot_{job['n_query']}query"
            / f"seed{job['seed']}" / 'dgcg-deg4-6' / 'metrics.json')
    if not path.is_file():
        return {'baseline_status': 'ausente'}
    try:
        baseline = json.loads(path.read_text())
    except (OSError, ValueError):
        return {'baseline_status': 'indisponivel'}
    # So compara se a condicao experimental e igual, exceto grau/densidade.
    ignored = {'work_dir', 'cache_dir', 'data_root', 'openml_id', 'device',
               'dgcg_target_degree', 'dgcg_target_density'}
    a, b = result['args'], baseline['args']
    mismatched = sorted(k for k in set(a) | set(b)
                        if k not in ignored and a.get(k) != b.get(k))
    if baseline.get('class_splits') != result.get('class_splits'):
        mismatched.append('class_splits')
    if (b.get('dgcg_target_degree') != [4., 6.]
            or b.get('dgcg_target_density') is not None
            or b.get('dgcg_threshold') is not None):
        mismatched.append('baseline_target')
    if mismatched:
        return {'baseline_status': 'incompativel', 'baseline_diferencas': ';'.join(mismatched)}
    return dict(baseline_status='compativel',
        baseline_val_accuracy=100*baseline['validation']['accuracy'],
        baseline_test_accuracy=100*baseline['test']['accuracy'],
        ganho_teste_vs_4_6_pp=100*(result['test']['accuracy']-baseline['test']['accuracy']))


def select_densities(rows, plan):
    groups, expected = {}, {}
    for job in plan:
        expected.setdefault(tuple(job[k] for k in GROUP_KEYS), set()).add(job['alvo'])
    for row in rows:
        groups.setdefault(tuple(row[k] for k in GROUP_KEYS), []).append(row)
    selected = []
    for key, group in groups.items():
        if {r['alvo'] for r in group} != expected[key]:
            continue
        best = max(r['val_accuracy'] for r in group)
        ties = sorted((r for r in group if r['val_accuracy'] == best), key=lambda r:float(r['alvo']))
        # Menor densidade desempata, sem olhar teste; todos os empates sao exportados.
        selected.append(dict(ties[0], alvos_empatados=';'.join(r['alvo'] for r in ties),
                             numero_alvos_testados=len(group)))
    return selected


def density_variation(selected):
    rows = base.degree_variation(selected)
    rename = {'amplitude_centro': 'amplitude_densidade',
              'desvio_centro': 'desvio_densidade', 'intervalos_distintos': 'densidades_distintas'}
    return [{rename.get(k,k):v for k,v in row.items()
             if k not in ['amplitude_limite_inferior', 'amplitude_limite_superior']}
            for row in rows]


def plot_results(rows, selected, root):
    os.environ.setdefault('MPLCONFIGDIR', str(root/'_matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    charts = root/'graficos'
    charts.mkdir(exist_ok=True)
    # Curvas apenas de grupos com TODOS os alvos; evita comparar pools diferentes.
    complete = {tuple(r[k] for k in GROUP_KEYS) for r in selected}
    for dataset in sorted({r['dataset'] for r in rows}):
        base_rows = [r for r in rows if r['dataset']==dataset]
        chosen = [r for r in selected if r['dataset']==dataset]
        curves = [r for r in base_rows if tuple(r[k] for k in GROUP_KEYS) in complete]
        if curves:
            ways = sorted({r['n_way'] for r in curves})
            fig, axes = plt.subplots(1,len(ways),figsize=(5*len(ways),4),squeeze=False)
            for ax, way in zip(axes[0],ways):
                rhos = sorted({float(r['alvo']) for r in curves if r['n_way']==way})
                for split,label in [('val','Validacao'),('test','Teste descritivo')]:
                    values=[statistics.mean(r[f'{split}_accuracy'] for r in curves
                            if r['n_way']==way and float(r['alvo'])==rho) for rho in rhos]
                    ax.plot(rhos,values,marker='o',label=label)
                ax.set(xlabel='Densidade-alvo',ylabel='Acuracia (%)',title=f'{way}-way')
                ax.legend()
            fig.suptitle(f'{dataset} - media de shot/query/seeds completos')
            fig.tight_layout();fig.savefig(charts/f'{dataset}_curvas_densidade.png',dpi=160);plt.close(fig)
        for seed in sorted({r['seed'] for r in chosen}):
            group=[r for r in chosen if r['seed']==seed]
            ways=sorted({r['n_way'] for r in group})
            shots=sorted({r['n_shot'] for r in group});queries=sorted({r['n_query'] for r in group})
            fig,axes=plt.subplots(1,len(ways),figsize=(5*len(ways),4),squeeze=False)
            for ax,way in zip(axes[0],ways):
                values=np.full((len(shots),len(queries)),np.nan)
                for r in group:
                    if r['n_way']==way:values[shots.index(r['n_shot']),queries.index(r['n_query'])]=float(r['alvo'])
                ax.set_facecolor('#dddddd')
                ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
                for (i,j),value in np.ndenumerate(values):
                    if np.isfinite(value):ax.text(j,i,f'{value:g}',ha='center',va='center',color='white',
                                               bbox=dict(facecolor='black',alpha=.4,edgecolor='none'))
                ax.set(xticks=range(len(queries)),xticklabels=queries,yticks=range(len(shots)),
                       yticklabels=shots,xlabel='n-query',ylabel='n-shot',title=f'{way}-way')
            fig.suptitle(f'{dataset} - densidade escolhida na validacao - seed {seed}')
            fig.tight_layout();fig.savefig(charts/f'{dataset}_escolhas_seed{seed}.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5))
        for rho in sorted({float(r['alvo']) for r in base_rows}):
            group=[r for r in base_rows if float(r['alvo'])==rho]
            ax.scatter([r['episode_nodes'] for r in group],[r['val_density_mean'] for r in group],label=f'{rho:g}',alpha=.6)
        ax.set(xlabel='Nos = way x (shot + query)',ylabel='Densidade realizada na validacao',title=dataset)
        ax.legend(title='Densidade-alvo',bbox_to_anchor=(1.01,1),loc='upper left',fontsize=8)
        fig.tight_layout();fig.savefig(charts/f'{dataset}_densidade_realizada.png',dpi=160);plt.close(fig)


def summarize(root, plan, baseline_root):
    rows = []
    for job in plan:
        path=Path(job['directory'])/'metrics.json'
        if path.is_file():
            result=json.loads(path.read_text())
            row=flatten(job,result)
            row.update(baseline_comparison(job,result,baseline_root))
            rows.append(row)
    selected=select_densities(rows,plan)
    base.save_csv(root/'resultados_densidades_protocolos.csv',rows)
    base.save_csv(root/'melhores_densidades_por_protocolo.csv',selected)
    base.save_csv(root/'variacao_densidades.csv',density_variation(selected))
    print(f'Resultados: {len(rows)}/{len(plan)}; protocolos com grade completa: {len(selected)}',flush=True)
    if rows:plot_results(rows,selected,root)
    return len(rows)


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if any(w<2 or w>5 for w in args.ways):p.error('way deve estar entre 2 e 5.')
    if min(args.shots+args.queries)<1 or max(args.shots)+max(args.queries)>40:
        p.error('shot/query positivos e shot+query <=40 nos datasets Micro.')
    if min(args.seeds)<0:p.error('Seeds devem ser nao negativas.')
    for name in ['datasets','ways','shots','queries','seeds','densities']:
        if len(set(getattr(args,name)))!=len(getattr(args,name)):p.error(f'Valores repetidos em --{name}.')
    for name in ['output_root','cache_dir','baseline_root','feature_cache_root']:
        setattr(args,name,getattr(args,name).expanduser().resolve())
    if args.output_root==args.baseline_root:
        p.error('Use uma pasta de saida diferente da bateria de graus.')
    manifest=args.output_root/'manifest.json'
    if args.summarize_only:
        if not manifest.is_file():p.error('Nao existe manifesto nessa pasta.')
        saved=json.loads(manifest.read_text())
        summarize(args.output_root,saved['runs'],Path(saved['configuration']['baseline_root']))
        return 0
    plan=list(jobs(args))
    print(f'{len(plan)} execucoes DGCG | {len(args.datasets)} bases | seeds={args.seeds}',flush=True)
    print('Densidades:', ' '.join(f'{rho:g}' for rho in args.densities),flush=True)
    print('Sem extras; rankings episodicos; split 10/5/5; baselines nao reexecutadas.',flush=True)
    if args.dry_run:
        for job in plan:print(shlex.join(command(job,args)))
        return 0
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                   if k not in ['dry_run','summarize_only','no_resume','fail_fast']}
    if manifest.exists() and json.loads(manifest.read_text())['configuration']!=configuration:
        p.error('Pasta pertence a outra configuracao; use outro --output-root.')
    args.output_root.mkdir(parents=True,exist_ok=True)
    if not manifest.exists():manifest.write_text(json.dumps(dict(configuration=configuration,runs=plan),indent=2)+'\n')
    failures=0
    try:
        for index,job in enumerate(plan,1):
            folder=Path(job['directory']);result_path=folder/'metrics.json'
            print(f"[{index}/{len(plan)}] {job['dataset']} rho={job['alvo']} "
                  f"{job['n_way']}w/{job['n_shot']}s/{job['n_query']}q seed={job['seed']}",flush=True)
            if result_path.is_file() and not args.no_resume:
                print('  concluido; pulando.',flush=True);continue
            folder.mkdir(parents=True,exist_ok=True)
            work=seed_feature_cache(args,job['dataset'])
            before={path:path.stat().st_mtime_ns for path in work.glob('*.metrics.json')}
            log_path=folder/'run.log'
            with log_path.open('w') as log:
                completed=subprocess.run(command(job,args),cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT,
                                         env=dict(os.environ,PYTHONUNBUFFERED='1'))
            fresh=[path for path in work.glob('*.metrics.json') if before.get(path)!=path.stat().st_mtime_ns]
            if completed.returncode or len(fresh)!=1:
                failures+=1
                print(f'  FALHOU: veja {log_path}',flush=True)
                print('\n'.join(log_path.read_text(errors='replace').splitlines()[-12:]),flush=True)
                if args.fail_fast:break
                continue
            result=json.loads(fresh[0].read_text())
            shutil.copy2(result['checkpoint'],folder/'checkpoint.pt')
            result['checkpoint']=str(folder/'checkpoint.pt')
            result_path.write_text(json.dumps(result,indent=2)+'\n')
            test=result['test']
            print(f"  teste={100*test['accuracy']:.2f}% | F1={100*test['macro_f1']:.2f}% | "
                  f"BA={100*test['balanced_accuracy']:.2f}% | rho_real={test['graph']['density_mean']:.4f}",flush=True)
    finally:
        summarize(args.output_root,plan,args.baseline_root)
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
