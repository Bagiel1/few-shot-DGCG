"""Transfere alvos DGCG escolhidos no Micro para Micro/Mini/Extended, sem extras."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

import experimentos_metricas_graus as base
from experimentos_extras import atomic_json, source_digest

# Fontes oficiais: https://meta-album.github.io/datasets/{FLW,CRS}.html
DATASETS = {
    'flowers': {'micro': (44239, (10, 5, 5)), 'mini': (44283, (51, 25, 26)),
                'extended': (44318, (51, 25, 26))},
    'cars': {'micro': (44245, (10, 5, 5)), 'mini': (44289, (98, 49, 49)),
             'extended': (44323, (98, 49, 49))},
}
SOURCES = [base.ROOT/'outputs/analise_experimentos_20261026/fontes'/name for name in
           ('resultados_metricas.csv', 'resultados_densidades_protocolos_dgcg.csv')]
GROUP = ('dataset', 'n_way', 'n_shot', 'n_query', 'metodo', 'target_kind', 'alvo')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--versions', nargs='+', choices=['micro','mini','extended'], default=['micro','mini','extended'])
    p.add_argument('--ways', nargs='+', type=int, default=[5])
    p.add_argument('--shots', nargs='+', type=int, default=[1,5])
    p.add_argument('--queries', nargs='+', type=int, default=[10])
    p.add_argument('--seeds', nargs='+', type=int, default=[7,17,27])
    p.add_argument('--methods', nargs='+', choices=['dgcg','knn-union','no-graph'], default=['dgcg','knn-union','no-graph'])
    p.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    p.add_argument('--output-root', type=Path, default=Path('runs/comparacao_escalas'))
    p.add_argument('--cache-dir', type=Path, default=Path('data/openml'))
    for flag in ['dry-run','summarize-only','quick']:
        p.add_argument('--'+flag, action='store_true')
    return p


def select_micro(args):
    """Somente validacao seed7; teste nao entra na selecao/desempate."""
    candidates = []
    for path, kind in zip(SOURCES, ['degree','density']):
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['metodo']=='dgcg' and int(row['seed'])==7:
                    candidates.append(dict(row, target_kind=kind))
    chosen = {}
    for family, way, shot, query in itertools.product(args.datasets,args.ways,args.shots,args.queries):
        rows = [r for r in candidates if r['dataset']==family+'_micro' and
                tuple(int(r[k]) for k in ['n_way','n_shot','n_query'])==(way,shot,query)]
        if len(rows)!=19 or len({(r['target_kind'],r['alvo']) for r in rows})!=19:
            raise ValueError(f'Grade Micro incompleta/duplicada para {family} {way}/{shot}/{query}: {len(rows)}/19')
        # Empate: grau default4:6, demais graus crescentes, densidades crescentes.
        def order(r):
            target=tuple(float(x) for x in r['alvo'].split(':'))
            return (-float(r['val_accuracy']), 0 if r['target_kind']=='degree' and target==(4.,6.) else
                    1 if r['target_kind']=='degree' else 2, target)
        best=sorted(rows,key=order)[0]
        chosen[(family,way,shot,query)] = dict(target_kind=best['target_kind'], alvo=best['alvo'],
            selection_val_accuracy=float(best['val_accuracy']), selection_seed=7,
            selection_ties=[f"{r['target_kind']}:{r['alvo']}" for r in sorted(rows,key=order)
                            if float(r['val_accuracy'])==float(best['val_accuracy'])])
    return chosen


def jobs(args, selected):
    for family,version,way,shot,query,seed,method in itertools.product(
        args.datasets,args.versions,args.ways,args.shots,args.queries,args.seeds,args.methods):
        dataset=family+'_'+version
        oid,split=DATASETS[family][version]
        target=selected[(family,way,shot,query)] if method=='dgcg' else dict(target_kind='none',alvo='-')
        folder=args.output_root/dataset/f'{way}way_{shot}shot_{query}query'/f'seed{seed}'/method
        yield dict(family=family, version=version, dataset=dataset, openml_id=oid,class_split=list(split),
            n_way=way,n_shot=shot,n_query=query,seed=seed,metodo=method,**target,directory=str(folder))


def command(job,args):
    cmd=[sys.executable,str(base.ROOT/'protosgc_fewshot.py'),'--openml-id',str(job['openml_id']),
         '--cache-dir',str(args.cache_dir),'--work-dir',str(args.output_root/'_work'/job['dataset']),
         '--class-split',*map(str,job['class_split']),'--n-way',str(job['n_way']),
         '--n-shot',str(job['n_shot']),'--n-query',str(job['n_query']),
         '--seed',str(job['seed']),'--device',args.device]
    cache=args.cache_dir/'org/openml/www/datasets'/str(job['openml_id'])
    labels=list(cache.glob('labels.csv'))+list(cache.glob('*/labels.csv'))
    if len(labels)==1:
        cmd+=['--data-root',str(labels[0].parent)]
    if job['metodo']=='no-graph':cmd+=['--model','protonet']
    else:cmd+=['--graph-type',job['metodo']]
    if job['target_kind']=='density':cmd+=['--dgcg-target-density',job['alvo'],job['alvo']]
    if job['target_kind']=='degree':cmd+=['--dgcg-target-degree',*job['alvo'].split(':')]
    if args.quick:cmd+=['--quick']
    return cmd


def matches(job,result):
    a=result.get('args',{})
    expected={k:job[k] for k in ['n_way','n_shot','n_query','seed','openml_id','class_split']}
    expected.update(model='protonet' if job['metodo']=='no-graph' else 'proto-sgc',extra_images=None)
    if job['metodo']!='no-graph':expected['graph_type']=job['metodo']
    if job['target_kind']=='density':expected['dgcg_target_density']=[float(job['alvo'])]*2
    if job['target_kind']=='degree':expected['dgcg_target_degree']=[float(x) for x in job['alvo'].split(':')]
    return all(a.get(k)==v for k,v in expected.items())


def summarize(root,plan):
    rows,status=[],[]
    for job in plan:
        folder=Path(job['directory']);path=folder/'metrics.json';state='pending'
        if path.exists():
            result=json.loads(path.read_text())
            if not matches(job,result):raise ValueError(f'Resultado incompativel: {path}')
            rows.append(base.flatten(job,result));state='complete'
        elif (folder/'status.json').exists():state=json.loads((folder/'status.json').read_text())['status']
        status.append(dict(job,status=state))
    means=[]
    for key in sorted({tuple(j[k] for k in GROUP) for j in plan}):
        group=[r for r in rows if tuple(r[k] for k in GROUP)==key]
        expected=sum(tuple(j[k] for k in GROUP)==key for j in plan)
        row=dict(zip(GROUP,key),concluidos=len(group),planejados=expected,
                 seeds=';'.join(str(r['seed']) for r in group))
        for metric in ['val_accuracy','test_accuracy','test_macro_f1','test_balanced_accuracy']:
            values=[r[metric] for r in group]
            row[metric+'_mean']=statistics.mean(values) if values else ''
            row[metric+'_std_seed']=statistics.stdev(values) if len(values)>1 else ''
        means.append(row)
    for name,data in [('resultados',rows),('status',status),('medias',means)]:base.save_csv(root/(name+'.csv'),data)
    print(f'Concluidos: {len(rows)}/{len(plan)}',flush=True)


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    for k in ['datasets','versions','ways','shots','queries','seeds','methods']:
        if len(set(getattr(args,k)))!=len(getattr(args,k)):p.error(f'Valores repetidos: {k}')
    if not all(2<=w<=5 for w in args.ways) or min(args.shots+args.queries)<1 or max(args.shots)+max(args.queries)>40 or min(args.seeds)<0:
        p.error('Use way2..5, shot/query positivos com soma<=40 e seeds>=0.')
    args.output_root=args.output_root.expanduser().resolve();args.cache_dir=args.cache_dir.expanduser().resolve()
    manifest=args.output_root/'manifest.json'
    if args.summarize_only:
        if not manifest.exists():p.error('Manifesto nao encontrado.')
        summarize(args.output_root,json.loads(manifest.read_text())['runs']);return 0
    selected=select_micro(args);plan=list(jobs(args,selected))
    print(f'{len(plan)} execucoes; sem extras e sem banco reservado; seeds={args.seeds}',flush=True)
    for key,value in selected.items():print('Selecao Micro:',key,value,flush=True)
    if args.dry_run:
        for job in plan:print(shlex.join(command(job,args)))
        return 0
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['dry_run','summarize_only']}
    config.update(code_sha256=source_digest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES})
    if manifest.exists() and json.loads(manifest.read_text())['configuration']!=config:
        p.error('Outra configuracao/codigo nesta pasta; use outro --output-root.')
    from proto_sgc.runtime import select_device
    print('Dispositivo:',select_device(args.device),flush=True)
    args.output_root.mkdir(parents=True,exist_ok=True)
    if not manifest.exists():atomic_json(manifest,dict(configuration=config,runs=plan))
    failure=0
    try:
        for i,job in enumerate(plan,1):
            folder=Path(job['directory']);path=folder/'metrics.json'
            print(f'[{i}/{len(plan)}] {folder.relative_to(args.output_root)}',flush=True)
            if path.exists():
                if not matches(job,json.loads(path.read_text())):raise ValueError(f'Incompativel: {path}')
                continue
            folder.mkdir(parents=True,exist_ok=True);cmd=command(job,args)
            atomic_json(folder/'command.json',cmd);atomic_json(folder/'status.json',{'status':'running'})
            work=args.output_root/'_work'/job['dataset']
            before={p:p.stat().st_mtime_ns for p in work.glob('*.metrics.json')}
            try:
                with (folder/'run.log').open('w') as log:
                    proc=subprocess.run(cmd,cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT,
                                        env=dict(os.environ,PYTHONUNBUFFERED='1'))
                fresh=[p for p in work.glob('*.metrics.json') if before.get(p)!=p.stat().st_mtime_ns]
                result=json.loads(fresh[0].read_text()) if proc.returncode==0 and len(fresh)==1 else None
                if result is None or not matches(job,result):raise RuntimeError(f'Execucao falhou; veja {folder / "run.log"}')
                shutil.copy2(result['checkpoint'],folder/'checkpoint.pt');result['checkpoint']=str(folder/'checkpoint.pt')
                atomic_json(path,result);atomic_json(folder/'status.json',{'status':'complete'})
            except KeyboardInterrupt:
                atomic_json(folder/'status.json',{'status':'interrupted'});raise
            except Exception as exc:
                atomic_json(folder/'status.json',{'status':'failed','error':str(exc)})
                print(str(exc),flush=True);failure=1;break
            summarize(args.output_root,plan)
    except KeyboardInterrupt:failure=130
    finally:summarize(args.output_root,plan)
    return failure


if __name__=='__main__':raise SystemExit(main())
