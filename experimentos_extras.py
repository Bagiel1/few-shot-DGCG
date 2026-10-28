"""Extras sem rotulos: fases separadas, contexto dos rankings ou propagacao."""

import argparse
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

from experimentos_metricas_graus import DATASETS, METHODS, ROOT, command as base_command
from experimentos_metricas_graus import flatten, save_csv, parse_degree, degree_name

PHASES = {"train": ["train"], "val": ["val"], "test": ["test"],
          "train-val": ["train", "val"], "all": ["train", "val", "test"]}
PROTOCOL_SELECTION = {
    "protocol": [2, 10, 10], "criterion": "mean validation accuracy across 10 datasets and 7 configurations",
    "configurations": ["knn-in", "knn-out", "knn-mutual", "knn-union", "no-graph", "dgcg 2:4", "dgcg 4:6"],
    "source": "resultados_metricas.csv (2430 runs, seed 7)",
    "source_sha256": "6fbe5683ff62b970664928414beaafafbc6d652b2394ab80e2e55d27b8007e28",
    "validation_accuracy_percent": 92.155000,
    "warning": "ways changes task difficulty; this is not a universal optimum",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--degrees', nargs='+', type=parse_degree, default=[(2., 4.), (4., 6.)])
    p.add_argument('--n-way', type=int, default=2)
    p.add_argument('--n-shot', type=int, default=10)
    p.add_argument('--n-query', type=int, default=10)
    p.add_argument('--extras-per-class', nargs='+', type=int, default=[10])
    p.add_argument('--bank-per-class', type=int, default=20)
    p.add_argument('--phases', nargs='+', choices=PHASES, default=list(PHASES))
    p.add_argument('--modes', nargs='+', choices=['rankings', 'propagation'], default=['rankings', 'propagation'])
    p.add_argument('--seeds', nargs='+', type=int, default=[7])
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--output-root', type=Path, default=Path('runs/busca_extras'))
    p.add_argument('--cache-dir', type=Path, default=Path('data/openml'))
    p.add_argument('--work-root', type=Path, default=None,
                   help='Pasta de embeddings compartilhados por base; default OUTPUT/_work.')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--fail-fast', action='store_true')
    return p


def jobs(args):
    for dataset, seed, method in itertools.product(args.datasets, args.seeds, args.methods):
        for degree in (args.degrees if method == 'dgcg' else [None]):
            target = degree_name(degree) if degree else '-'
            method_tag = method + ('-deg' + target.replace(':', '-') if degree else '')
            conditions = [('none', 'none', 0)]
            if method != 'no-graph':
                for mode, phase, count in itertools.product(args.modes, args.phases, args.extras_per_class):
                    if mode == 'rankings' and method != 'dgcg':
                        continue  # Nao existe RBO/ranked-list context nos baselines kNN.
                    conditions.append((mode, phase, count))
            for mode, phase, count in conditions:
                tag = f'{mode}-{phase}-u{count}'
                folder = (args.output_root / dataset / f'{args.n_way}way_{args.n_shot}shot_{args.n_query}query'
                          / f'seed{seed}' / method_tag / tag)
                yield dict(dataset=dataset, seed=seed, metodo=method, alvo=target,
                           n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query,
                           extra_mode=mode, extra_phase=phase, extras_per_class=count,
                           extras_total=count*args.n_way, bank_per_class=args.bank_per_class,
                           directory=str(folder))


def command(job, args):
    cmd = base_command(job, args)
    cmd[cmd.index('--work-dir') + 1] = str(args.work_root / job['dataset'])
    cmd += ['--extra-images', str(job['extras_total']), '--extra-bank-per-class', str(args.bank_per_class),
            '--extra-mode', 'propagation' if job['extra_mode'] == 'none' else job['extra_mode'],
            '--extra-phases', *PHASES.get(job['extra_phase'], PHASES['all'])]
    return cmd


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def source_digest():
    paths = [Path(__file__), ROOT/'experimentos_metricas_graus.py', ROOT/'protosgc_fewshot.py']
    paths += sorted((ROOT/'proto_sgc').rglob('*.py'))
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def matches(job, result):
    a = result.get('args', {})
    checks = {'n_way': job['n_way'], 'n_shot': job['n_shot'], 'n_query': job['n_query'],
              'seed': job['seed'], 'extra_images': job['extras_total'],
              'extra_bank_per_class': job['bank_per_class'],
              'extra_mode': 'propagation' if job['extra_mode'] == 'none' else job['extra_mode'],
              'extra_phases': PHASES.get(job['extra_phase'], PHASES['all']),
              'model': 'protonet' if job['metodo'] == 'no-graph' else 'proto-sgc'}
    if job['metodo'] != 'no-graph':
        checks['graph_type'] = job['metodo']
    if job['metodo'] == 'dgcg':
        checks['dgcg_target_degree'] = list(parse_degree(job['alvo']))
    return all(a.get(key) == value for key, value in checks.items())


def summarize(root, plan):
    rows, status = [], []
    for job in plan:
        folder = Path(job['directory'])
        result_path = folder/'metrics.json'
        state = 'pending'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if not matches(job, result):
                raise ValueError(f'Resultado incompativel: {result_path}')
            rows.append(flatten(job, result))
            state = 'complete'
        elif (folder/'status.json').exists():
            state = json.loads((folder/'status.json').read_text())['status']
        status.append(dict(job, status=state, log=str(folder/'run.log')))
    key = lambda r: tuple(r[k] for k in ('dataset', 'seed', 'metodo', 'alvo', 'n_way', 'n_shot', 'n_query'))
    controls = {key(r): r for r in rows if r['extra_mode'] == 'none'}
    for row in rows:
        control = controls.get(key(row))
        for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            row[f'ganho_{metric}_vs_sem_extras_pp'] = (row['test_'+metric] - control['test_'+metric]
                                                      if control else '')
    save_csv(root/'resultados.csv', rows)
    save_csv(root/'status.csv', status)
    group_fields = ('metodo', 'alvo', 'extra_mode', 'extra_phase', 'extras_per_class')
    means = []
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_fields), []).append(row)
    for key_, group in groups.items():
        mean = dict(zip(group_fields, key_), concluidos=len(group),
                    planejados=sum(tuple(j[k] for k in group_fields) == key_ for j in plan))
        for column in ('val_accuracy', 'test_accuracy', 'test_macro_f1', 'test_balanced_accuracy',
                       'ganho_accuracy_vs_sem_extras_pp'):
            values = [r[column] for r in group if r[column] != '']
            mean[column] = statistics.mean(values) if values else ''
        means.append(mean)
    save_csv(root/'medias.csv', means)
    print(f'Resultados salvos: {len(rows)}/{len(plan)} em {root}', flush=True)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.work_root = (args.work_root or args.output_root/'_work').expanduser().resolve()
    manifest_path = args.output_root/'manifest.json'
    if args.summarize_only:
        if not manifest_path.exists():
            p.error('Nao existe manifesto nessa pasta.')
        summarize(args.output_root, json.loads(manifest_path.read_text())['runs'])
        return 0
    if not 2 <= args.n_way <= 5:
        p.error('n-way deve ser de 2 a 5 para o split 10/5/5.')
    if min(args.n_shot, args.n_query, args.bank_per_class, *args.extras_per_class) < 1:
        p.error('Shot/query, reserva e extras por classe devem ser positivos; controle zero e automatico.')
    if args.n_shot + args.n_query + args.bank_per_class > 40:
        p.error('Micro: shot + query + banco por classe deve ser <= 40. Nao reutilize imagens.')
    if max(args.extras_per_class) > args.bank_per_class or min(args.seeds) < 0:
        p.error('Extras por classe nao podem exceder o banco; seeds devem ser >= 0.')
    for name in ('datasets', 'methods', 'degrees', 'extras_per_class', 'phases', 'modes', 'seeds'):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            p.error(f'Valores repetidos em --{name}.')
    plan = list(jobs(args))
    print(f'{len(plan)} execucoes | {len(args.datasets)} bases | '
          f'{args.n_way}way/{args.n_shot}shot/{args.n_query}query | seeds={args.seeds}', flush=True)
    print('Controle pareado sem extras; bancos fixos por split. Rankings: DGCG. Propagacao: DGCG e kNN.', flush=True)
    if args.dry_run:
        for job in plan:
            print(shlex.join(command(job, args)))
        return 0
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('dry_run', 'summarize_only', 'fail_fast')}
    config['code_sha256'] = source_digest()
    config = json.loads(json.dumps(config))
    args.output_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['configuration'] != config:
            p.error('Pasta pertence a outra configuracao/versao de codigo. Escolha outro --output-root.')
    else:
        atomic_json(manifest_path, dict(configuration=config, runs=plan, protocol_selection=PROTOCOL_SELECTION))
    failures, interrupted = 0, False
    try:
        for index, job in enumerate(plan, 1):
            folder = Path(job['directory'])
            result_path = folder/'metrics.json'
            print(f"[{index}/{len(plan)}] {job['dataset']} {job['metodo']} {job['alvo']} "
                  f"{job['extra_mode']} {job['extra_phase']} extras={job['extras_total']} seed={job['seed']}", flush=True)
            if result_path.exists():
                if not matches(job, json.loads(result_path.read_text())):
                    raise ValueError(f'Resultado incompativel: {result_path}')
                print('  concluido; pulando.', flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            cmd = command(job, args)
            atomic_json(folder/'command.json', cmd)
            atomic_json(folder/'status.json', {'status': 'running'})
            work = args.work_root/job['dataset']
            before = {path: path.stat().st_mtime_ns for path in work.glob('*.metrics.json')}
            try:
                with (folder/'run.log').open('w') as log:
                    completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               env=dict(os.environ, PYTHONUNBUFFERED='1'))
            except KeyboardInterrupt:
                atomic_json(folder/'status.json', {'status': 'interrupted'})
                raise
            fresh = [path for path in work.glob('*.metrics.json') if before.get(path) != path.stat().st_mtime_ns]
            result = json.loads(fresh[0].read_text()) if not completed.returncode and len(fresh) == 1 else None
            if result is None or not matches(job, result):
                failures += 1
                atomic_json(folder/'status.json', {'status': 'failed', 'returncode': completed.returncode})
                print(f'  FALHOU: {folder / "run.log"}', flush=True)
                summarize(args.output_root, plan)
                if args.fail_fast:
                    break
                continue
            shutil.copy2(result['checkpoint'], folder/'checkpoint.pt')
            result['checkpoint'] = str(folder/'checkpoint.pt')
            atomic_json(result_path, result)
            atomic_json(folder/'status.json', {'status': 'complete'})
            print(f"  teste={100*result['test']['accuracy']:.2f}% | F1={100*result['test']['macro_f1']:.2f}% | "
                  f"BA={100*result['test']['balanced_accuracy']:.2f}%", flush=True)
            summarize(args.output_root, plan)
    except KeyboardInterrupt:
        interrupted = True
        print('Interrompido; resultados concluidos preservados.', flush=True)
    finally:
        summarize(args.output_root, plan)
    return 130 if interrupted else 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
