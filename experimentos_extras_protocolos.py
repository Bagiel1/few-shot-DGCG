"""Grade de protocolos e quantidades de extras, com resultados separados por protocolo."""
import itertools
import json
import hashlib
from pathlib import Path
import statistics
import sys

import experimentos_extras as base


def parser():
    p = base.parser()
    p.description = __doc__
    p.add_argument('--ways', nargs='+', type=int, default=[2, 5])
    p.add_argument('--shots', nargs='+', type=int, default=[1, 5, 10])
    p.add_argument('--queries', nargs='+', type=int, default=[10])
    p.set_defaults(extras_per_class=[5, 10, 20], phases=['train', 'train-val', 'all'],
                   output_root=Path('runs/busca_extras_protocolos'), fail_fast=True)
    return p


def child_args(args, way, shot, query):
    folder = args.output_root / f'{way}way_{shot}shot_{query}query'
    argv = ['--output-root', str(folder), '--work-root', str(args.work_root),
            '--cache-dir', str(args.cache_dir), '--device', args.device,
            '--n-way', str(way), '--n-shot', str(shot), '--n-query', str(query),
            '--bank-per-class', str(args.bank_per_class)]
    for key in ('datasets', 'methods', 'extras_per_class', 'phases', 'modes', 'seeds'):
        argv += ['--'+key.replace('_', '-'), *map(str, getattr(args, key))]
    argv += ['--degrees', *[base.degree_name(d) for d in args.degrees]]
    for key in ('quick', 'fail_fast', 'dry_run'):
        if getattr(args, key): argv.append('--'+key.replace('_', '-'))
    return argv


def summarize(root, plan):
    rows, statuses = [], []
    for job in plan:
        folder = Path(job['directory'])
        state = 'pending'
        if (folder/'metrics.json').exists():
            result = json.loads((folder/'metrics.json').read_text())
            if not base.matches(job, result):
                raise ValueError(f'Resultado incompativel: {folder}')
            rows.append(base.flatten(job, result))
            state = 'complete'
        elif (folder/'status.json').exists():
            state = json.loads((folder/'status.json').read_text())['status']
        statuses.append(dict(job, status=state, log=str(folder/'run.log')))
    pair = lambda r: tuple(r[k] for k in ('dataset','seed','metodo','alvo','n_way','n_shot','n_query'))
    controls = {pair(r): r for r in rows if r['extra_mode']=='none'}
    for row in rows:
        control = controls.get(pair(row))
        for metric in ('accuracy','macro_f1','balanced_accuracy'):
            row[f'ganho_{metric}_vs_sem_extras_pp'] = (
                row['test_'+metric]-control['test_'+metric] if control else '')
    base.save_csv(root/'resultados.csv', rows)
    base.save_csv(root/'status.csv', statuses)
    fields = ('n_way', 'n_shot', 'n_query', 'metodo', 'alvo', 'extra_mode',
              'extra_phase', 'extras_per_class')
    groups, planned = {}, {}
    for job in plan:
        key = tuple(str(job[k]) for k in fields)
        planned[key] = planned.get(key, 0) + 1
    for row in rows:
        groups.setdefault(tuple(str(row[k]) for k in fields), []).append(row)
    means = []
    for key, group in groups.items():
        row = dict(zip(fields, key), concluidos=len(group), planejados=planned[key])
        for metric in ('val_accuracy', 'test_accuracy', 'test_macro_f1',
                       'test_balanced_accuracy', 'ganho_accuracy_vs_sem_extras_pp'):
            values = [float(r[metric]) for r in group if r[metric] != '']
            row[metric] = statistics.mean(values) if values else ''
        means.append(row)
    base.save_csv(root/'medias.csv', means)
    print(f'Grade consolidada: {len(rows)}/{len(plan)} em {root}', flush=True)


def main(argv=None):
    p = parser()
    raw = sys.argv[1:] if argv is None else argv
    if any(a.split('=')[0] in ('--n-way', '--n-shot', '--n-query') for a in raw):
        p.error('Nesta grade use --ways, --shots e --queries (listas).')
    args = p.parse_args(raw)
    args.output_root = args.output_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.work_root = (args.work_root or args.output_root/'_work').expanduser().resolve()
    manifest = args.output_root/'manifest.json'
    if args.summarize_only:
        if not manifest.exists(): p.error('Manifesto nao encontrado.')
        summarize(args.output_root, json.loads(manifest.read_text())['runs'])
        return 0
    for field in ('ways', 'shots', 'queries', 'extras_per_class', 'datasets', 'methods',
                  'degrees', 'phases', 'modes', 'seeds'):
        values = getattr(args, field)
        if len(set(values)) != len(values): p.error(f'Valores repetidos: {field}.')
    if not all(2 <= w <= 5 for w in args.ways): p.error('Ways devem estar entre 2 e 5.')
    if min(*args.shots, *args.queries, *args.extras_per_class, args.bank_per_class) < 1:
        p.error('Shots, queries, extras e banco devem ser positivos; controle zero e automatico.')
    if max(args.shots)+max(args.queries)+args.bank_per_class > 40:
        p.error('Micro: shots + queries + banco devem ser <= 40 em todos os protocolos.')
    if max(args.extras_per_class) > args.bank_per_class or min(args.seeds) < 0:
        p.error('Extras nao podem exceder o banco; seeds devem ser nao negativas.')
    protocols = list(itertools.product(args.ways, args.shots, args.queries))
    commands = [child_args(args, *protocol) for protocol in protocols]
    plan = [job for command in commands for job in base.jobs(base.parser().parse_args(command))]
    print(f'GRADE: {len(plan)} execucoes | {len(protocols)} protocolos | {len(args.datasets)} bases', flush=True)
    print(f'Extras/classe: {args.extras_per_class}; fases: {args.phases}; seeds: {args.seeds}', flush=True)
    if args.dry_run:
        for protocol, command in zip(protocols, commands):
            print(f'Protocolo {protocol}: {len(list(base.jobs(base.parser().parse_args(command))))} execucoes')
        return 0
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
              if k not in ('dry_run','summarize_only','fail_fast')}
    config['code_sha256'] = base.source_digest()
    config['grid_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = json.loads(json.dumps(config))
    args.output_root.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        if json.loads(manifest.read_text())['configuration'] != config:
            p.error('Pasta pertence a outra configuracao/versao. Escolha outro --output-root.')
    else:
        base.atomic_json(manifest, {'configuration':config, 'runs':plan,
                                   'selection':'Factorial design, not a ranking of task difficulty.'})
    # Fail before hundreds of subprocess launches if the requested device is unavailable.
    from proto_sgc.runtime import select_device
    print(f'Dispositivo: {select_device(args.device)}', flush=True)
    result = 0
    try:
        for protocol, command in zip(protocols, commands):
            print(f'Iniciando protocolo: {protocol}', flush=True)
            result = base.main(command)
            summarize(args.output_root, plan)
            if result: break
    except KeyboardInterrupt:
        result = 130
    finally:
        summarize(args.output_root, plan)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
