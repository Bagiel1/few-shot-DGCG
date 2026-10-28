import copy
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import experimentos_extras as sweep
from proto_sgc.config import parse_args, validate_hyperparameters
from proto_sgc.episodes import FeatureEpisodeSampler, episode_to_device
from proto_sgc.model import ProtoSGC
from proto_sgc.runtime import seed_everything
from proto_sgc.training import extra_graph_options, prepare_extra_banks, train_model


def cli(*argv):
    with patch.object(sys, 'argv', ['test', *argv]):
        args = parse_args()
    validate_hyperparameters(args)
    return args


class ExtraContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        gen = torch.Generator().manual_seed(123)
        self.features = {str(i): torch.randn(40, 8, generator=gen) for i in range(6)}
        self.splits = {'train': ['0', '1'], 'val': ['2', '3'], 'test': ['4', '5']}
        self.sampler = FeatureEpisodeSampler(self.features, 2, 1, 2)

    def args(self, *argv):
        return cli('--graph-type', 'dgcg', '--n-way', '2', '--n-shot', '1', '--n-query', '2',
                   '--extra-images', '4', '--extra-bank-per-class', '3', '--train-episodes', '2',
                   '--val-episodes', '2', '--test-episodes', '2', '--val-every', '1', *argv)

    def model(self, graph='dgcg', name='proto-sgc'):
        return ProtoSGC(8, 8, name, 5, graph, 2, .2)

    def test_phase_masks_and_fixed_pools(self):
        baseline = prepare_extra_banks(self.sampler, self.splits, self.args('--extra-images', '0'))
        for phases in sweep.PHASES.values():
            with self.subTest(phases=phases):
                banks = prepare_extra_banks(self.sampler, self.splits, self.args('--extra-phases', *phases))
                for phase, bank in banks.items():
                    self.assertEqual(bank.count, 4 if phase in phases else 0)
                    self.assertEqual(bank.bank_ids, baseline[phase].bank_ids)
                    self.assertEqual(set(bank.classes), set(self.splits[phase]))
                    ep = bank.sample(np.random.default_rng(17))
                    old_ep = baseline[phase].sample(np.random.default_rng(17))
                    torch.testing.assert_close(ep.support_x, old_ep.support_x)
                    torch.testing.assert_close(ep.query_x, old_ep.query_x)
                    for label, idx in ep.sample_ids:
                        self.assertNotIn(bank.episode_ids[label][idx], bank.bank_ids[label])
                    self.assertEqual(len(bank.extra_features(ep)), bank.count)

    def test_same_rankings_different_node_sets_and_no_extra_labels(self):
        bank = prepare_extra_banks(self.sampler, self.splits, self.args())['train']
        ep = bank.sample(np.random.default_rng(21))
        model = self.model()
        refs = extra_graph_options(model, ep, bank, 'rankings')
        prop = extra_graph_options(model, ep, bank, 'propagation')
        self.assertEqual(refs['dgcg_correlations'].shape, (6, 6))
        self.assertEqual(prop['extra_x'].shape, (4, 8))
        # Mesmo universo de imagens, com diferenca apenas na ordenacao dos IDs.
        from proto_sgc.graphs.dgcg import build_ranked_lists, dgcg_correlation_matrix
        union = torch.cat((ep.support_x, ep.query_x, prop['extra_x']))
        ranks = build_ranked_lists(union, len(union), metric='euclidean')
        corr = dgcg_correlation_matrix(ranks, correlation='rbo', top_k=40, rbo_p=.9, dtype=torch.float32)
        torch.testing.assert_close(refs['dgcg_correlations'], corr[:6, :6])
        for mode, options, nodes in [('rankings', refs, 6), ('propagation', prop, 10)]:
            sizes = []
            model.graph_observer = lambda graph: sizes.append(graph.shape)
            logits = model(ep.support_x, ep.support_y, ep.query_x, 2, **options)
            self.assertEqual(logits.shape, (4, 2))
            self.assertEqual(sizes, [torch.Size([nodes, nodes])])
            ep.query_y.fill_(99)
            again = model(ep.support_x, ep.support_y, ep.query_x, 2, **options)
            torch.testing.assert_close(logits, again)

    def test_propagation_actually_uses_extra_features_and_gradients(self):
        model = self.model('knn-union')
        ep = self.sampler.sample(self.splits['train'], np.random.default_rng(7))
        extras = torch.randn(4, 8, requires_grad=True)
        # Grafo controlado: queries recebem mistura distinta de atributos extras.
        adjacency = torch.eye(10)
        adjacency[2:6, 6:10] = .4 * torch.eye(4)
        with patch.object(model, 'normalized_graph_adjacency', return_value=adjacency):
            logits = model(ep.support_x, ep.support_y, ep.query_x, 2, extra_x=extras)
            loss = torch.nn.functional.cross_entropy(logits, ep.query_y)
            loss.backward()
            self.assertGreater(extras.grad.abs().sum().item(), 0)
            changed = model(ep.support_x, ep.support_y, ep.query_x, 2, extra_x=extras.detach()+2)
            self.assertFalse(torch.allclose(logits, changed))

    def test_empty_extras_preserve_original_forward(self):
        ep = self.sampler.sample(self.splits['train'], np.random.default_rng(7))
        for graph in ['knn-in', 'knn-out', 'knn-mutual', 'knn-union', 'dgcg']:
            model = self.model(graph)
            original = model(ep.support_x, ep.support_y, ep.query_x, 2)
            actual = model(ep.support_x, ep.support_y, ep.query_x, 2, extra_x=torch.empty(0, 8))
            torch.testing.assert_close(original, actual, rtol=0, atol=0)

    def test_validation_rejects_incompatible_configs(self):
        for argv in [('--extra-images', '-1'), ('--extra-images', '7'),
                     ('--dgcg-extra-references', '0'), ('--dgcg-train-global-rankings',),
                     ('--dgcg-reference-train-rankings',), ('--extra-phases', 'train', 'train'),
                     ('--graph-type', 'knn-in'), ('--model', 'protonet')]:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                self.args(*argv)
        self.args('--graph-type', 'knn-in', '--extra-mode', 'propagation')
        self.args('--model', 'protonet', '--extra-images', '0')

    def test_training_and_metrics_for_every_mode_and_phase(self):
        for mode in ['rankings', 'propagation']:
            for phases in sweep.PHASES.values():
                with self.subTest(mode=mode, phases=phases), tempfile.TemporaryDirectory() as directory:
                    args = self.args('--extra-mode', mode, '--extra-phases', *phases, '--work-dir', directory)
                    model = self.model()
                    train_sizes = []
                    original = model.forward
                    def track(*a, **kw):
                        if model.training:
                            train_sizes.append(len(kw.get('extra_x', [])))
                            if mode == 'rankings':
                                self.assertEqual('dgcg_correlations' in kw, 'train' in phases)
                        return original(*a, **kw)
                    with patch.object(model, 'forward', side_effect=track), redirect_stdout(io.StringIO()):
                        _, metrics, checkpoint = train_model(model, self.sampler, self.splits, args, torch.device('cpu'))
                    self.assertEqual(train_sizes, [4 if mode == 'propagation' and 'train' in phases else 0]*2)
                    result = json.loads(checkpoint.with_suffix('.metrics.json').read_text())
                    self.assertEqual(metrics.graph['nodes_per_episode'], 10 if mode == 'propagation' and 'test' in phases else 6)
                    self.assertEqual(result['validation']['graph']['nodes_per_episode'],
                                     10 if mode == 'propagation' and 'val' in phases else 6)
                    self.assertEqual(set(result['extra_context']['banks']), {'train', 'val', 'test'})
                    self.assertTrue(np.isfinite(metrics.loss))

    def test_test_only_does_not_change_selected_weights(self):
        states = []
        for count in (0, 4):
            with tempfile.TemporaryDirectory() as directory:
                args = self.args('--extra-mode', 'propagation', '--extra-phases', 'test',
                                 '--extra-images', str(count), '--work-dir', directory)
                seed_everything(7)
                model = self.model()
                with redirect_stdout(io.StringIO()):
                    train_model(model, self.sampler, self.splits, args, torch.device('cpu'))
                states.append(copy.deepcopy(model.state_dict()))
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


class ExtraSweepTests(unittest.TestCase):
    def args(self, *argv):
        args = sweep.parser().parse_args(list(argv))
        args.work_root = args.work_root or args.output_root/'_work'
        return args

    def test_default_plan_and_valid_commands(self):
        args = self.args()
        jobs = list(sweep.jobs(args))
        self.assertEqual(len(jobs), 470)
        self.assertEqual(len({j['directory'] for j in jobs}), 470)
        for job in jobs[:47]:
            cmd = sweep.command(job, args)
            cli(*cmd[2:])
            self.assertEqual(job['extras_total'], job['extras_per_class']*2)
        self.assertEqual(sum(j['extra_mode'] == 'none' for j in jobs), 70)
        self.assertEqual(sum(j['extra_mode'] == 'rankings' for j in jobs), 100)
        self.assertEqual(sum(j['extra_mode'] == 'propagation' for j in jobs), 300)

    def test_multiple_counts_no_duplicate_controls(self):
        args = self.args('--extras-per-class', '5', '10', '20')
        jobs = list(sweep.jobs(args))
        self.assertEqual(len(jobs), 1270)
        self.assertEqual(sum(j['extra_mode'] == 'none' for j in jobs), 70)

    def test_impossible_capacity_rejected_before_running(self):
        for argv in [('--n-shot', '21'), ('--extras-per-class', '21'), ('--n-way', '6')]:
            with self.assertRaises(SystemExit):
                sweep.main(['--dry-run', *argv])

    def test_dry_run_never_creates_output(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            out = Path(directory)/'new'
            self.assertEqual(sweep.main(['--dry-run', '--output-root', str(out)]), 0)
            self.assertFalse(out.exists())

    def test_runner_failure_resume_and_summary(self):
        from experimentos_metricas_graus import METRICS
        calls = []
        fail_once = [True]

        def run(cmd, **kwargs):
            args = cli(*cmd[2:])
            calls.append(args.extra_images)
            if args.extra_images and fail_once[0]:
                fail_once[0] = False
                return SimpleNamespace(returncode=1)
            args.work_dir.mkdir(parents=True, exist_ok=True)
            path = args.work_dir/f'{args.model}-{args.graph_type}-{args.extra_images}.pt'
            path.write_bytes(b'fixture')
            scores = {metric: .5 for metric in METRICS}
            scores['graph'] = {'nodes_per_episode': 40, 'episodes': 1000}
            result = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      'checkpoint': str(path), 'validation': scores, 'test': scores}
            sweep.atomic_json(path.with_suffix('.metrics.json'), result)
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            argv = ['--output-root', directory, '--datasets', 'bacteria_micro',
                    '--methods', 'no-graph', 'knn-in', '--phases', 'test', '--modes', 'propagation']
            with patch.object(sweep.subprocess, 'run', side_effect=run):
                self.assertEqual(sweep.main(argv), 1)
                self.assertEqual(len(calls), 3)
                with (Path(directory)/'status.csv').open() as stream:
                    statuses = list(csv.DictReader(stream))
                self.assertEqual([r['status'] for r in statuses], ['complete', 'complete', 'failed'])
                self.assertEqual(sweep.main(argv), 0)
                self.assertEqual(len(calls), 4)  # Retenta somente o que falhou.
                self.assertEqual(sweep.main(argv), 0)
                self.assertEqual(len(calls), 4)
            self.assertEqual(sweep.main(['--output-root', directory, '--summarize-only']), 0)
            with (Path(directory)/'resultados.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual([r['ganho_accuracy_vs_sem_extras_pp'] for r in rows], ['0.0']*3)
            for row in json.loads((Path(directory)/'manifest.json').read_text())['runs']:
                self.assertTrue((Path(row['directory'])/'checkpoint.pt').exists())
            with self.assertRaises(SystemExit):
                sweep.main([*argv, '--seeds', '17'])


if __name__ == '__main__':
    unittest.main()
