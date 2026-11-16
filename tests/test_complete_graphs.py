import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import experimentos_grafos_completos as battery
from proto_sgc.config import GRAPH_TYPES, parse_args, validate_hyperparameters
from proto_sgc.episodes import FeatureEpisodeSampler
from proto_sgc.graphs.dgcg import build_ranked_lists, dgcg_correlation_matrix
from proto_sgc.model import ProtoSGC
from proto_sgc.reference_bank import SplitPoolBank
from proto_sgc.training import prepare_extra_banks, train_model


class CompleteGraphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def model(self, graph, sigma=32.):
        return ProtoSGC(4, 4, "proto-sgc", 5, graph, 2, sigma)

    def test_rbf_is_raw_euclidean_kernel_before_degree_normalization(self):
        x = torch.tensor([[0., 0., 0., 0.], [3., 4., 0., 0.], [6., 8., 0., 0.]])
        model = self.model("complete-rbf", sigma=5)
        raw = model.graph_adjacency(x)
        self.assertAlmostEqual(float(raw[0, 1]), math.exp(-.5), places=6)
        self.assertAlmostEqual(float(raw[0, 2]), math.exp(-2), places=6)
        self.assertTrue(torch.equal(raw, raw.T))
        self.assertEqual(int(raw.ne(0).sum()), 6)
        with_loops = raw + torch.eye(3)
        factor = with_loops.sum(1).rsqrt()
        expected = factor[:, None] * with_loops * factor[None, :]
        torch.testing.assert_close(model.normalized_graph_adjacency(x), expected)
        self.assertNotIn("complete-rbf", GRAPH_TYPES)  # --graph-type all remains historical.

    def test_rbo_matches_existing_pairwise_scores_without_threshold(self):
        x = torch.randn(10, 4, generator=torch.Generator().manual_seed(7))
        ranks = build_ranked_lists(x, 10, metric="euclidean")
        expected = dgcg_correlation_matrix(ranks, correlation="rbo", rbo_p=.9, top_k=40, dtype=x.dtype)
        expected.fill_diagonal_(0)
        torch.testing.assert_close(self.model("complete-rbo").graph_adjacency(x), expected)

    def test_episode_median_uses_unique_pairs_and_changes_with_extra_nodes(self):
        model = ProtoSGC(4,4,"proto-sgc",5,"complete-rbf",2,32.,rbf_sigma_mode="episode-median")
        x = torch.tensor([[0.,0.,0.,0.],[3.,4.,0.,0.],[6.,8.,0.,0.]])
        raw = model.graph_adjacency(x)
        self.assertEqual(float(model.last_rbf_sigma),5.)  # Distances 5, 5, 10.
        self.assertAlmostEqual(float(raw[0,1]),math.exp(-.5),places=6)
        extended = torch.cat((x,torch.tensor([[60.,80.,0.,0.]])))
        model.graph_adjacency(extended)
        self.assertEqual(float(model.last_rbf_sigma),50.)  # Median of 5,5,10,90,95,100.
        scaled = model.graph_adjacency(10*x)
        torch.testing.assert_close(scaled,raw)
        zero = model.graph_adjacency(torch.zeros(4,4))
        self.assertTrue(torch.isfinite(zero).all())
        torch.testing.assert_close(zero,torch.ones(4,4)-torch.eye(4))

    def test_extras_are_disjoint_nested_and_do_not_change_evaluated_images(self):
        features = {str(k): torch.arange(k*80, (k+1)*80).reshape(20,4).float() for k in range(8)}
        sampler = FeatureEpisodeSampler(features, 2, 5, 5)
        classes = sorted(features)
        small = SplitPoolBank(sampler, classes, count=10, seed=4108)
        large = SplitPoolBank(sampler, classes, count=100, seed=4108)
        rng_small, rng_large, rng_plain = (np.random.default_rng(108) for _ in range(3))
        for _ in range(3):
            a,b,plain = small.sample(rng_small),large.sample(rng_large),sampler.sample(classes,rng_plain)
            self.assertEqual(a.sample_ids,b.sample_ids)
            self.assertEqual(a.sample_ids,plain.sample_ids)
            ax,bx = small.extra_features(a),large.extra_features(b)
            torch.testing.assert_close(ax,bx[:10])
            self.assertFalse(set(large.last_extra_ids)&set(a.sample_ids))
            self.assertEqual(len(set(large.last_extra_ids)),100)
            episode_classes = {c for c,_ in a.sample_ids}
            self.assertTrue(all(c in episode_classes for c,_ in small.last_extra_ids))
            self.assertTrue(all(c in episode_classes for c,_ in large.last_extra_ids[:20]))
            self.assertTrue(all(c not in episode_classes for c,_ in large.last_extra_ids[20:]))
            self.assertTrue(set(c for c,_ in large.last_extra_ids)-set(c for c,_ in a.sample_ids))
        self.assertEqual(small.metadata()["episode_ids_sha256"],large.metadata()["episode_ids_sha256"])
        with self.assertRaises(ValueError):
            SplitPoolBank(sampler,classes,count=141,seed=7)

    def test_defaults_and_new_mode_validation(self):
        args = battery.parser().parse_args([])
        jobs = list(battery.jobs(args))
        self.assertEqual(len(jobs),2640)
        self.assertEqual(len({j['directory'] for j in jobs}),2640)
        controls = [j for j in jobs if j['metodo']=='no-graph']
        self.assertEqual(len(controls),80)
        self.assertTrue(all(j['extras_total']==0 for j in controls))
        self.assertEqual(len({(j['dataset'],j['protocol']) for j in controls}),80)
        config = battery.arguments(args,controls[0],[10,5,5])
        self.assertEqual(config.model,'protonet')
        self.assertEqual(config.extra_images,0)
        rbf = next(j for j in jobs if j['metodo']=='complete-rbf')
        self.assertEqual(rbf['rbf_sigma_mode'],'episode-median')
        self.assertIsNone(rbf['rbf_sigma'])
        self.assertEqual(battery.arguments(args,rbf,[10,5,5]).rbf_sigma_mode,'episode-median')
        self.assertEqual({(j['n_way'],j['n_shot'],j['n_query']) for j in jobs},set(battery.PROTOCOLS.values()))
        self.assertEqual(len({(j['version'],j['protocol'],j['metodo']) for j in jobs if j['metodo']!='no-graph'}),16)
        self.assertEqual({j['openml_id'] for j in jobs if j['version']=='mini'},set(range(44281,44291)))
        core = parse_args(['--extra-images','200','--extra-source','split-pool','--extra-mode','propagation',
                           '--graph-type','complete-rbf','--graph-temperature','32'])
        validate_hyperparameters(core)
        core.extra_source = 'reserved'
        with self.assertRaises(ValueError):
            validate_hyperparameters(core)

    def test_normal_training_entry_point_uses_split_pool_and_saves_provenance(self):
        features = {str(i): torch.randn(8,4) for i in range(6)}
        sampler = FeatureEpisodeSampler(features,2,1,1)
        splits = dict(train=['0','1'],val=['2','3'],test=['4','5'])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for graph in battery.GRAPHS:
                args = parse_args(['--model','proto-sgc','--graph-type',graph,'--feature-dim','4',
                                   '--n-way','2','--n-shot','1','--n-query','1','--extra-images','3',
                                   '--extra-source','split-pool','--extra-mode','propagation',
                                   '--work-dir',tmp,'--graph-temperature','32','--train-episodes','2',
                                   '--val-every','1','--val-episodes','2','--test-episodes','3'])
                validate_hyperparameters(args)
                banks = prepare_extra_banks(sampler,splits,args)
                self.assertTrue(all(set(bank.classes)==set(splits[phase]) for phase,bank in banks.items()))
                model, metrics, path = train_model(self.model(graph),sampler,splits,args,torch.device('cpu'))
                self.assertTrue(math.isfinite(metrics.loss))
                self.assertEqual(metrics.graph['nodes_per_episode'],7)
                import json
                saved = json.loads(path.with_suffix('.metrics.json').read_text())
                self.assertEqual(saved['extra_context']['source'],'split-pool')
                self.assertFalse(saved['extra_context']['labels_exposed_to_model'])
                self.assertTrue(all(b['count']==3 for b in saved['extra_context']['banks'].values()))


if __name__ == '__main__':
    unittest.main()
