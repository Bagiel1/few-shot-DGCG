import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import experimentos_metricas_graus as sweep
from proto_sgc.config import parse_args, validate_hyperparameters
from proto_sgc.episodes import FeatureEpisodeSampler
from proto_sgc.metrics import classification_scores, GraphStatistics, record_graphs
from proto_sgc.model import ProtoSGC
from proto_sgc.training import evaluate, train_model


class MetricsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def model(self, graph='dgcg', name='proto-sgc'):
        return ProtoSGC(8,8,name,5,graph,2,.2)

    def sampler(self):
        gen=torch.Generator().manual_seed(33)
        return FeatureEpisodeSampler({str(i):torch.randn(12,8,generator=gen) for i in range(6)},2,1,2)

    def test_scores_hand_calculated_and_missing_prediction(self):
        # confusion [[2,0],[1,1]]: F1=(.8+2/3)/2; BA=.75.
        f1,ba=classification_scores(torch.tensor([0,0,1,1]),torch.tensor([0,0,0,1]),2)
        self.assertAlmostEqual(f1,(.8+2/3)/2)
        self.assertEqual(ba,.75)
        f1,ba=classification_scores(torch.tensor([0,0,1,1]),torch.tensor([0,0,0,0]),2)
        self.assertAlmostEqual(f1,1/3)
        self.assertEqual(ba,.5)

    def test_unbalanced_queries_distinguish_balanced_accuracy(self):
        # Accuracy 75%, balanced accuracy 50%.
        _,ba=classification_scores(torch.tensor([0,0,0,1]),torch.tensor([0,0,0,0]),2)
        self.assertEqual(ba,.5)

    def test_weighted_degrees_ignore_loops_and_weights(self):
        stats=GraphStatistics()
        stats.update(torch.tensor([[10.,.1,0.],[0.,10.,.9],[0.,0.,10.]]))
        result=stats.summary()
        self.assertAlmostEqual(result['degree_mean'],2/3)
        self.assertEqual(result['degree_max'],1)
        self.assertEqual(result['density_mean'],1/3)
        self.assertEqual(result['isolated_node_fraction'],0.)
        self.assertEqual(result['zero_row_fraction'],1/3)

    def test_mutual_is_reciprocal_not_union(self):
        x=torch.tensor([[0.],[1.],[3.],[8.]])
        from proto_sgc.graphs.knn import knn_adjacency
        mutual=knn_adjacency(x,knn=1,graph_type='knn-mutual')
        reciprocal=knn_adjacency(x,knn=1,graph_type='knn-reciprocal')
        union=knn_adjacency(x,knn=1,graph_type='knn-union')
        torch.testing.assert_close(mutual,reciprocal)
        torch.testing.assert_close(mutual,mutual.T)
        self.assertTrue(bool((union>=mutual).all()))
        self.assertGreater(union.sum(),mutual.sum())

    def test_observer_preserves_logits_and_is_restored(self):
        model=self.model()
        features=torch.randn(6,8)
        original=model.encode_episode(features)
        with record_graphs(model) as stats:
            actual=model.encode_episode(features)
        torch.testing.assert_close(original,actual,rtol=0,atol=0)
        self.assertEqual(stats.summary()['episodes'],1)
        self.assertIsNone(model.graph_observer)
        with self.assertRaises(RuntimeError):
            with record_graphs(model):
                raise RuntimeError('test')
        self.assertIsNone(model.graph_observer)

    def test_evaluation_metrics_and_degree_clipping(self):
        sampler=self.sampler()
        model=self.model()
        result=evaluate(model,sampler,['0','1'],5,2,torch.device('cpu'),7)
        self.assertAlmostEqual(result.accuracy,result.balanced_accuracy,places=6)
        self.assertGreaterEqual(result.macro_f1,0)
        self.assertLessEqual(result.macro_f1,1)
        self.assertEqual(result.graph['episodes'],5)
        self.assertEqual(result.graph['nodes_per_episode'],6)
        self.assertEqual(result.graph['effective_degree_high'],5)
        self.assertTrue(result.graph['degree_target_clipped'])
        second=evaluate(model,sampler,['0','1'],5,2,torch.device('cpu'),7)
        self.assertEqual(asdict(result),asdict(second))

    def test_no_graph_has_zero_degree_and_no_cross_sample_influence(self):
        model=self.model(name='protonet')
        sampler=self.sampler()
        result=evaluate(model,sampler,['0','1'],2,2,torch.device('cpu'),7)
        self.assertEqual(result.graph['degree_mean'],0)
        self.assertEqual(result.graph['density_mean'],0)
        x=torch.randn(6,8)
        before=model.encode_episode(x)
        x[3:]+=100
        after=model.encode_episode(x)
        torch.testing.assert_close(before[:3],after[:3],rtol=0,atol=0)

    def test_real_training_six_methods_produces_structured_metrics(self):
        for method in sweep.METHODS:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as directory:
                argv=['test','--n-way','2','--n-shot','1','--n-query','2',
                      '--train-episodes','2','--val-every','1','--val-episodes','2',
                      '--test-episodes','3','--work-dir',directory]
                argv+=['--model','protonet'] if method=='no-graph' else ['--graph-type',method]
                with patch.object(sys,'argv',argv):args=parse_args()
                validate_hyperparameters(args)
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    _,metrics,path=train_model(self.model(args.graph_type,args.model),self.sampler(),
                        {'train':['0','1'],'val':['2','3'],'test':['4','5']},args,torch.device('cpu'))
                saved=json.loads(path.with_suffix('.metrics.json').read_text())
                self.assertEqual(saved['test'],asdict(metrics))
                self.assertEqual(saved['selection_metric'],'validation.accuracy')
                self.assertEqual(saved['validation']['graph']['episodes'],2)
                self.assertEqual(saved['test']['graph']['episodes'],3)


class SweepTests(unittest.TestCase):
    def test_default_count_and_no_extra_context(self):
        args=sweep.parser().parse_args([])
        plan=list(sweep.jobs(args))
        self.assertEqual(len(plan),2430)
        self.assertEqual(len({j['directory'] for j in plan}),2430)
        for job in plan[:9]:
            cmd=sweep.command(job,args)
            self.assertNotIn('--dgcg-extra-references',cmd)
            self.assertNotIn('--dgcg-train-global-rankings',cmd)
            self.assertIn('--class-split',cmd)

    def fixtures(self):
        args=sweep.parser().parse_args(['--datasets','flowers_micro','--ways','5',
             '--shots','1','--queries','5','--methods','dgcg','--degrees','2:4','4:6'])
        plan=list(sweep.jobs(args))
        return plan,[dict(j,val_accuracy=80-i,test_accuracy=70+i) for i,j in enumerate(plan)]

    def test_selects_validation_not_test_and_requires_full_grid(self):
        plan,rows=self.fixtures()
        selected=sweep.select_degrees(rows,plan)
        self.assertEqual(selected[0]['alvo'],'2:4')
        self.assertEqual(selected[0]['ganho_teste_vs_4_6_pp'],-1)
        self.assertEqual(sweep.select_degrees(rows[:1],plan),[])
        rows[1]['val_accuracy']=rows[0]['val_accuracy']
        selected=sweep.select_degrees(rows,plan)
        self.assertEqual(selected[0]['alvo'],'4:6')
        self.assertEqual(selected[0]['alvos_empatados'],'2:4;4:6')

    def test_variation_changes_one_axis(self):
        plan,rows=self.fixtures()
        first=rows[0]
        second=dict(first,n_shot=5,alvo='4:6')
        result=sweep.degree_variation([first,second])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['eixo_variado'],'n_shot')
        self.assertEqual(result[0]['amplitude_centro'],2)

    def test_invalid_degrees(self):
        for target in ['0','8:2','1:2:3','nan','inf']:
            with self.assertRaises(Exception):sweep.parse_degree(target)

    def test_plots_and_summary_smoke(self):
        plan,rows=self.fixtures()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for job in plan:
                job['directory']=str(root/job['alvo'].replace(':','-'))
                folder=Path(job['directory']);folder.mkdir()
                metrics=dict(accuracy=.8,ci95=.02,macro_f1=.79,macro_f1_ci95=.03,
                             balanced_accuracy=.8,balanced_accuracy_ci95=.02,loss=.5,
                             graph={'degree_mean':5.,'nodes_per_episode':30})
                (folder/'metrics.json').write_text(json.dumps(dict(validation=metrics,test=metrics)))
            with contextlib.redirect_stdout(io.StringIO()):sweep.summarize(root,plan)
            self.assertTrue((root/'resultados.csv').is_file())
            self.assertIn('macro_f1',(root/'resultados.csv').read_text())
            self.assertIn('4:6',(root/'melhores_graus_por_protocolo.csv').read_text())
            self.assertEqual(len(list((root/'graficos').glob('*.png'))),5)


if __name__=='__main__':unittest.main()
