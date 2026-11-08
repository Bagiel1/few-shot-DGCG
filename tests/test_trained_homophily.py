import argparse
import contextlib
from io import StringIO
import json
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import experimento_homofilia_treinado as run
from proto_sgc.episodes import FeatureEpisodeSampler, split_classes
from proto_sgc.homophily import edge_agreement, measure_homophily, pearson
from proto_sgc.model import ProtoSGC
from proto_sgc.training import train_model


class HomophilyTests(unittest.TestCase):
    def test_weighted_query_rows_exclude_diagonal_and_preserve_input(self):
        a=torch.tensor([[100.,2,3,0],[0,100,4,5],[7,0,100,11],[0,13,17,100]])
        labels=torch.tensor([0,1,0,1]);original=a.clone()
        counts=edge_agreement(a,labels,2)
        self.assertEqual(counts,dict(same_all=28.,weight_all=62.,same_query=20.,weight_query=48.))
        torch.testing.assert_close(a,original)

    def test_theta_cannot_change_topology_homophily_or_episode_pairing(self):
        sampler=FeatureEpisodeSampler({str(i):torch.randn(16,4) for i in range(5)},5,1,10)
        model=ProtoSGC(4,3,"proto-sgc",5,"knn-union",2,.2)
        before=measure_homophily(model,sampler,list(sampler.features_by_class),5,3,torch.device('cpu'),916)
        with torch.no_grad():model.theta.weight.uniform_(-10,10)
        state={k:v.clone() for k,v in model.state_dict().items()}
        after=measure_homophily(model,sampler,list(sampler.features_by_class),5,3,torch.device('cpu'),916)
        self.assertEqual(before,after)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k])

    def test_correlation_constant_is_undefined(self):
        self.assertAlmostEqual(pearson([1,2,3],[2,4,6]),1)
        self.assertIsNone(pearson([1,1],[2,3]))
        self.assertIsNone(pearson([],[]))

    def test_fixed_grid_and_checkpoint_compatibility(self):
        args=run.parser().parse_args([])
        jobs=run.plan(args,[])
        self.assertEqual(len(jobs),60)
        self.assertEqual(sum(j['method']!='no-graph' for j in jobs),40)
        a=run.core_args(args,'flowers_micro',1,7,'dgcg-4-6',Path('/tmp/test/flowers_micro'))
        saved=vars(a).copy()
        self.assertTrue(run.compatible(saved,a,'flowers_micro','flowers_micro/model.pt'))
        for key,value in [('backbone','vit-b-16'),('train_free',True),('extra_images',0),
                          ('feature_dim',512),('dgcg_target_density',[.125,.125]),
                          ('dgcg_target_degree',[2,4]),('device','cpu')]:
            self.assertFalse(run.compatible(dict(saved,**{key:value}),a,'flowers_micro','flowers_micro/model.pt'),key)
        self.assertFalse(run.compatible(saved,a,'cars_micro','flowers_micro/model.pt'))
        self.assertFalse(run.compatible(saved,a,'flowers_micro','flowers_micro_cl2n/model.pt'))

    def test_pdf_reference_reproduces_query_definition(self):
        ref=run.pdf_reference()
        self.assertEqual(ref['1']['n'],20)
        self.assertAlmostEqual(ref['1']['pearson_query'],.6472814894037344)
        self.assertAlmostEqual(ref['5']['pearson_query'],.8612571004983662)

    def test_dry_run_does_not_extract_train_or_write(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(StringIO()):
            out=Path(tmp)/'new'
            with patch.object(run,'extract_or_load_features',side_effect=AssertionError('extract')):
                self.assertEqual(run.main(['--dry-run','--output-root',str(out)]),0)
            self.assertFalse(out.exists())

    def test_trained_pattern_from_sources_requires_no_training_or_gpu(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(StringIO()):
            with patch.object(run,'train_model',side_effect=AssertionError('repeated training')), \
                 patch.object(run,'select_device',side_effect=AssertionError('unneeded GPU')), \
                 patch.object(run,'plot'):
                self.assertEqual(run.main(['--output-root',tmp]),0)
            with (Path(tmp)/'correlacoes.csv').open() as stream:
                corr={int(r['shot']):r for r in csv.DictReader(stream)}
            self.assertEqual(int(corr[1]['points']),20)
            self.assertAlmostEqual(float(corr[1]['pearson_query']),.6930406192060107)
            self.assertAlmostEqual(float(corr[5]['pearson_query']),.7627671434929334)
            with (Path(tmp)/'resultados.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),60)
            self.assertTrue(all(r['classification_reused']=='True' and r['device']=='cpu' for r in rows))

    def test_real_short_training_checkpoint_and_diagnostic(self):
        previous=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(StringIO()):
                args=run.parser().parse_args(['--device','cpu'])
                features={str(i):torch.randn(16,512) for i in range(20)}
                sampler=FeatureEpisodeSampler(features,5,1,10)
                splits=split_classes(sorted(features),5,[10,5,5],7)
                for method in run.METHODS:
                    a=run.core_args(args,'flowers_micro',1,7,method,Path(tmp)/method)
                    a.work_dir.mkdir();a.train_episodes=2;a.val_every=1;a.val_episodes=2;a.test_episodes=3
                    model,metrics,checkpoint=train_model(run.build_model(a),sampler,splits,a,torch.device('cpu'))
                    index=run.checkpoint_index([Path(tmp)/method])
                    self.assertEqual(len(index),1)
                    self.assertEqual(index[0][1],checkpoint)
                    self.assertTrue(run.compatible(index[0][2],a,'flowers_micro',checkpoint))
                    if method!='no-graph':
                        hom=measure_homophily(model,sampler,splits['test'],5,3,torch.device('cpu'),916)
                        self.assertTrue(0<=hom['homophily_query_percent']<=100)
        finally:torch.set_num_threads(previous)


if __name__=='__main__':unittest.main()
