"""Regression checks for train-only references and transductive evaluation."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from proto_sgc.config import parse_args, validate_hyperparameters
from proto_sgc.episodes import FeatureEpisodeSampler
from proto_sgc.graphs.dgcg import dgcg_correlation_matrix
from proto_sgc.graphs.training_rankings import TrainingRankings
from proto_sgc.model import ProtoSGC
from proto_sgc.training import train_model, evaluate


class ReferenceRankingsTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(17)
        self.features = {str(i): torch.randn(6, 8, generator=generator) for i in range(6)}
        self.splits = {'train': ['0', '1'], 'val': ['2', '3'], 'test': ['4', '5']}

    def oracle(self, universe, metric):
        if metric == 'cosine':
            unit = F.normalize(universe, dim=1)
            scores = unit @ unit.T
            scores.fill_diagonal_(torch.inf)
        else:
            scores = torch.cdist(universe, universe)
            scores.fill_diagonal_(-torch.inf)
        return scores.argsort(dim=1, descending=metric == 'cosine', stable=True)

    def test_rankings_and_correlations_match_full_union(self):
        for metric in ('euclidean', 'cosine'):
            for length in (5, 200):
                with self.subTest(metric=metric, length=length):
                    ref = TrainingRankings(self.features, self.splits['train'], list_size=length, metric=metric)
                    episode = torch.cat((self.features['4'][:2], self.features['5'][:2]))
                    union = torch.cat((ref.features, episode))
                    oracle = self.oracle(union, metric)[len(ref.features):, :min(length,len(union))]
                    actual = ref.episode_rankings(episode)
                    torch.testing.assert_close(actual, oracle)
                    self.assertEqual(actual.shape[0],len(episode))
                    torch.testing.assert_close(actual[:,0],torch.arange(12,16))
                    for correlation in ('rbo','jaccardk','jaccard-median','jaccard-max'):
                        torch.testing.assert_close(
                            ref.episode_correlations(episode,correlation=correlation,top_k=40,rbo_p=.9,dtype=torch.float32),
                            dgcg_correlation_matrix(oracle,correlation=correlation,top_k=40,rbo_p=.9,dtype=torch.float32),
                        )

    def test_train_universe_does_not_duplicate_episode_images(self):
        ref = TrainingRankings(self.features,self.splits['train'],list_size=200,metric='euclidean')
        self.assertEqual(ref.rankings.shape,(12,12))
        ids=[('0',1),('1',2)]
        expected=dgcg_correlation_matrix(self.oracle(ref.features,'euclidean')[[1,8]],correlation='rbo',top_k=40,rbo_p=.9,dtype=torch.float32)
        actual=ref.correlations(ids,correlation='rbo',top_k=40,rbo_p=.9,dtype=torch.float32,device=torch.device('cpu'))
        torch.testing.assert_close(actual,expected)

    def test_unrelated_heldout_images_never_enter_rankings(self):
        episode=self.features['4'][:3].clone()
        ref=TrainingRankings(self.features,self.splits['train'],list_size=200,metric='euclidean')
        before=ref.episode_rankings(episode)
        altered=dict(self.features)
        for label in ['2','3','4','5']:altered[label]=torch.full_like(self.features[label],1e8)
        other=TrainingRankings(altered,self.splits['train'],list_size=200,metric='euclidean')
        torch.testing.assert_close(before,other.episode_rankings(episode))
        ref.episode_rankings(self.features['2'])
        torch.testing.assert_close(before,ref.episode_rankings(episode))
        self.assertEqual(len(ref.features),12)

    def test_self_first_with_duplicate_vectors(self):
        ref=TrainingRankings({'t':torch.ones(3,8)},['t'],list_size=200,metric='cosine')
        torch.testing.assert_close(ref.episode_rankings(torch.ones(2,8))[:,0],torch.tensor([3,4]))

    def test_200_limit_searches_all_references(self):
        generator=torch.Generator().manual_seed(29)
        training=torch.randn(230,8,generator=generator)
        episode=torch.randn(4,8,generator=generator)
        ref=TrainingRankings({'t':training},['t'],list_size=200,metric='euclidean')
        self.assertEqual(ref.rankings.shape,(230,200))
        expected=self.oracle(torch.cat((training,episode)),'euclidean')[230:,:200]
        actual=ref.episode_rankings(episode)
        self.assertEqual(actual.shape,(4,200))
        torch.testing.assert_close(actual,expected)

    def args(self, directory, mode):
        argv=['test','--graph-type','dgcg','--n-way','2','--n-shot','1','--n-query','2',
              '--train-episodes','2','--val-every','1','--val-episodes','2','--test-episodes','2',
              '--work-dir',str(directory),'--device','cpu']
        if mode:argv.append(mode)
        with patch.object(sys,'argv',argv):return parse_args()

    def test_real_training_validation_test_and_legacy_modes(self):
        torch.set_num_threads(1)
        for flag,tag in [(None,''),('--dgcg-train-global-rankings','train-global-rankings'),
                         ('--dgcg-reference-train-rankings','reference-train-rankings')]:
            with self.subTest(flag=flag),tempfile.TemporaryDirectory() as directory:
                args=self.args(directory,flag)
                validate_hyperparameters(args)
                model=ProtoSGC(input_dim=8,output_dim=8,model_name='proto-sgc',graph_type='dgcg',knn=5,sgc_hops=2,graph_temperature=.2)
                sampler=FeatureEpisodeSampler(self.features,n_way=2,n_shot=1,n_query=2)
                with patch('proto_sgc.training.evaluate',wraps=evaluate) as evaluations:
                    _,metrics,path=train_model(model,sampler,self.splits,args,torch.device('cpu'))
                self.assertTrue(path.is_file())
                self.assertTrue(np.isfinite(metrics.loss))
                if tag:self.assertIn(tag,path.name)
                self.assertEqual(evaluations.call_count,3)
                for call in evaluations.call_args_list:
                    reference=call.kwargs['training_rankings']
                    if flag=='--dgcg-reference-train-rankings':
                        self.assertIsNotNone(reference)
                        self.assertEqual(set(label for label,_ in reference.row_by_id),{'0','1'})
                    else:self.assertIsNone(reference)

    def test_invalid_model_and_evaluation_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.args(directory,'--dgcg-reference-train-rankings')
            args.graph_type='knn-in'
            with self.assertRaises(ValueError):validate_hyperparameters(args)
        ref=TrainingRankings(self.features,self.splits['train'],list_size=200,metric='euclidean')
        model=ProtoSGC(input_dim=8,output_dim=8,model_name='proto-sgc',graph_type='dgcg',knn=5,sgc_hops=2,graph_temperature=.2)
        sampler=FeatureEpisodeSampler(self.features,2,1,2)
        with self.assertRaisesRegex(ValueError,'disjunto'):
            evaluate(model,sampler,self.splits['train'],1,2,torch.device('cpu'),7,training_rankings=ref)


if __name__=='__main__':unittest.main()
