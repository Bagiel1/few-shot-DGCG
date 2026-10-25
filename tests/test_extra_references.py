import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from proto_sgc.config import parse_args, validate_hyperparameters
from proto_sgc.episodes import FeatureEpisodeSampler
from proto_sgc.graphs.dgcg import dgcg_correlation_matrix
from proto_sgc.graphs.training_rankings import build_reference_ranked_lists
from proto_sgc.model import ProtoSGC
from proto_sgc.reference_bank import ExtraReferenceBank
from proto_sgc.training import evaluate, train_model


class ExtraReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator=torch.Generator().manual_seed(27)
        self.features={str(i):torch.randn(40,8,generator=generator) for i in range(6)}
        self.sampler=FeatureEpisodeSampler(self.features,2,1,2)
        self.classes=['2','3']

    def bank(self,count):
        return ExtraReferenceBank(self.sampler,self.classes,count=count,per_class=20,seed=77)

    def model(self):
        return ProtoSGC(8,8,'proto-sgc',5,'dgcg',2,.2)

    def test_reserved_bank_disjoint_from_every_episode(self):
        bank=self.bank(10)
        rng=np.random.default_rng(7)
        for _ in range(30):
            episode=bank.sample(rng)
            self.assertEqual(len(episode.support_x)+len(episode.query_x),6)
            for label,index in episode.sample_ids:
                original=bank.episode_ids[label][index]
                self.assertNotIn(original,bank.bank_ids[label])
            extras=bank.extra_features(episode)
            self.assertEqual(extras.shape,(10,8))
        self.assertEqual(set(bank.bank),set(self.classes))

    def test_count_changes_only_extras_not_episode_or_bank(self):
        banks=[self.bank(count) for count in [0,1,10,25,40]]
        episodes=[bank.sample(np.random.default_rng(7)) for bank in banks]
        for bank,episode in zip(banks,episodes):
            self.assertEqual(bank.metadata()['bank_ids'],banks[0].metadata()['bank_ids'])
            self.assertEqual(episode.sample_ids,episodes[0].sample_ids)
            torch.testing.assert_close(episode.support_x,episodes[0].support_x)
            torch.testing.assert_close(episode.query_x,episodes[0].query_x)
            self.assertEqual(len(bank.extra_features(episode)),bank.count)

    def test_only_selected_classes_supply_extras(self):
        bank=ExtraReferenceBank(self.sampler,['2','3','4','5'],count=5,per_class=20,seed=77)
        episode=bank.sample(np.random.default_rng(7))
        labels=list(dict.fromkeys(label for label,_ in episode.sample_ids))
        expected=torch.cat((bank.bank[labels[0]][:3],bank.bank[labels[1]][:2]))
        torch.testing.assert_close(bank.extra_features(episode),expected)

    def test_no_propagation_through_extra_nodes(self):
        model=self.model()
        bank=self.bank(10)
        sizes=[]
        original=model.graph_adjacency
        def track(features,**options):
            self.assertEqual(features.shape,(6,8))
            self.assertEqual(options['dgcg_correlations'].shape,(6,6))
            adjacency=original(features,**options)
            sizes.append(adjacency.shape)
            return adjacency
        with patch.object(model,'graph_adjacency',side_effect=track):
            result=evaluate(model,self.sampler,self.classes,3,2,torch.device('cpu'),7,extra_reference_bank=bank)
        self.assertTrue(np.isfinite(result.loss))
        self.assertEqual(sizes,[torch.Size([6,6])]*3)

    def test_rankings_reference_union_and_label_free_correlations(self):
        bank=self.bank(10)
        episode=bank.sample(np.random.default_rng(7))
        features=torch.cat((episode.support_x,episode.query_x))
        extra=bank.extra_features(episode)
        rankings=build_reference_ranked_lists(features,extra,list_size=200,metric='euclidean')
        union=torch.cat((extra,features))
        distances=torch.cdist(union,union)
        distances.fill_diagonal_(-torch.inf)
        torch.testing.assert_close(rankings,distances.argsort(stable=True)[10:])
        self.assertEqual(rankings.shape,(6,16))
        corr=dgcg_correlation_matrix(rankings,correlation='rbo',rbo_p=.9,top_k=40,dtype=torch.float32)
        episode.query_y.fill_(99)
        again=build_reference_ranked_lists(features,extra,list_size=200,metric='euclidean')
        torch.testing.assert_close(corr,dgcg_correlation_matrix(again,correlation='rbo',rbo_p=.9,top_k=40,dtype=torch.float32))

    def test_capacity_and_partition_checks(self):
        with self.assertRaisesRegex(ValueError,'banco'):
            ExtraReferenceBank(FeatureEpisodeSampler(self.features,2,10,15),self.classes,count=10,per_class=20,seed=77)
        with self.assertRaisesRegex(ValueError,'particao'):
            evaluate(self.model(),self.sampler,['4','5'],1,2,torch.device('cpu'),7,extra_reference_bank=self.bank(10))

    def test_cli_and_real_training(self):
        with patch.object(sys,'argv',['test']):
            self.assertIsNone(parse_args().dgcg_extra_references)
        for count in [0,10]:
            with tempfile.TemporaryDirectory() as folder:
                argv=['test','--graph-type','dgcg','--n-way','2','--n-shot','1','--n-query','2',
                      '--dgcg-extra-references',str(count),'--train-episodes','2',
                      '--val-every','1','--val-episodes','2','--test-episodes','2','--work-dir',folder]
                with patch.object(sys,'argv',argv):args=parse_args()
                validate_hyperparameters(args)
                _,metrics,path=train_model(self.model(),self.sampler,
                    {'train':['0','1'],'val':['2','3'],'test':['4','5']},args,torch.device('cpu'))
                self.assertIn(f'extra{count}-bank20',path.name)
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                banks=checkpoint['extra_reference_banks']
                self.assertEqual(set(banks['val']['bank_ids']),{'2','3'})
                self.assertEqual(set(banks['test']['bank_ids']),{'4','5'})
                self.assertTrue(np.isfinite(metrics.loss))
                args.dgcg_reference_train_rankings=True
                with self.assertRaises(ValueError):validate_hyperparameters(args)


if __name__=='__main__':unittest.main()
