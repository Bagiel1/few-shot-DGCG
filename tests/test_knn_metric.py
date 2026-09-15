"""Confere seleção de vizinhos e integração da métrica kNN."""

import unittest
from unittest.mock import patch

import torch

from proto_sgc.config import parse_args
from proto_sgc.graphs.knn import knn_adjacency
from simulador.engine import SimulationConfig, model_arguments, parse_config, simulate
from proto_sgc.model import ProtoSGC


class KnnMetricTests(unittest.TestCase):
    def test_neighbor_choice_and_topologies(self):
        # Para [1,0], [1,1] está mais perto; [10,0] tem a mesma direção.
        x = torch.tensor([[1., 0.], [10., 0.], [1., 1.]])
        euclidean = knn_adjacency(x, knn=1, graph_type="knn-out")
        cosine = knn_adjacency(x, knn=1, graph_type="knn-out", metric="cosine")
        self.assertEqual(euclidean[0].argmax().item(), 2)
        self.assertEqual(cosine[0].argmax().item(), 1)
        for metric, out in (("euclidean", euclidean), ("cosine", cosine)):
            for graph, expected in (("knn-out", out), ("knn-in", out.T),
                                    ("knn-union", torch.maximum(out, out.T)),
                                    ("knn-reciprocal", torch.minimum(out, out.T))):
                actual = knn_adjacency(x, knn=1, graph_type=graph, metric=metric)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(actual.diag(), torch.zeros(3)))
                self.assertTrue(((actual == 0) | (actual == 1)).all())
                model = ProtoSGC(**model_arguments(SimulationConfig(
                    graph_type=graph, knn=1, knn_metric=metric), input_dim=2))
                self.assertTrue(torch.equal(model.graph_adjacency(x), expected))

    def test_options_and_simulation(self):
        with patch("sys.argv", ["test"]):
            self.assertEqual(parse_args().knn_metric, "euclidean")
        with patch("sys.argv", ["test", "--knn-metric", "cosine"]):
            args = parse_args()
            self.assertEqual(args.knn_metric, "cosine")
            self.assertEqual(args.dgcg_metric, "euclidean")
        with self.assertRaises(ValueError):
            parse_config({"knn_metric": "invalid"})
        with self.assertRaises(ValueError):
            knn_adjacency(torch.ones(3, 2), knn=1, graph_type="knn-out", metric="invalid")
        result = simulate(parse_config({"knn_metric": "cosine", "steps": 1}))
        self.assertEqual(result["config"]["knn_metric"], "cosine")
