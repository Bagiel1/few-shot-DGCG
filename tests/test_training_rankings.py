"""Verifica contexto global so no treino e equivalencia das correlacoes."""

import contextlib
import io
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from proto_sgc.episodes import FeatureEpisodeSampler, episode_to_device, split_classes
from proto_sgc.config import parse_args, validate_hyperparameters
from proto_sgc.graphs import dgcg
from proto_sgc.graphs.training_rankings import TrainingRankings
from proto_sgc.model import ProtoSGC
from proto_sgc.training import train_model
from simulador.engine import SimulationConfig, model_arguments, synthetic_features


def reference_correlations(rankings, correlation, top_k, p=0.9):
    """Oraculo com conjuntos Python, independente da vetorizacao PyTorch."""
    rows = rankings[:, :top_k].tolist()
    result = []
    for left in rows:
        values = []
        for right in rows:
            jaccards, rbo = [], 0.0
            for depth in range(1, len(left) + 1):
                a, b = set(left[:depth]), set(right[:depth])
                overlap = len(a & b)
                jaccards.append(overlap / len(a | b))
                rbo += (1 - p) * p ** (depth - 1) * overlap / depth
            values.append({
                "rbo": rbo, "jaccardk": np.mean(jaccards),
                "jaccard-median": np.median(jaccards), "jaccard-max": max(jaccards),
            }[correlation])
        result.append(values)
    return torch.tensor(result, dtype=torch.float64)


class TrainingRankingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rankings_match_full_training_search_and_exclude_heldout(self):
        generator = torch.Generator().manual_seed(42)
        features = {label: torch.randn(7, 5, generator=generator, dtype=torch.float64)
                    for label in ("a", "b", "validation", "test")}
        for metric in ("cosine", "euclidean"):
            cache = TrainingRankings(features, ["b", "a"], list_size=200,
                                     metric=metric, batch_size=3)
            expected = dgcg.build_ranked_lists(torch.cat([features["a"], features["b"]]),
                                               14, metric=metric)
            torch.testing.assert_close(cache.rankings, expected)
            self.assertEqual({key[0] for key in cache.row_by_id}, {"a", "b"})
            changed = dict(features, validation=features["validation"] * 1000,
                           test=features["test"] * -100)
            other = TrainingRankings(changed, ["a", "b"], list_size=200, metric=metric)
            torch.testing.assert_close(cache.rankings, other.rankings)
            with self.assertRaisesRegex(ValueError, "fora do conjunto"):
                cache.correlations([("test", 0)], correlation="rbo", top_k=4,
                                   rbo_p=.9, dtype=torch.float64, device=torch.device("cpu"))

    def test_cli_is_opt_in_and_rejects_incompatible_graphs(self):
        with patch.object(sys, "argv", ["protosgc_fewshot.py"]):
            self.assertFalse(parse_args().dgcg_train_global_rankings)
        for graph in ("dgcg", "dgcg-plus", "all", "knn-union"):
            with patch.object(sys, "argv", ["protosgc_fewshot.py", "--graph-type", graph,
                                           "--dgcg-train-global-rankings"]):
                args = parse_args()
                if graph == "knn-union":
                    with self.assertRaises(ValueError):
                        validate_hyperparameters(args)
                else:
                    validate_hyperparameters(args)
        with patch.object(sys, "argv", ["protosgc_fewshot.py", "--model", "protonet",
                                       "--graph-type", "dgcg", "--dgcg-train-global-rankings"]):
            with self.assertRaises(ValueError):
                validate_hyperparameters(parse_args())

    def test_correlation_of_selected_global_rows_matches_full_oracle(self):
        features = {"a": torch.tensor([[0., 0.], [1., 0.], [4., 0.], [9., 0.], [15., 0.]])}
        cache = TrainingRankings(features, ["a"], list_size=5, metric="euclidean")
        rows = [4, 1, 3]  # ordem do episodio e diferente da ordem do cache
        for correlation in ("rbo", "jaccardk", "jaccard-median", "jaccard-max"):
            for top_k in (2, 3, 40):
                expected_full = reference_correlations(cache.rankings, correlation, top_k)
                actual = cache.correlations([("a", i) for i in rows], correlation=correlation,
                                            top_k=top_k, rbo_p=.9, dtype=torch.float64,
                                            device=torch.device("cpu"))
                torch.testing.assert_close(actual, expected_full[rows][:, rows])
        local_rankings = dgcg.build_ranked_lists(features["a"][rows], 3, metric="euclidean")
        self.assertFalse(torch.allclose(
            reference_correlations(cache.rankings[rows], "rbo", 5),
            reference_correlations(local_rankings, "rbo", 5),
        ))

    def test_sample_identity_order_and_duplicate_vectors(self):
        features = {name: torch.ones(6, 3) for name in ("a", "b")}
        sampler = FeatureEpisodeSampler(features, 2, 2, 2)
        episode = sampler.sample(["a", "b"], np.random.default_rng(7))
        moved = episode_to_device(episode, torch.device("cpu"))
        self.assertEqual(episode.sample_ids, moved.sample_ids)
        self.assertEqual(len(set(episode.sample_ids)), 8)
        expected = torch.stack([features[label][index] for label, index in episode.sample_ids])
        torch.testing.assert_close(torch.cat((episode.support_x, episode.query_x)), expected)
        cache = TrainingRankings(features, ["a", "b"], list_size=6, metric="euclidean")
        torch.testing.assert_close(cache.rankings[:, 0], torch.arange(12))
        first_labels = [label for label, _ in episode.sample_ids[:4]]
        self.assertEqual(first_labels[0], first_labels[1])
        self.assertNotEqual(first_labels[0], first_labels[2])

    def test_external_correlations_control_only_selection(self):
        features = torch.tensor([[0., 0.], [1., 0.], [4., 0.], [9., 0.]])
        cfg = SimulationConfig(graph_type="dgcg", knn=1, dgcg_threshold=0.5)
        model = ProtoSGC(**model_arguments(cfg, 2))
        model.dgcg_candidate_k = 1
        local = model.dgcg_correlation_matrix(model.ranked_lists(features, 4))
        torch.testing.assert_close(model.graph_adjacency(features),
                                   model.graph_adjacency(features, dgcg_correlations=local))
        none = model.graph_adjacency(features, dgcg_correlations=torch.zeros(4, 4))
        all_candidates = model.graph_adjacency(features, dgcg_correlations=torch.ones(4, 4))
        self.assertEqual(none.count_nonzero(), 0)
        expected = torch.zeros(4, 4)
        expected[1, 0] = expected[0, 1] = expected[1, 2] = expected[2, 3] = 1
        torch.testing.assert_close(all_candidates, expected)
        model.graph_type = "dgcg-plus"
        weighted = model.graph_adjacency(features, dgcg_correlations=torch.ones(4, 4))
        self.assertTrue(torch.equal(weighted.ne(0), expected.ne(0)))
        self.assertTrue(torch.all(weighted[expected.bool()] < 1))
        with self.assertRaisesRegex(ValueError, "formato"):
            model.graph_adjacency(features, dgcg_correlations=torch.ones(5, 5))

    def test_train_uses_global_but_validation_and_test_never_do(self):
        class RecordingModel(ProtoSGC):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.calls = []

            def forward(self, *args, **kwargs):
                corr = kwargs.get("dgcg_correlations")
                self.calls.append((self.training, corr is not None))
                return super().forward(*args, **kwargs)

        for graph, grande, rbf, enabled in (
            ("dgcg", "off", False, True),
            ("dgcg-plus", "euclidean", False, True),
            ("dgcg", "rbo", True, True),
            ("knn-out", "off", False, True),  # --graph-type all deixa kNN intacto
            ("dgcg", "off", False, False),
        ):
            with self.subTest(graph=graph, grande=grande, enabled=enabled), tempfile.TemporaryDirectory() as directory:
                cfg = SimulationConfig(graph_type=graph, grande_metric=grande,
                                       cosine_rbf_weight=rbf, steps=3, n_way=2)
                features = synthetic_features(cfg)
                splits = split_classes(sorted(features), cfg.n_way, None, cfg.seed)
                sampler = FeatureEpisodeSampler(features, cfg.n_way, cfg.n_shot, cfg.n_query)
                args = SimpleNamespace(**(asdict(cfg) | {
                    "model": cfg.model_name, "work_dir": Path(directory),
                    "grande": grande != "off", "dgcg_target_density": None,
                    "dgcg_target_degree": (4., 6.), "dgcg_list_size": 200,
                    "dgcg_rbo_p": .9, "dgcg_train_global_rankings": enabled,
                    "train_episodes": 3, "val_episodes": 2, "test_episodes": 2,
                    "val_every": 2,
                }))
                model = RecordingModel(**model_arguments(cfg, 8))
                before = model.theta.weight.detach().clone()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    _, metrics, checkpoint = train_model(model, sampler, splits, args, torch.device("cpu"))
                active = enabled and graph.startswith("dgcg")
                self.assertEqual(model.calls, [(True, active), (True, active),
                                              (False, False), (False, False), (True, active),
                                              (False, False), (False, False), (False, False), (False, False)])
                self.assertEqual("train-global-rankings" in checkpoint.name, active)
                self.assertTrue(math.isfinite(metrics.loss))
                self.assertFalse(torch.equal(before, model.theta.weight))
                self.assertFalse(any("rankings" in key for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
