"""Verificações sem downloads: python -m unittest discover -s tests -v."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from proto_sgc.config import GRAPH_TYPES
from proto_sgc.episodes import FeatureEpisodeSampler, split_classes
from proto_sgc.graphs.knn import knn_adjacency
from proto_sgc.model import ProtoSGC
from proto_sgc.training import train_model
from simulador.engine import (
    ObservedProtoSGC, SimulationConfig, inspect_episode, load_features,
    model_arguments, parse_config, simulate, synthetic_features,
)
from simulador.server import make_handler
from simulador.source_code import CATALOG, ROOT, code_catalog


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_observation_matches_original_forward_and_gradients(self):
        for graph in GRAPH_TYPES:
            for grande in ("off", "euclidean", "cosine", "rbo"):
                for rbf in (False, True):
                    with self.subTest(graph=graph, grande=grande, rbf=rbf):
                        cfg = SimulationConfig(graph_type=graph, grande_metric=grande,
                                               cosine_rbf_weight=rbf, n_way=2, steps=1)
                        features = synthetic_features(cfg)
                        sampler = FeatureEpisodeSampler(features, 2, cfg.n_shot, cfg.n_query)
                        episode = sampler.sample(sorted(features), np.random.default_rng(108))
                        torch.manual_seed(7)
                        original = ProtoSGC(**model_arguments(cfg, 8))
                        observed = ObservedProtoSGC(**model_arguments(cfg, 8))
                        observed.load_state_dict(original.state_dict())
                        expected = original(episode.support_x, episode.support_y, episode.query_x, 2)
                        actual = observed(episode.support_x, episode.support_y, episode.query_x, 2)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        for logits in (expected, actual):
                            F.cross_entropy(logits, episode.query_y).backward()
                        for p, q in zip(original.parameters(), observed.parameters(), strict=True):
                            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                        trace = inspect_episode(observed, episode, cfg)
                        torch.testing.assert_close(torch.tensor(trace["logits"]), expected, rtol=0, atol=0)
                        self.assertEqual(len(trace["hops"]), cfg.sgc_hops + 1)

    def test_knn_is_euclidean_and_binary_by_default(self):
        # Para o no 0, cosseno escolheria o no 1 (mesma direcao), enquanto
        # Euclidiana escolhe o no 2 (menor distancia absoluta).
        features = torch.tensor([[1.0, 0.0], [2.0, 0.0], [1.0, 0.2]])
        adjacency = knn_adjacency(features, knn=1, graph_type="knn-out")
        self.assertEqual(adjacency[0, 1].item(), 0.0)
        self.assertEqual(adjacency[0, 2].item(), 1.0)
        self.assertEqual(set(adjacency.unique().tolist()), {0.0, 1.0})

    def test_standard_dgcg_is_binary(self):
        cfg = SimulationConfig(
            graph_type="dgcg", dgcg_metric="euclidean", n_way=2, steps=1
        )
        model = ProtoSGC(**model_arguments(cfg, 2))
        features = torch.tensor(
            [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]]
        )
        adjacency = model.graph_adjacency(features)
        self.assertTrue(set(adjacency.unique().tolist()) <= {0.0, 1.0})

    def test_training_matches_original_training_loop(self):
        # Testa seleção por validação, restauração para o teste e seeds iguais.
        for options in ({}, {"graph_type": "dgcg-plus", "grande_metric": "cosine"}, {"model_name": "protonet"}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                cfg = SimulationConfig(steps=6, n_way=2, **options)
                result = simulate(cfg)
                features = synthetic_features(cfg)
                sampler = FeatureEpisodeSampler(features, cfg.n_way, cfg.n_shot, cfg.n_query)
                splits = split_classes(sorted(features), cfg.n_way, None, cfg.seed)
                torch.manual_seed(cfg.seed)
                model = ProtoSGC(**model_arguments(cfg, 8))
                args = SimpleNamespace(**(asdict(cfg) | {
                    "model": cfg.model_name, "work_dir": Path(directory),
                    "grande": cfg.grande_metric != "off", "dgcg_target_density": None,
                    "dgcg_target_degree": (4.0, 6.0), "train_episodes": cfg.steps,
                    "val_episodes": 10, "test_episodes": 20, "val_every": 5,
                }))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    _, metrics, _ = train_model(model, sampler, splits, args, torch.device("cpu"))
                self.assertEqual(asdict(metrics), result["test"])
                selected = result["steps"][result["best_step"] - 1]["parameters_after"]["theta"]
                torch.testing.assert_close(torch.tensor(selected), model.theta.weight.T, rtol=0, atol=0)

    def test_same_episode_before_after_and_adamw_formula(self):
        cfg = SimulationConfig(steps=3, grad_clip=0.05)
        first, second = simulate(cfg), simulate(cfg)
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)
        self.assertTrue(any(snapshot["clip_factor"] < 1 for snapshot in first["steps"]))
        for snapshot in first["steps"]:
            self.assertEqual(snapshot["before"]["x"], snapshot["after"]["x"])
            self.assertEqual(snapshot["before"]["adjacency"], snapshot["after"]["adjacency"])
            self.assertEqual(snapshot["probe"]["x"], first["initial_probe"]["x"])
            t = snapshot["step"]
            m = snapshot["adam_first_moment"] / (1 - 0.9 ** t)
            v = snapshot["adam_second_moment"] / (1 - 0.999 ** t)
            before = snapshot["parameters_before"]["theta"][0][0]
            expected = before * (1 - cfg.learning_rate * cfg.weight_decay) - cfg.learning_rate * m / (math.sqrt(v) + 1e-8)
            self.assertAlmostEqual(expected, snapshot["parameters_after"]["theta"][0][0], places=6)
            self.assertAlmostEqual(snapshot["clip_factor"], min(1, cfg.grad_clip / (snapshot["gradient_norm"] + 1e-6)))
            nodes = snapshot["episode"]["nodes"]
            support = {node["id"] for node in nodes if node["role"] == "support"}
            query = {node["id"] for node in nodes if node["role"] == "query"}
            self.assertFalse(support & query)

    def test_baseline_and_cache(self):
        cfg = SimulationConfig(steps=1, model_name="protonet", n_way=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pt"
            torch.save({"features_by_class": synthetic_features(cfg)}, path)
            result = simulate(cfg, load_features(path))
        self.assertEqual(result["source"], "cache")
        trace = result["steps"][0]["before"]
        self.assertEqual(len(trace["hops"]), 1)
        self.assertEqual(sum(map(sum, trace["adjacency"])), 0)

    def test_invalid_parameters(self):
        for values in ({"steps": 121}, {"steps": 2.5}, {"seed": True}, {"noise": float("nan")},
                       {"model_name": "protonet", "grande_metric": "rbo"}, {"dgcg_threshold": 2},
                       {"cosine_rbf_weight": "true"}, {"unknown": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_config(values)

    def test_high_dimension_cache_keeps_full_computation(self):
        cfg = SimulationConfig(steps=1, n_way=2)
        generator = torch.Generator().manual_seed(11)
        features = {f"C{i}": torch.randn(16, 512, generator=generator) for i in range(6)}
        result = simulate(cfg, features)
        self.assertEqual(result["input_dim"], 512)
        self.assertEqual(len(result["steps"][0]["before"]["x"][0]), 8)
        self.assertEqual(len(result["initial_parameters"]["theta"]), 8)
        self.assertTrue(math.isfinite(result["test"]["loss"]))

    def test_http_routes_in_memory(self):
        # Passa requisições HTTP pelo handler real, sem abrir portas.
        class Connection:
            def __init__(self, request):
                self.request, self.response = io.BytesIO(request), bytearray()

            def makefile(self, *args):
                return self.request

            def sendall(self, data):
                self.response.extend(data)

        handler = make_handler()

        def request(method, path, payload=None, host="127.0.0.1:8765", origin=None):
            body = json.dumps(payload).encode() if payload is not None else b""
            headers = f"{method} {path} HTTP/1.0\r\nHost: {host}\r\nContent-Length: {len(body)}\r\n"
            if origin:
                headers += f"Origin: {origin}\r\n"
            connection = Connection(headers.encode() + b"\r\n" + body)
            with contextlib.redirect_stderr(io.StringIO()):
                handler(connection, ("127.0.0.1", 10000), SimpleNamespace(server_port=8765))
            head, body = bytes(connection.response).split(b"\r\n\r\n", 1)
            return int(head.split()[1]), body

        for path in ("/", "/app.js", "/style.css", "/api/config", "/api/code"):
            self.assertEqual(request("GET", path)[0], 200)
        status, code = request("GET", "/api/code")
        self.assertEqual(json.loads(code)["prediction"]["path"], "proto_sgc/model.py")
        status, body = request("POST", "/api/simulate", {"steps": 1, "n_way": 2})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["steps"]), 1)
        self.assertEqual(request("POST", "/api/simulate", {"steps": 500})[0], 400)
        self.assertEqual(request("GET", "/../requirements.txt")[0], 404)
        self.assertEqual(request("GET", "/", host="example.com")[0], 403)
        self.assertEqual(request("POST", "/api/simulate", {}, origin="https://example.com")[0], 403)

    def test_code_excerpts_are_exact_source_with_correct_lines(self):
        catalog = code_catalog()
        self.assertEqual(set(catalog), set(CATALOG))
        for identifier, item in catalog.items():
            with self.subTest(identifier=identifier):
                original = (ROOT / item["path"]).read_text().splitlines()
                self.assertEqual(item["lines"], original[item["start_line"] - 1:item["end_line"]])
                self.assertGreater(len(item["lines"]), 0)
        self.assertIn("loss.backward()", "\n".join(catalog["train_step"]["lines"]))
        self.assertNotIn("loss.backward()", "\n".join(catalog["loss"]["lines"]))
        self.assertIn("return -scale * squared_distances", "\n".join(catalog["prediction"]["lines"]))


if __name__ == "__main__":
    unittest.main()
