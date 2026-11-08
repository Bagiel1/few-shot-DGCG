import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from proto_sgc.config import parse_args, validate_hyperparameters
from proto_sgc.data import ImageRecord
from proto_sgc.episodes import FeatureEpisodeSampler
from proto_sgc.features import build_vit_b_16, extract_or_load_features
from proto_sgc.model import ProtoSGC
from proto_sgc.training import evaluate_without_training


def cli(*argv):
    with patch.object(sys, "argv", ["test", *argv]):
        args = parse_args()
    validate_hyperparameters(args)
    return args


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_defaults_and_invalid_vit_resolution(self):
        self.assertEqual(cli().image_size, 128)
        self.assertEqual(cli("--backbone", "vit-b-16").image_size, 224)
        with self.assertRaisesRegex(ValueError, "224"):
            cli("--backbone", "vit-b-16", "--image-size", "128")
        with self.assertRaisesRegex(ValueError, "sem extras"):
            cli("--train-free", "--extra-images", "0")

    def test_real_vit_forward_and_frozen_parameters_offline(self):
        # Pesos aleatorios aqui verificam arquitetura/saida, nao acuracia.
        model, dim = build_vit_b_16("random")
        self.assertEqual(dim, 768)
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 768))
        self.assertTrue(torch.isfinite(output).all())

    def test_vit_cache_preserves_sample_order_and_avoids_second_extraction(self):
        class TinyBackbone(torch.nn.Module):
            def forward(self, images):
                self.last_shape = tuple(images.shape)
                return images.mean(dim=(2, 3))

        with tempfile.TemporaryDirectory() as tmp:
            args = cli("--backbone", "vit-b-16", "--work-dir", tmp)
            records = []
            for i, label in enumerate(("b", "a", "b")):
                path = Path(tmp) / f"{i}.png"
                Image.new("RGB", (128, 128), color=(i * 50, 0, 0)).save(path)
                records.append(ImageRecord(path, label))
            backbone = TinyBackbone()
            with patch("proto_sgc.features.build_vit_b_16", return_value=(backbone, 3)):
                features, dim = extract_or_load_features(records, args, torch.device("cpu"))
            self.assertEqual(backbone.last_shape, (3, 3, 224, 224))
            self.assertEqual(dim, 3)
            self.assertEqual(features["b"].shape, (2, 3))
            self.assertLess(features["b"][0, 0], features["b"][1, 0])
            with patch("proto_sgc.features.build_vit_b_16", side_effect=AssertionError("cache miss")):
                cached, _ = extract_or_load_features(records, args, torch.device("cpu"))
            torch.testing.assert_close(features["b"], cached["b"])
            cache_path = next(Path(tmp).glob("features_*.pt"))
            content = torch.load(cache_path, weights_only=False)
            self.assertEqual(content["feature_spec"]["weights"], "IMAGENET1K_V1")
            # Um cache com metadados incorretos precisa ser invalidado.
            content["feature_spec"]["weights"] = "different"
            torch.save(content, cache_path)
            with patch("proto_sgc.features.build_vit_b_16", return_value=(backbone, 3)) as builder:
                extract_or_load_features(records, args, torch.device("cpu"))
                builder.assert_called_once()

    def test_train_free_has_identity_no_optimizer_and_paired_test_episodes(self):
        for method in ("protonet", "proto-sgc"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                args = cli("--train-free", "--model", method, "--n-way", "2",
                           "--val-episodes", "2", "--test-episodes", "3", "--work-dir", tmp)
                features = {str(i): torch.randn(10, 8) for i in range(6)}
                sampler = FeatureEpisodeSampler(features, 2, 1, 5)
                splits = {"train": ["0", "1"], "val": ["2", "3"], "test": ["4", "5"]}
                args.feature_dim = 8
                model = ProtoSGC(8, 8, method, 5, "knn-union", 2, 0.2)
                with patch("torch.optim.AdamW", side_effect=AssertionError("optimizer forbidden")):
                    model, metrics, path = evaluate_without_training(
                        model, sampler, splits, args, torch.device("cpu"))
                torch.testing.assert_close(model.theta.weight, torch.eye(8))
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                self.assertTrue(0 <= metrics.accuracy <= 1)
                result = json.loads(path.with_suffix(".metrics.json").read_text())
                self.assertIsNone(result["selection_metric"])
                self.assertEqual(result["training_episodes_executed"], 0)
                first = sampler.sample(splits["test"], np.random.default_rng(args.seed + 303))
                # Dimensao do embedding nao altera a identidade dos exemplos sorteados.
                wider = FeatureEpisodeSampler({k: torch.cat((v, v), 1) for k, v in features.items()}, 2, 1, 5)
                second = wider.sample(splits["test"], np.random.default_rng(args.seed + 303))
                self.assertEqual(first.sample_ids, second.sample_ids)


if __name__ == "__main__":
    unittest.main()
