import sys
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import experimentos_extras as extra_sweep
from proto_sgc.config import parse_args, validate_hyperparameters
from proto_sgc.episodes import FeatureEpisodeSampler
from proto_sgc.model import ProtoSGC
from proto_sgc.reference_bank import QueryContextBank
from proto_sgc.training import extra_graph_options, prepare_extra_banks, train_model


def cli(*argv):
    with patch.object(sys, "argv", ["test", *argv]):
        args = parse_args()
    validate_hyperparameters(args)
    return args


class QueryContextTests(unittest.TestCase):
    def setUp(self):
        features = {
            str(label): torch.arange(30 * 4, dtype=torch.float32).reshape(30, 4) + label * 1000
            for label in range(6)
        }
        self.sampler = FeatureEpisodeSampler(features, n_way=2, n_shot=1, n_query=1)
        self.classes = ["0", "1", "2"]

    def bank(self, active):
        return QueryContextBank(
            self.sampler,
            self.classes,
            pool_per_class=9,
            active_per_class=active,
        )

    def test_zero_and_nine_contexts_keep_support_and_evaluated_queries_paired(self):
        control = self.bank(0).sample(np.random.default_rng(17))
        treatment_bank = self.bank(9)
        treatment = treatment_bank.sample(np.random.default_rng(17))

        self.assertEqual(control.sample_ids, treatment.sample_ids)
        torch.testing.assert_close(control.support_x, treatment.support_x)
        torch.testing.assert_close(control.query_x, treatment.query_x)
        self.assertEqual(treatment.query_x.shape, (2, 4))
        self.assertEqual(treatment_bank.extra_features(treatment).shape, (18, 4))

    def test_context_subsets_are_nested(self):
        contexts = []
        for active in (1, 4, 9):
            bank = self.bank(active)
            episode = bank.sample(np.random.default_rng(23))
            contexts.append(bank.extra_features(episode))

        torch.testing.assert_close(contexts[0], contexts[1][[0, 4]])
        torch.testing.assert_close(
            contexts[1], contexts[2][[0, 1, 2, 3, 9, 10, 11, 12]]
        )

    def test_only_features_are_passed_to_propagation(self):
        bank = self.bank(9)
        episode = bank.sample(np.random.default_rng(29))

        class Model:
            model_name = "proto-sgc"

        options = extra_graph_options(Model(), episode, bank, mode="propagation")
        self.assertEqual(set(options), {"extra_x"})
        self.assertEqual(options["extra_x"].shape, (18, 4))
        self.assertEqual(episode.query_y.tolist(), [0, 1])

    def test_phase_mask_changes_only_active_context_count(self):
        args = cli(
            "--n-way", "2", "--n-shot", "1", "--n-query", "1",
            "--extra-source", "queries", "--extra-mode", "propagation",
            "--extra-bank-per-class", "9", "--extra-images", "18",
            "--extra-phases", "test",
        )
        splits = {"train": ["0", "1"], "val": ["2", "3"], "test": ["4", "5"]}
        banks = prepare_extra_banks(self.sampler, splits, args)
        self.assertEqual({phase: bank.count for phase, bank in banks.items()}, {
            "train": 0, "val": 0, "test": 18,
        })
        train_episode = banks["train"].sample(np.random.default_rng(7))
        test_episode = banks["test"].sample(np.random.default_rng(7))
        self.assertEqual(len(train_episode.query_y), 2)
        self.assertEqual(len(test_episode.query_y), 2)

    def test_invalid_combinations_are_rejected(self):
        invalid = [
            ("--extra-source", "queries", "--extra-mode", "propagation", "--extra-images", "46"),
            ("--extra-source", "queries", "--extra-mode", "propagation", "--extra-images", "-1"),
            ("--extra-source", "queries", "--extra-mode", "propagation", "--extra-images", "44"),
            ("--extra-source", "queries", "--extra-mode", "propagation", "--extra-images", "45", "--model", "protonet"),
            ("--extra-source", "queries", "--extra-images", "45"),
            ("--extra-source", "queries", "--extra-mode", "propagation", "--extra-images", "45", "--extra-phases", "test", "test"),
        ]
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                cli(*argv)

    def test_training_uses_18_unlabeled_nodes_and_only_two_query_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            args = cli(
                "--n-way", "2", "--n-shot", "1", "--n-query", "1",
                "--extra-source", "queries", "--extra-mode", "propagation",
                "--extra-bank-per-class", "9", "--extra-images", "18",
                "--train-episodes", "2", "--val-episodes", "2",
                "--test-episodes", "2", "--val-every", "1",
                "--work-dir", directory,
            )
            splits = {"train": ["0", "1"], "val": ["2", "3"], "test": ["4", "5"]}
            model = ProtoSGC(4, 4, "proto-sgc", 2, "knn-union", 2, 0.2)
            query_counts = []
            extra_counts = []
            original_forward = model.forward

            def track_forward(support_x, support_y, query_x, n_way, **options):
                query_counts.append(len(query_x))
                extra_counts.append(len(options.get("extra_x", [])))
                return original_forward(support_x, support_y, query_x, n_way, **options)

            with patch.object(model, "forward", side_effect=track_forward), redirect_stdout(StringIO()):
                _, metrics, checkpoint = train_model(
                    model, self.sampler, splits, args, torch.device("cpu")
                )

            self.assertEqual(set(query_counts), {2})
            self.assertEqual(set(extra_counts), {18})
            self.assertEqual(metrics.graph["nodes_per_episode"], 22)
            result = json.loads(Path(checkpoint).with_suffix(".metrics.json").read_text())
            self.assertEqual(result["extra_context"]["requested_total"], 18)
            self.assertEqual(
                result["extra_context"]["source"],
                "queries_from_same_master_episode",
            )

    def test_battery_plan_has_one_control_and_eight_context_conditions(self):
        args = extra_sweep.parser().parse_args([
            "--datasets", "flowers_micro",
            "--methods", "knn-union",
            "--n-way", "5", "--n-shot", "1", "--n-query", "1",
            "--extra-source", "queries",
            "--extras-per-class", "5", "9",
            "--bank-per-class", "9",
            "--phases", "train", "train-val", "test", "all",
            "--modes", "propagation",
            "--seeds", "7",
        ])
        args.work_root = args.output_root / "_work"
        plan = list(extra_sweep.jobs(args))

        self.assertEqual(len(plan), 9)
        self.assertEqual(sum(job["extras_total"] == 0 for job in plan), 1)
        self.assertEqual(
            {(job["extras_total"], job["extra_phase"]) for job in plan if job["extras_total"]},
            {(count, phase) for count in (25, 45) for phase in ("train", "train-val", "test", "all")},
        )
        command = extra_sweep.command(plan[-1], args)
        self.assertEqual(command[command.index("--extra-source") + 1], "queries")


if __name__ == "__main__":
    unittest.main()
