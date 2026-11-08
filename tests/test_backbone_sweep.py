import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import experimentos_backbones as grid
from proto_sgc.config import parse_args, validate_hyperparameters


def parsed(job, args):
    with patch("sys.argv", grid.command(job, args)[1:]):
        a = parse_args()
    validate_hyperparameters(a)
    if a.train_free:
        a.feature_dim = 768 if a.backbone == "vit-b-16" else 512
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}


def fake_result(job, args, accuracy=0.8):
    metric = {k: accuracy for k in ("accuracy", "ci95", "macro_f1", "macro_f1_ci95",
                                   "balanced_accuracy", "balanced_accuracy_ci95", "loss")}
    metric["graph"] = {}
    return dict(args=parsed(job, args), validation=metric, test=metric,
                execution={"elapsed_seconds": 10})


class BackboneSweepTests(unittest.TestCase):
    def test_real_inventory_reuses_44_and_preserves_original_means(self):
        from proto_sgc.result_inventory import audit_results, reusable_result
        audit = audit_results(grid.ROOT)
        self.assertEqual(len(audit["gpu_trained_records"]), 36)
        self.assertEqual(len(audit["gpu_degree_records"]), 8)
        self.assertEqual(audit["validated_workbook_groups"], 12)
        self.assertEqual(len(audit["cpu_references"]), 24)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = grid.parser().parse_args(["--output-root", tmp])
            plan = list(grid.jobs(args))
            for job in plan:
                record = reusable_result(job, args, audit)
                if record:
                    job["reuse"] = record
            self.assertEqual(sum(bool(j.get("reuse")) for j in plan), 44)
            self.assertTrue(all(j["backbone"] == "resnet18" and j["mode"] == "trained"
                                for j in plan if j.get("reuse")))
            grid.summarize(args.output_root, plan, args)
            with (args.output_root / "resultados.csv").open() as stream:
                results = list(csv.DictReader(stream))
            self.assertEqual(len(results), 44)
            self.assertTrue(all(r["origin"] == "reused" and r["elapsed_seconds"] == "" for r in results))
            timing = json.loads((args.output_root / "tempo_estimado.json").read_text())
            self.assertEqual((timing["reused"], timing["pending"], timing["measured_seconds"]), (44, 196, 0))

    def test_reuse_rejects_wrong_device_weights_quick_and_density(self):
        from proto_sgc.result_inventory import audit_results, reusable_result
        audit = audit_results(grid.ROOT)
        args = grid.parser().parse_args([])
        job = next(j for j in grid.jobs(args) if j["metodo"] == "dgcg")
        self.assertIsNotNone(reusable_result(job, args, audit))
        for key, value in (("device", "cpu"), ("device", "auto"), ("weights", "random"), ("quick", True)):
            import argparse
            changed = argparse.Namespace(**dict(vars(args), **{key: value}))
            self.assertIsNone(reusable_result(job, changed, audit))
        self.assertIsNone(reusable_result(dict(job, alvo="0.9"), args, audit))
        self.assertIsNone(reusable_result(dict(job, mode="train-free"), args, audit))

    def test_modified_source_blocks_reuse(self):
        from proto_sgc.result_inventory import audit_results, WORKBOOK, INVENTORY, SCALES, CLAUDE, HISTORICAL_METRICS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (WORKBOOK, INVENTORY, SCALES, CLAUDE, HISTORICAL_METRICS):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(grid.ROOT / name, target)
            (root / SCALES).write_text((root / SCALES).read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "alterada"):
                audit_results(root)

    def test_only_historical_results_never_probe_gpu_or_run_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(grid, "hardware_info", side_effect=AssertionError("GPU probe")), \
                 patch.object(grid, "run_job", side_effect=AssertionError("duplicated experiment")):
                self.assertEqual(grid.main(["--backbones", "resnet18", "--modes", "trained",
                                            "--methods", "no-graph", "knn-union", "dgcg",
                                            "--output-root", tmp]), 0)
            timing = json.loads((Path(tmp) / "tempo_estimado.json").read_text())
            self.assertEqual(timing["reused"], 36)
            self.assertEqual(timing["pending"], 0)

    def test_grid_real_cli_and_fixed_hyperparameters(self):
        args = grid.parser().parse_args([])
        plan = list(grid.jobs(args))
        self.assertEqual(len(plan), 240)
        self.assertEqual(len({j["directory"] for j in plan}), 240)
        self.assertEqual(sum(j["mode"] == "train-free" for j in plan), 120)
        for job in plan:
            result = fake_result(job, args)
            self.assertTrue(grid.matches(job, result, args))
            self.assertIsNone(result["args"]["extra_images"])
            self.assertEqual(result["args"]["class_split"], [10, 5, 5])
            result["args"]["backbone"] = "other"
            self.assertFalse(grid.matches(job, result, args))
        self.assertEqual({(j["backbone"], j["mode"], j["n_shot"], j["metodo"]) for j in plan[:40]},
                         {(j["backbone"], j["mode"], j["n_shot"], j["metodo"]) for j in plan})

    def test_quick_and_mismatched_protocols_are_rejected(self):
        args = grid.parser().parse_args(["--quick"])
        job = next(grid.jobs(args))
        result = fake_result(job, args)
        self.assertTrue(grid.matches(job, result, args))
        for key, wrong in (("weights", "random"), ("test_episodes", 1000), ("knn", 8),
                           ("feature_dim", 512), ("cosine_rbf_weight", True), ("seed", 99)):
            broken = dict(result, args=dict(result["args"], **{key: wrong}))
            self.assertFalse(grid.matches(job, broken, args), key)

    def test_densities_match_historical_validation_selection(self):
        import argparse
        from experimentos_escalas import select_micro
        selected = select_micro(argparse.Namespace(datasets=["flowers", "cars"], ways=[5], shots=[1, 5], queries=[10]))
        for (family, _, shot, _), chosen in selected.items():
            self.assertEqual(chosen["target_kind"], "density")
            self.assertEqual(float(chosen["alvo"]), grid.DENSITIES[family + "_micro"][shot])

    def test_dry_run_writes_nothing_and_does_not_require_gpu(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp) / "new"
            with patch.object(grid, "hardware_info", side_effect=AssertionError("GPU probe")):
                self.assertEqual(grid.main(["--dry-run", "--output-root", str(root)]), 0)
            self.assertFalse(root.exists())

    def test_summary_keeps_backbone_mode_and_seed_groups_separate(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = grid.parser().parse_args(["--datasets", "flowers_micro", "--shots", "1", "--seeds", "7", "17"])
            args.output_root = Path(tmp)
            plan = list(grid.jobs(args))
            for job in plan:
                folder = Path(job["directory"])
                folder.mkdir(parents=True)
                accuracy = 0.7 if job["seed"] == 7 else 0.9
                if job["metodo"] != "no-graph":
                    accuracy += 0.02
                (folder / "metrics.json").write_text(json.dumps(fake_result(job, args, accuracy)))
            grid.summarize(args.output_root, plan, args)
            with (Path(tmp) / "medias.csv").open() as stream:
                means = list(csv.DictReader(stream))
            self.assertEqual(len(means), 20)
            for row in means:
                self.assertEqual(row["concluidos"], "2")
                self.assertAlmostEqual(float(row["test_accuracy_std_seed"]), 14.14213562)
                self.assertAlmostEqual(float(row["ganho_vs_sem_grafo_pp_mean"]),
                                       0 if row["metodo"] == "no-graph" else 2)

    def test_degree_variants_are_independent_from_density_and_each_other(self):
        args = grid.parser().parse_args([])
        for method, interval in grid.DEGREES.items():
            job = next(j for j in grid.jobs(args) if j["metodo"] == method)
            cmd = grid.command(job, args)
            self.assertNotIn("--dgcg-target-density", cmd)
            self.assertEqual(cmd[cmd.index("--dgcg-target-degree")+1:cmd.index("--dgcg-target-degree")+3],
                             [str(v) for v in interval])
            result = fake_result(job, args)
            self.assertEqual(result["args"]["graph_type"], "dgcg")
            self.assertIsNone(result["args"]["dgcg_target_density"])
            self.assertTrue(grid.matches(job, result, args))
            result["args"]["dgcg_target_density"] = [0.125, 0.125]
            self.assertFalse(grid.matches(job, result, args))
            result = fake_result(job, args)
            result["args"]["dgcg_target_degree"] = [8, 12]
            self.assertFalse(grid.matches(job, result, args))

    def test_degree_reuse_only_matches_original_seed_and_target(self):
        from proto_sgc.result_inventory import audit_results, reusable_result
        audit = audit_results(grid.ROOT)
        args = grid.parser().parse_args([])
        for method in grid.DEGREES:
            job = next(j for j in grid.jobs(args) if j["metodo"] == method)
            record = reusable_result(job, args, audit)
            self.assertIsNotNone(record)
            self.assertEqual(record["source_row"]["alvo"], job["alvo"])
            self.assertIsNone(reusable_result(dict(job, seed=17), args, audit))
            self.assertIsNone(reusable_result(dict(job, alvo="8:12"), args, audit))

    def test_resume_limit_failure_and_manifest_summary(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            argv = ["--datasets", "flowers_micro", "--backbones", "resnet18", "--modes", "trained",
                    "--shots", "1", "--seeds", "7", "--methods", "no-graph", "knn-union",
                    "--device", "cpu", "--output-root", tmp]
            calls = []

            def run(job, args):
                calls.append(job)
                folder = Path(job["directory"])
                folder.mkdir(parents=True)
                (folder / "metrics.json").write_text(json.dumps(fake_result(job, args)))

            with patch.object(grid, "hardware_info", return_value={"device": "cpu"}), patch.object(grid, "run_job", side_effect=run):
                self.assertEqual(grid.main(argv + ["--max-runs", "1"]), 0)
                self.assertEqual(len(calls), 1)
                self.assertEqual(grid.main(argv), 0)
                self.assertEqual(len(calls), 2)
                self.assertEqual(grid.main(argv), 0)
                self.assertEqual(len(calls), 2)
            # Usa o protocolo salvo, mesmo sem repetir as opcoes de selecao.
            self.assertEqual(grid.main(["--summarize-only", "--output-root", tmp]), 0)
            with self.assertRaises(SystemExit):
                grid.main(argv + ["--quick"])

    def test_subprocess_collects_nested_metrics_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = grid.parser().parse_args(["--output-root", tmp])
            job = next(grid.jobs(args))

            def execute(*unused, **kwargs):
                work = args.output_root / "_work" / job["dataset"] / job["backbone"] / "nested"
                work.mkdir(parents=True)
                checkpoint = work / "best.pt"
                checkpoint.write_bytes(b"test snapshot")
                result = fake_result(job, args)
                result["checkpoint"] = str(checkpoint)
                (work / "best.metrics.json").write_text(json.dumps(result))
                return type("Completed", (), {"returncode": 0})()

            with patch.object(grid.subprocess, "run", side_effect=execute):
                grid.run_job(job, args)
            folder = Path(job["directory"])
            self.assertEqual((folder / "checkpoint.pt").read_bytes(), b"test snapshot")
            self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "complete")
            with patch.object(grid.subprocess, "run", side_effect=RuntimeError("fake failure")):
                with self.assertRaises(RuntimeError):
                    grid.run_job(job, args)
            self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
