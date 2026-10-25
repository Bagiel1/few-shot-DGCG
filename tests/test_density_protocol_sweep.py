import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import experimentos_densidades_protocolos as sweep
from proto_sgc.config import parse_args, validate_hyperparameters
from proto_sgc.episodes import FeatureEpisodeSampler
from proto_sgc.model import ProtoSGC
from proto_sgc.training import train_model


class DensityProtocolTests(unittest.TestCase):
    def args(self, *extra):
        return sweep.parser().parse_args(['--datasets','bacteria_micro','--ways','2',
            '--shots','1','--queries','2','--densities','.125','.25',*extra])

    def fixture(self, args):
        plan=list(sweep.jobs(args))
        metrics=dict(loss=.5,accuracy=.8,ci95=.02,macro_f1=.78,macro_f1_ci95=.02,
                     balanced_accuracy=.8,balanced_accuracy_ci95=.02,
                     graph=dict(degree_mean=1.,density_mean=.2))
        results=[]
        for job in plan:
            results.append(dict(args=dict(dgcg_candidate_k=200,dgcg_list_size=200,seed=7,
                dgcg_target_degree=[4.,6.],dgcg_target_density=[float(job['alvo'])]*2,
                train_episodes=2000,work_dir='density'),class_splits={'test':['0','1']},
                validation=dict(metrics),test=dict(metrics)))
        return plan,results

    def test_default_grid_count_and_unique_paths(self):
        args=sweep.parser().parse_args([])
        plan=list(sweep.jobs(args))
        self.assertEqual(len(plan),4050)
        self.assertEqual(len({j['directory'] for j in plan}),4050)
        self.assertEqual(len({j['alvo'] for j in plan}),15)
        self.assertIn('0.125',{j['alvo'] for j in plan})

    def test_commands_set_density_only_and_preserve_protocol(self):
        args=self.args('--quick','--device','cuda')
        for job in sweep.jobs(args):
            cmd=sweep.command(job,args)
            self.assertNotIn('--dgcg-target-degree',cmd)
            self.assertNotIn('--dgcg-extra-references',cmd)
            self.assertNotIn('--dgcg-train-global-rankings',cmd)
            self.assertIn('--quick',cmd)
            index=cmd.index('--dgcg-target-density')
            self.assertEqual(cmd[index+1:index+3],[job['alvo']]*2)
            self.assertEqual(cmd[cmd.index('--n-query')+1],'2')
            self.assertEqual(cmd[cmd.index('--class-split')+1:cmd.index('--class-split')+4],['10','5','5'])
            with patch.object(sys,'argv',['test',*cmd[2:]]):parsed=parse_args()
            validate_hyperparameters(parsed)

    def test_invalid_density_rejected(self):
        for value in ['0','1','-0.1','nan','inf','x','0.1:0.2']:
            with self.assertRaises(Exception):sweep.density(value)

    def test_only_validation_selects_and_partial_grid_not_selected(self):
        args=self.args();plan,results=self.fixture(args)
        rows=[sweep.flatten(j,r) for j,r in zip(plan,results)]
        rows[0]['val_accuracy']=81;rows[0]['test_accuracy']=60
        rows[1]['val_accuracy']=80;rows[1]['test_accuracy']=99
        self.assertEqual(sweep.select_densities(rows,plan)[0]['alvo'],'0.125')
        self.assertEqual(sweep.select_densities(rows[:1],plan),[])
        rows[1]['val_accuracy']=81
        winner=sweep.select_densities(rows,plan)[0]
        self.assertEqual(winner['alvo'],'0.125')
        self.assertEqual(winner['alvos_empatados'],'0.125;0.25')

    def test_effective_degree_and_actual_density_distinct(self):
        plan,results=self.fixture(self.args())
        row=sweep.flatten(plan[0],results[0])
        self.assertEqual(row['episode_nodes'],6)
        self.assertEqual(row['grau_alvo_equivalente'],.625)
        self.assertEqual(row['test_density_mean'],.2)
        self.assertAlmostEqual(row['test_desvio_densidade'],.075)

    def test_baseline_compatibility_and_missing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,results=self.fixture(self.args())
            job,result=plan[0],results[0]
            self.assertEqual(sweep.baseline_comparison(job,result,root)['baseline_status'],'ausente')
            folder=root/job['dataset']/'2way_1shot_2query'/'seed7'/'dgcg-deg4-6'
            folder.mkdir(parents=True)
            baseline=json.loads(json.dumps(result))
            baseline['args']['dgcg_target_density']=None
            baseline['args']['work_dir']='other'
            baseline['test']['accuracy']=.7
            path=folder/'metrics.json';path.write_text(json.dumps(baseline))
            comparison=sweep.baseline_comparison(job,result,root)
            self.assertEqual(comparison['baseline_status'],'compativel')
            self.assertAlmostEqual(comparison['ganho_teste_vs_4_6_pp'],10)
            baseline['args']['train_episodes']=200
            path.write_text(json.dumps(baseline))
            comparison=sweep.baseline_comparison(job,result,root)
            self.assertEqual(comparison['baseline_status'],'incompativel')
            self.assertNotIn('ganho_teste_vs_4_6_pp',comparison)

    def test_cache_copy_does_not_share_or_overwrite_work_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.args()
            args.output_root=root/'new';args.feature_cache_root=root/'old'
            source=args.feature_cache_root/'bacteria_micro';source.mkdir(parents=True)
            # Teste do mecanismo de copia: arquivo representa apenas bytes de cache.
            cached=source/'features_resnet18_imagenet_128px.pt';cached.write_bytes(b'cache')
            (source/'checkpoint.pt').write_bytes(b'not-embeddings')
            work=sweep.seed_feature_cache(args,'bacteria_micro')
            self.assertEqual((work/cached.name).read_bytes(),b'cache')
            self.assertFalse((work/'checkpoint.pt').exists())
            (work/cached.name).write_bytes(b'local')
            sweep.seed_feature_cache(args,'bacteria_micro')
            self.assertEqual((work/cached.name).read_bytes(),b'local')
            self.assertEqual(cached.read_bytes(),b'cache')

    def test_variation_one_axis_and_plot_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.args('--output-root',directory)
            plan,results=self.fixture(args)
            rows=[sweep.flatten(j,r) for j,r in zip(plan,results)]
            changes=sweep.density_variation([rows[0],dict(rows[1],n_query=5)])
            self.assertEqual(changes[0]['eixo_variado'],'n_query')
            self.assertEqual(changes[0]['amplitude_densidade'],.125)
            for job,result in zip(plan,results):
                folder=Path(job['directory']);folder.mkdir(parents=True)
                (folder/'metrics.json').write_text(json.dumps(result))
            with contextlib.redirect_stdout(io.StringIO()):
                sweep.summarize(Path(directory),plan,Path(directory)/'absent')
            self.assertIn('macro_f1',(Path(directory)/'resultados_densidades_protocolos.csv').read_text())
            self.assertIn('0.125',(Path(directory)/'melhores_densidades_por_protocolo.csv').read_text())
            self.assertEqual(len(list((Path(directory)/'graficos').glob('*.png'))),3)

    def test_real_training_with_density_and_reported_metrics(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys,'argv',['test','--graph-type','dgcg',
                '--dgcg-target-density','.125','.125','--n-way','2','--n-shot','1','--n-query','2',
                '--train-episodes','2','--val-episodes','2','--val-every','1',
                '--test-episodes','3','--work-dir',directory]):args=parse_args()
            validate_hyperparameters(args)
            gen=torch.Generator().manual_seed(7)
            sampler=FeatureEpisodeSampler({str(i):torch.randn(12,8,generator=gen) for i in range(6)},2,1,2)
            model=ProtoSGC(8,8,'proto-sgc',5,'dgcg',2,.2,dgcg_target_density=(.125,.125))
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                _,metrics,path=train_model(model,sampler,{'train':['0','1'],'val':['2','3'],'test':['4','5']},args,torch.device('cpu'))
            result=json.loads(path.with_suffix('.metrics.json').read_text())
            self.assertEqual(result['test'],asdict(metrics))
            self.assertIn('rho0p125',path.name)
            self.assertAlmostEqual(metrics.accuracy,metrics.balanced_accuracy,places=6)
            # No modo automatico o fallback preserva >=1 aresta por candidato/no.
            self.assertGreaterEqual(metrics.graph['density_mean'],.2)

    def test_runner_resume_failure_and_summary_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);argv=['--datasets','bacteria_micro','--ways','2','--shots','1',
                 '--queries','2','--densities','.125','--output-root',str(root/'runs'),
                 '--baseline-root',str(root/'baseline'),'--feature-cache-root',str(root/'cache')]
            def train(cmd,**kwargs):
                work=Path(cmd[cmd.index('--work-dir')+1]);work.mkdir(parents=True,exist_ok=True)
                checkpoint=work/'fake.pt';checkpoint.write_bytes(b'checkpoint')
                _,results=self.fixture(self.args())
                result=results[0];result['checkpoint']=str(checkpoint)
                (work/'fake.metrics.json').write_text(json.dumps(result))
                return SimpleNamespace(returncode=0)
            with patch.object(sweep.subprocess,'run',side_effect=train) as process, \
                    patch.object(sweep,'plot_results'),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sweep.main(argv),0)
                self.assertEqual(process.call_count,1)
                self.assertEqual(sweep.main(argv),0)
                self.assertEqual(process.call_count,1)
                self.assertEqual(sweep.main(['--summarize-only','--output-root',str(root/'runs')]),0)
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    sweep.main(argv+['--densities','.25'])
            def fail(cmd,**kwargs):
                kwargs['stdout'].write('test failure\n')
                return SimpleNamespace(returncode=1)
            with patch.object(sweep.subprocess,'run',side_effect=fail), \
                    patch.object(sweep,'plot_results'),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sweep.main(argv+['--no-resume']),1)


if __name__=='__main__':unittest.main()
