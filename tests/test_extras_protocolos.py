import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experimentos_extras_protocolos as grid


class ProtocolGridTests(unittest.TestCase):
    def test_default_design(self):
        args = grid.parser().parse_args([])
        args.work_root = Path('/tmp/grid-work')
        jobs = [j for w in args.ways for s in args.shots for q in args.queries
                for j in grid.base.jobs(grid.base.parser().parse_args(grid.child_args(args,w,s,q)))]
        self.assertEqual(len(jobs),4740)
        self.assertEqual(len({j['directory'] for j in jobs}),4740)
        self.assertEqual(sum(j['extra_mode']=='none' for j in jobs),420)
        self.assertEqual({j['extra_phase'] for j in jobs}, {'none','train','train-val','all'})
        self.assertTrue(all(j['extras_total']==j['extras_per_class']*j['n_way'] for j in jobs))

    def test_dry_run_and_invalid_capacity_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            out = Path(tmp)/'new'
            self.assertEqual(grid.main(['--dry-run','--output-root',str(out)]),0)
            self.assertFalse(out.exists())
            with self.assertRaises(SystemExit):
                grid.main(['--shots','20','--output-root',str(out)])
            self.assertFalse(out.exists())

    def test_summary_keeps_protocols_and_controls_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs=[]
            for way,mode,acc in [(2,'none',90),(2,'propagation',91),(5,'none',70),(5,'propagation',68)]:
                folder=root/f'{way}-{mode}'
                folder.mkdir()
                (folder/'metrics.json').write_text(json.dumps({'score':acc}))
                jobs.append(dict(directory=str(folder),dataset='bacteria_micro',seed=7,metodo='knn-in',alvo='-',
                                 n_way=way,n_shot=10,n_query=10,extra_mode=mode,extra_phase='train' if mode!='none' else 'none',
                                 extras_per_class=5 if mode!='none' else 0))
            def flatten(job,result):
                return dict(job,val_accuracy=result['score'],test_accuracy=result['score'],
                            test_macro_f1=result['score'],test_balanced_accuracy=result['score'])
            with patch.object(grid.base,'matches',return_value=True),patch.object(grid.base,'flatten',side_effect=flatten):
                grid.summarize(root,jobs)
            with (root/'medias.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4)
            gains={int(r['n_way']):float(r['ganho_accuracy_vs_sem_extras_pp']) for r in rows if r['extra_mode']=='propagation'}
            self.assertEqual(gains,{2:1,5:-2})


if __name__ == '__main__':
    unittest.main()
