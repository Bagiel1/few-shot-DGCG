import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experimentos_escalas as grid
from proto_sgc.config import parse_args


class ScaleTests(unittest.TestCase):
    def test_plan_and_real_cli(self):
        args=grid.parser().parse_args([])
        selected=grid.select_micro(args)
        self.assertEqual([selected[('flowers',5,s,10)]['alvo'] for s in [1,5]],['0.125','0.05'])
        plan=list(grid.jobs(args,selected))
        self.assertEqual(len(plan),108)
        self.assertEqual(len({j['directory'] for j in plan}),108)
        for j in plan:
            with patch('sys.argv',grid.command(j,args)[1:]):a=vars(parse_args())
            self.assertTrue(grid.matches(j,{'args':a}))
            self.assertIsNone(a['extra_images'])
            self.assertEqual(sum(a['class_split']),20 if j['version']=='micro' else 102 if j['family']=='flowers' else 196)
            a['seed']+=1
            self.assertFalse(grid.matches(j,{'args':a}))

    def test_selection_ignores_test_and_rejects_missing_candidate(self):
        args=grid.parser().parse_args([]);expected=grid.select_micro(args)
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for source in grid.SOURCES:
                with source.open() as f:rows=list(csv.DictReader(f))
                for i,r in enumerate(rows):r['test_accuracy']=str(1000000-i)
                p=Path(tmp)/source.name
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
                paths.append(p)
            with patch.object(grid,'SOURCES',paths):self.assertEqual(grid.select_micro(args),expected)
            paths[1].write_text(paths[1].read_text().splitlines()[0]+'\n')
            with patch.object(grid,'SOURCES',paths),self.assertRaises(ValueError):grid.select_micro(args)

    def test_dry_run_no_write(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp)/'new'
            self.assertEqual(grid.main(['--dry-run','--output-root',str(out)]),0)
            self.assertFalse(out.exists())

    def test_summary_separates_versions_and_seed_std(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);args=grid.parser().parse_args(['--datasets','flowers','--shots','1','--seeds','17','27','--methods','no-graph'])
            args.output_root=root;plan=list(grid.jobs(args,grid.select_micro(args)))
            for j in plan:
                folder=Path(j['directory']);folder.mkdir(parents=True)
                with patch('sys.argv',grid.command(j,args)[1:]):a=vars(parse_args())
                a={k:str(v) if isinstance(v,Path) else v for k,v in a.items()}
                metric={k:(.7 if j['seed']==17 else .9) for k in grid.base.METRICS};metric['graph']={}
                (folder/'metrics.json').write_text(json.dumps({'args':a,'validation':metric,'test':metric}))
            grid.summarize(root,plan)
            with (root/'medias.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),3)
            for r in rows:
                self.assertEqual(float(r['test_accuracy_mean']),80)
                self.assertAlmostEqual(float(r['test_accuracy_std_seed']),14.14213562)
                self.assertEqual(r['concluidos'],'2')

if __name__=='__main__':unittest.main()
