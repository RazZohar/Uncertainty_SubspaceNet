"""Orchestration/export tests with explicit fake predictors; no neural claims."""
from pathlib import Path
import copy
import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_source_sweep_comparison as runner
from inter_source_tools.comparison import resolve_targets

class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        cfg={'L':1,'subarray_config':[{'system_model':dict(N=8,M=2,T=100,snr=10,signal_type='NarrowBand',signal_nature='non-coherent',codebook_size=128)}]}
        self.entries=[]
        for model in ['multi_subarray','transmusic','data_driven_complex']:
            folder=self.root/model;folder.mkdir();cp=folder/'best_base_reliability.pth';cp.write_bytes(b'INPUT-PATH-TEST-ONLY')
            config=folder/'config.json';config.write_text(json.dumps(cfg))
            self.entries.append(dict(model_type=model,checkpoint_path=str(cp),config_path=str(config)))
        self.models=self.root/'models.json';self.models.write_text(json.dumps({'models':self.entries}))
        self.argv=['--models_json',str(self.models),'--moving_values','35','45','--trials','8','--results_dir',str(self.root/'results')]
    def tearDown(self):self.tmp.cleanup()
    def test_preflight_and_mismatch_rejection(self):
        args=runner.parse_args(self.argv);targets,cfg=resolve_targets(args)
        self.assertEqual(len(targets),4);self.assertEqual(targets[-1]['model_type'],'esprit')
        p=Path(self.entries[1]['config_path']);other=json.loads(p.read_text());other['subarray_config'][0]['system_model']['snr']=3;p.write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError,'simulator settings differ'):resolve_targets(args)
        args.simulation_config=Path(self.entries[0]['config_path']);targets,_=resolve_targets(args)
        self.assertEqual(targets[1]['evaluation_shift_fields'],['snr'])
        other['subarray_config'][0]['system_model']['M']=3;p.write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError,'M=2'):resolve_targets(args)
    def test_same_inputs_csvs_and_resume(self):
        observed={};loaded=[];generated=[];self_root=self.root
        original={e['checkpoint_path']:Path(e['checkpoint_path']).read_bytes() for e in self.entries}
        class FakePredictor:
            def __init__(self,target,config,config_path,args,device):
                self.label=target['label'];loaded.append(self.label)
            def __call__(self,x):
                observed.setdefault(int(x[0,0,0].real),[]).append(x.copy())
                truth=np.deg2rad([40,float(x[0,0,0].real)])
                error=np.arange(len(x))[:,None]*np.array([[.0001,-.0001]])
                return truth+error,np.tile(np.array([[.001,.0002],[.0002,.001]]),(len(x),1,1))
        def generate(case,system,trials):
            generated.append(case['case_id'])
            return np.full((trials,8,100),case['theta_moving_deg'],dtype=np.complex64),{}
        fake_torch=types.SimpleNamespace(__version__='SOFTWARE_TEST',cuda=types.SimpleNamespace(is_available=lambda:False))
        with patch.dict(sys.modules,{'torch':fake_torch}),patch('inter_source_tools.native.NativePredictor',FakePredictor),patch('inter_source_tools.native.generate_case',generate),patch('inter_source_tools.comparison.export_all'):
            runner.main(self.argv)
            self.assertEqual(len(loaded),4);self.assertEqual(len(generated),2)
            for batches in observed.values():
                self.assertEqual(len(batches),4)
                for x in batches[1:]:np.testing.assert_array_equal(x,batches[0])
            with (self.root/'results/benchmark_results.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),8)
            files=list((self.root/'results/trial_results').rglob('*.csv'));self.assertEqual(len(files),8)
            for p in files:
                with p.open() as f:trials=list(csv.DictReader(f))
                self.assertEqual(len(trials),8)
                self.assertAlmostEqual(float(trials[0]['theta_anchor_deg']),40)
                self.assertEqual(trials[0]['valid_covariance'],'1')
            runner.main(self.argv+['--resume'])
            self.assertEqual(len(generated),2)
        for path,data in original.items():self.assertEqual(Path(path).read_bytes(),data)

if __name__=='__main__':unittest.main()
