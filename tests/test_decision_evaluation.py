import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('decision_evaluation',ROOT/'tools/decision_evaluation.py')
ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)

class DecisionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.original=(ledger.RUNS,ledger.REPORT)
        ledger.RUNS=self.root/'runs';ledger.REPORT=self.root/'report.md'
        self.answers=self.root/'answers.csv';self.artifact=self.root/'artifact.zip'
        self.artifact.write_bytes(b'synthetic-test-only')
        self.rows=[{'id':c['id'],'answer':'Синтетический ответ для проверки учёта.'} for c in ledger.read_suite()]
        self.write_answers()

    def tearDown(self):
        ledger.RUNS,ledger.REPORT=self.original;self.temp.cleanup()

    def write_answers(self):
        with self.answers.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id','answer']);w.writeheader();w.writerows(self.rows)

    def test_suite_sources_and_pairs(self):
        cases=ledger.read_suite()
        self.assertEqual(len({c['id'] for c in cases}),20)
        for pair in {c['pair'] for c in cases}:
            self.assertEqual({c['id'] for c in cases if c['pair']==pair},{pair+'-A',pair+'-B'})
        for c in cases:
            self.assertIn(c['heading'],(ROOT/c['source']).read_text(encoding='utf-8'))

    def test_duplicate_rejected(self):
        self.rows[-1]=self.rows[0];self.write_answers()
        with self.assertRaises(ValueError): ledger.register(self.answers,'test',self.artifact,'duplicate')
        self.assertFalse(ledger.RUNS.exists())

    def test_unreviewed_and_no_overwrite(self):
        ledger.register(self.answers,'test',self.artifact,'test1')
        self.assertIn('| 0/20 | — |',ledger.REPORT.read_text(encoding='utf-8'))
        with self.assertRaises(FileExistsError): ledger.register(self.answers,'test',self.artifact,'test1')

    def test_pair_requires_both_and_critical_error_blocks_pass(self):
        ledger.register(self.answers,'test',self.artifact,'test2')
        path=ledger.RUNS/'test2.json';run=json.loads(path.read_text(encoding='utf-8'))
        run['reviewer']='synthetic unit test'
        for row in run['cases'][:2]:
            row['scores']={k:1 for k in ledger.CRITERIA};row['critical_error']=False
        run['cases'][1]['critical_error']=True
        path.write_text(json.dumps(run),encoding='utf-8');ledger.report()
        self.assertIn('| 2/20 | 8/8 | 0/10 |',ledger.REPORT.read_text(encoding='utf-8'))
        run['cases'][1]['critical_error']=False
        path.write_text(json.dumps(run),encoding='utf-8');ledger.report()
        self.assertIn('| 2/20 | 8/8 | 1/10 |',ledger.REPORT.read_text(encoding='utf-8'))

if __name__=='__main__': unittest.main()
