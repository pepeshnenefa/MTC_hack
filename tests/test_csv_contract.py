import csv
import importlib.util
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
sys.path.insert(0,str(ROOT/'tools'))
from batch_io import InputError, read_questions, write_answers, validate_answers, run_batch
from build_submission import build, FILES
from check_submission import check
import solution

class CSVContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def input(self,text):
        path=self.root/'questions.csv';path.write_text(text,encoding='utf-8');return path
    def test_bom_multiline_quotes_order_and_text_ids(self):
        p=self.input('\ufeffid,question\n001,"Пользователь: цех\nПользователь: ""сколько"", почему?"\nx-2,Еще вопрос\n')
        rows=read_questions(p)
        self.assertEqual(rows[0]['id'],'001');self.assertIn('\n',rows[0]['question'])
        out=self.root/'output.csv';write_answers(out,rows,['Да, "можно".\r\nДальше.','Ответ'])
        self.assertEqual(validate_answers(out,rows),2)
        self.assertEqual(out.read_text(encoding='utf-8').count('\n'),3)
    def test_bad_rows_rejected(self):
        for text in ['id;question\n1;вопрос','id,id,question\n1,1,вопрос',
                     'id,question\n1,','id,question\n,вопрос','id,question\n1,вопрос,лишнее',
                     'id,question\n1','id,question\n1,"незакрыто']:
            with self.subTest(text=text), self.assertRaises(InputError):read_questions(self.input(text))
    def test_empty_batch(self):
        rows=read_questions(self.input('id,question\n'))
        answers,errors=run_batch(rows,lambda q:self.fail('No calls for empty input'))
        out=self.root/'out.csv';write_answers(out,rows,answers)
        self.assertEqual(validate_answers(out,rows),0)
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(InputError):
            read_questions(self.input('id,question\n1,Первый\n1,Второй\n'))
        rows=[{'id':'1'},{'id':'1'}]
        with self.assertRaises(ValueError):write_answers(self.root/'out.csv',rows,['А','Б'])
        with self.assertRaises(ValueError):validate_answers(self.input('id,answer\n1,A\n1,B\n'),rows)
    def test_answer_length_boundaries(self):
        for length in [9999,10000]:
            out=self.root/'out.csv';write_answers(out,[{'id':'1'}],['я'*length])
            self.assertEqual(validate_answers(out,[{'id':'1'}]),1)
        with self.assertRaises(ValueError):write_answers(self.root/'out.csv',[{'id':'1'}],['я'*10001])
        answers,errors=run_batch([{'id':'1','question':'q'}],lambda q:'я'*10001)
        self.assertEqual(errors,['ValueError']);self.assertLess(len(answers[0]),10000)
    def test_failure_isolated_and_parallel_order(self):
        rows=[{'id':str(i),'question':q} for i,q in enumerate(['slow','bad','fast'])]
        seen=[]
        def answer(q):
            seen.append(q)
            if q=='slow':time.sleep(.03)
            if q=='bad':raise RuntimeError('private detail')
            return q
        answers,errors=run_batch(rows,answer,3)
        self.assertEqual(answers[0],'slow');self.assertEqual(answers[2],'fast')
        self.assertEqual(errors,[None,'RuntimeError',None]);self.assertCountEqual(seen,['slow','bad','fast'])
        self.assertNotIn('private detail',answers[1])
    def test_invalid_answers_do_not_replace_existing_file(self):
        out=self.root/'out.csv';out.write_text('old',encoding='utf-8')
        for answers in [[],[''],[None]]:
            with self.assertRaises(ValueError):write_answers(out,[{'id':'1'}],answers)
        self.assertEqual(out.read_text(),'old')
    def test_atomic_replace_failure_keeps_previous_output(self):
        out=self.root/'out.csv';out.write_text('old')
        with patch('batch_io.os.replace',side_effect=OSError('blocked')):
            with self.assertRaises(OSError):write_answers(out,[{'id':'1'}],['answer'])
        self.assertEqual(out.read_text(),'old');self.assertEqual(list(self.root.glob('*.tmp')),[])
    def test_output_validator_catches_wrong_ids(self):
        out=self.input('id,answer\nwrong,answer\n')
        with self.assertRaises(ValueError):validate_answers(out,[{'id':'001'}])
    def test_relative_data_dir_next_to_script(self):
        script=self.root/'app'/'solution.py';data=script.parent/'data';data.mkdir(parents=True)
        (data/'questions.csv').write_text('id,question\n')
        with patch.object(solution,'__file__',str(script)),patch.dict('os.environ',{'DATA_DIR':'data'}):
            self.assertEqual(solution.locate(['questions.csv']).resolve(),(data/'questions.csv').resolve())
    def test_package_reproducible_allowlist_and_extracted_run(self):
        first=self.root/'a.zip';second=self.root/'b.zip';build(first);build(second)
        self.assertEqual(first.read_bytes(),second.read_bytes())
        with zipfile.ZipFile(first) as z:self.assertEqual(set(z.namelist()),set(FILES))
        questions=self.input('id,question\n001,Есть панели?\n')
        kb=self.root/'kb';kb.mkdir();(kb/'one.md').write_text('# Панели\nЕсть панели.',encoding='utf-8')
        result=check(first,questions,kb,self.root/'result')
        self.assertTrue(result['passed']);self.assertFalse(result['model_quality_evaluated'])

if __name__=='__main__':unittest.main()
