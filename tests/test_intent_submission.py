import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
import intent
spec=importlib.util.spec_from_file_location('intent_builder',ROOT/'tools/build_decision_submission.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

def source(n):return {'source':f'{n}.md','heading':str(n),'text':f'Текст {n}'}

class IntentTests(unittest.TestCase):
    def test_assistant_quotes_rejected(self):
        task={'goal':'Проверка','kind':'fact','client_quotes':['обещаю всё'],'queries':['Условия']}
        with self.assertRaises(ValueError):intent.parse(json.dumps(task),'Пользователь: Можно?\nАссистент: обещаю всё\nПользователь: Точно?')

    def test_fallback_and_original_question(self):
        class Search:
            def find(self,q,k):return [source(1)]
        class API:
            def answer(self,q,s):return q,s
        with patch.object(intent,'plan',return_value=None):
            self.assertEqual(intent.answer(API(),Search(),'исходный',36),('исходный',[source(1)]))

    def test_preserves_original_and_promotes_unpacked_candidate(self):
        original=[source(i) for i in range(20)]
        class Search:
            def find(self,q,k):return original if q=='original' else [original[19]]
        class API:
            def answer(self,q,s):return s
        with patch.object(intent,'plan',return_value={'queries':['новый','новый']}):
            found=intent.answer(API(),Search(),'original',36)
        self.assertEqual(found[:20],original);self.assertEqual(len(found),21)
        self.assertTrue(found[-1]['intent_extra'])
        text=intent.append_evidence('BASE',found)
        self.assertTrue(text.startswith('BASE'));self.assertIn('Текст 19',text)

    def test_extra_budget_never_evicts_original(self):
        base='x'*39990
        self.assertEqual(intent.append_evidence(base,[dict(source(2),intent_extra=True)]),base)

    def test_plan_timeout_and_shared_deadline(self):
        class API:
            deadline=time.monotonic()+100;model='synthetic';temperature=0;max_tokens=1000
            def post(self,*args):raise TimeoutError('synthetic')
        api=API();deadline=api.deadline
        self.assertIsNone(intent.plan(api,'q'));self.assertEqual(api.deadline,deadline)

    def test_extracted_archive_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'intent.zip'
            base=ROOT/'dist/energytool-context-v1.zip'
            builder.build(base,archive,'intent')
            with zipfile.ZipFile(base) as before,zipfile.ZipFile(archive) as after:
                for n in before.namelist():
                    if n!='solution.py':self.assertEqual(before.read(n),after.read(n))
                after.extractall(root)
            harness=root/'smoke.py'
            harness.write_text('''import os,json,sys
os.environ.clear()
os.environ.update(LLM_BASE_URL='http://invalid',LLM_API_KEY='synthetic',LLM_MODEL='synthetic')
import solution
calls=[]
def post(self,endpoint,body):
    calls.append(body)
    if 'client_quotes' in body['messages'][0]['content']:
        text=json.dumps({'goal':'Понять вопрос','kind':'fact','client_quotes':[], 'queries':['Условия применения']})
    else:text='Синтетический ответ для проверки исполнения.'
    return {'choices':[{'message':{'content':text},'finish_reason':'stop'}]}
solution.API.post=post
sys.argv=['solution.py','--worker','--questions',sys.argv[1],'--kb',sys.argv[2],'--output',sys.argv[3]]
solution.main()
assert len(calls)==40,len(calls)
print('20 rows, 40 synthetic calls')
''',encoding='utf-8')
            run=subprocess.run([sys.executable,str(harness),str(ROOT/'tests/evaluation/decision-v1/questions.csv'),str(ROOT/'Knowledge-base'),str(root/'answers.csv')],capture_output=True,text=True,encoding='utf-8',timeout=90)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('20 rows, 40',run.stdout)

if __name__=='__main__':unittest.main()
