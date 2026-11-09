import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
import io

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
import solution
from batch_io import read_questions,validate_answers

BASE={'LLM_BASE_URL':'http://localhost:9999','LLM_API_KEY':'test-secret','LLM_MODEL':'provided-model'}

class RuntimeTests(unittest.TestCase):
    def test_model_required_and_aliases(self):
        with patch.dict(os.environ,{'LLM_BASE_URL':BASE['LLM_BASE_URL'],'LLM_API_KEY':'test'},clear=True):
            with self.assertRaises(ValueError):solution.API()
        with patch.dict(os.environ,{'OPENAI_BASE_URL':BASE['LLM_BASE_URL'],'OPENAI_API_KEY':'test','GEN_MODEL':'from-platform'},clear=True):
            self.assertEqual(solution.API().model,'from-platform')
    def test_settings_validated(self):
        for value in ['nan','-1','invalid','3']:
            with patch.dict(os.environ,dict(BASE,TEMPERATURE=value),clear=True):
                with self.assertRaises(ValueError):solution.API()
    def test_generation_settings_and_shared_key(self):
        captured=[]
        def transport(request,timeout):
            payload=json.loads(request.data);captured.append((request,payload))
            if request.full_url.endswith('/rerank'):
                return io.BytesIO(b'{"results":[{"index":0}]}')
            return io.BytesIO(json.dumps({'choices':[{'message':{'content':'Ответ'}}]}).encode())
        config=dict(BASE,TEMPERATURE='0.25',MAX_TOKENS='777',USE_RERANKER='1',RERANKER_MODEL='rank-platform',RERANKER_API_KEY='must-not-be-used')
        with patch.dict(os.environ,config,clear=True),patch.object(solution,'urlopen',side_effect=transport):
            self.assertEqual(solution.API().answer('question',[{'source':'a','heading':'h','text':'text'}]),'Ответ')
        self.assertEqual(len(captured),2)
        self.assertTrue(all(request.get_header('Authorization')=='Bearer test-secret' for request,_ in captured))
        self.assertEqual(captured[-1][1]['model'],'provided-model')
        self.assertEqual(captured[-1][1]['temperature'],.25)
        self.assertEqual(captured[-1][1]['max_tokens'],777)
    def test_retries_limited_and_401_not_retried(self):
        for status,expected in [(429,3),(503,3),(401,1)]:
            error=HTTPError('http://localhost',status,'error',{},None)
            with patch.dict(os.environ,BASE,clear=True),patch.object(solution,'urlopen',side_effect=error) as call,patch.object(solution.time,'sleep'):
                with self.assertRaises(HTTPError):solution.API().post('/chat/completions',{})
                self.assertEqual(call.call_count,expected)
    def test_expired_deadline_does_not_call_proxy(self):
        with patch.dict(os.environ,BASE,clear=True),patch.object(solution,'urlopen') as call:
            with self.assertRaises(TimeoutError):solution.API(deadline=time.monotonic()-1).post('/chat/completions',{})
            call.assert_not_called()
    def test_process_deadline_preserves_completed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir()
            (data/'questions.csv').write_text('id,question\n001,fast\n002,slow\n',encoding='utf-8')
            kb=root/'kb';kb.mkdir();(kb/'a.md').write_text('# Test\nfast slow',encoding='utf-8')
            harness=root/'runner.py'
            harness.write_text('import sys,time\nsys.path.insert(0,'+repr(str(ROOT/'submission'))+')\nimport solution\nclass Stub:\n def __init__(self,**kwargs):pass\n def answer(self,q,s):\n  if q=="slow":time.sleep(60)\n  return "Готовый ответ"\nsolution.API=Stub\nsolution.main()\n',encoding='utf-8')
            env=dict(os.environ,DATA_DIR=str(data),MAX_WORKERS='1',RUN_BUDGET_SECONDS='2')
            started=time.monotonic()
            run=subprocess.run([sys.executable,str(harness),'--kb',str(kb)],cwd=root,env=env,
                               timeout=8,capture_output=True,text=True,encoding='utf-8')
            elapsed=time.monotonic()-started
            self.assertLess(elapsed,6);self.assertEqual(run.returncode,2)
            rows=read_questions(data/'questions.csv')
            self.assertEqual(validate_answers(root/'submission.csv',rows),2)
            text=(root/'submission.csv').read_text(encoding='utf-8')
            self.assertIn('Готовый ответ',text);self.assertIn('технической ошибки',text)
            self.assertIn('budget exhausted',run.stderr)

if __name__=='__main__':unittest.main()
