import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
import decision
spec=importlib.util.spec_from_file_location('decision_builder',ROOT/'tools/build_decision_submission.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

def reply(data,reason='stop'):
    return {'choices':[{'message':{'content':json.dumps(data,ensure_ascii=False)},'finish_reason':reason}]}

class FakeAPI:
    def __init__(self,result):self.result=result;self.deadline=time.monotonic()+100;self.calls=[]
    def post(self,endpoint,body):
        self.calls.append(body)
        if isinstance(self.result,Exception):raise self.result
        return self.result

class DecisionSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.data={'client_facts':['500 кВт'],'checks':[{'condition':'Минимум','status':'unmet','evidence':'От 750 кВт'}],'action':'answer','missing':''}
        self.payload={'model':'synthetic','max_tokens':2048,'messages':[{'role':'system','content':'rules'},{'role':'user','content':'original evidence and dialogue'}]}

    def test_evidence_preserved_and_shared_deadline_unchanged(self):
        api=FakeAPI(reply(self.data));deadline=api.deadline
        result=decision.prepare(api,self.payload,'От 750 кВт','У нас 500 кВт')
        self.assertEqual(result['messages'][1],self.payload['messages'][1])
        self.assertEqual(api.deadline,deadline)
        self.assertEqual(len(self.payload['messages']),2)
        self.assertEqual(len(result['messages']),3)

    def test_reject_fabricated_quotes_and_truncation(self):
        self.data['checks'][0]['evidence']='От 100 кВт'
        for response in [reply(self.data),reply(self.data,'length'),TimeoutError('synthetic')]:
            self.assertIs(decision.prepare(FakeAPI(response),self.payload,'От 750 кВт','500 кВт'),self.payload)

    def test_deadline_skips_planner(self):
        api=FakeAPI(reply(self.data));api.deadline=time.monotonic()+15
        self.assertIs(decision.prepare(api,self.payload,'От 750 кВт','500 кВт'),self.payload)
        self.assertEqual(api.calls,[])

    def test_both_archives_real_pipeline_with_synthetic_transport(self):
        for name,mode in (('context','decision'),('hierarchy','decision'),('context','decision_v2')):
            with self.subTest(base=name,mode=mode),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);archive=root/'candidate.zip'
                base=ROOT/f'dist/energytool-{name}-v1.zip'
                builder.build(base,archive,mode)
                with zipfile.ZipFile(base) as before,zipfile.ZipFile(archive) as after:
                    for entry in before.namelist():
                        if entry!='solution.py':self.assertEqual(before.read(entry),after.read(entry))
                    self.assertLess(archive.stat().st_size,128_000_000)
                    after.extractall(root)
                harness=root/'smoke.py'
                harness.write_text('''import os,json,sys
from pathlib import Path
os.environ.clear()
os.environ.update(LLM_BASE_URL='http://invalid',LLM_API_KEY='synthetic',LLM_MODEL='synthetic')
import solution
calls=[]
def post(self,endpoint,body):
    calls.append(body)
    if 'client_facts' in body['messages'][0]['content']:
        text=json.dumps({'client_facts':[],'checks':[],'action':'clarify','missing':'Уточните данные'})
    else:text='Синтетическая проверка, не оценка качества.'
    return {'choices':[{'message':{'content':text},'finish_reason':'stop'}]}
solution.API.post=post
sys.argv=['solution.py','--worker','--questions',sys.argv[1],'--kb',sys.argv[2],'--output',sys.argv[3]]
solution.main()
assert len(calls)==40,len(calls)
print('40 logical calls / 20 rows; synthetic only')
''',encoding='utf-8')
                run=subprocess.run([sys.executable,str(harness),str(ROOT/'tests/evaluation/decision-v1/questions.csv'),str(ROOT/'Knowledge-base'),str(root/'answers.csv')],capture_output=True,text=True,encoding='utf-8',timeout=90)
                self.assertEqual(run.returncode,0,run.stderr)
                self.assertIn('40 logical calls',run.stdout)

if __name__=='__main__':unittest.main()
