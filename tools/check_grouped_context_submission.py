"""ZIP parent/worker smoke test with synthetic embeddings and answers; no network."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
from build_grouped_context_submission import ROOT,OUTPUT,BASE,files


def check():
    expected=files()
    with zipfile.ZipFile(OUTPUT) as z,zipfile.ZipFile(BASE) as base:
        assert z.testzip() is None and len(z.namelist())==10 and set(z.namelist())==set(expected)
        assert all(z.read(n)==b for n,b in expected.items())
        assert [n for n in base.namelist() if z.read(n)!=base.read(n)]==['context.py']
    metadata=json.loads(OUTPUT.with_suffix('.manifest.json').read_text())
    assert hashlib.sha256(OUTPUT.read_bytes()).hexdigest()==metadata['sha256']
    paths=[];started=time.monotonic()
    try:
        for suffix in ('.py','.csv'):
            path=ROOT/'work'/('grouped-check-'+uuid.uuid4().hex+suffix)
            with path.open('x',encoding='utf-8'):pass
            paths.append(path)
        harness,output=paths
        harness.write_text('import sys\nsys.path.insert(0,'+repr(str(OUTPUT))+')\n'+'''
import hashlib, solution
def forbidden(*args, **kwargs): raise AssertionError('Network forbidden')
solution.urlopen=forbidden
def fake(self, endpoint, body):
    if endpoint=='/embeddings':
        return {'data':[{'index':i,'embedding':[x+1 for x in hashlib.sha256(t.encode()).digest()[:8]]} for i,t in enumerate(body['input'])]}
    assert endpoint=='/chat/completions'
    assert '=== Блок ' in body['messages'][1]['content']
    return {'choices':[{'message':{'content':'ТЕСТ ФОРМАТА: ответ.'},'finish_reason':'stop'}]}
solution.API.post=fake
solution.main()
''',encoding='utf-8')
        environment={k:v for k,v in os.environ.items() if not k.startswith('ET_')}
        environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',
            LLM_BASE_URL='http://test.invalid',LLM_API_KEY='local-test',LLM_MODEL='local-test-model',
            USE_HYBRID='1',USE_HIERARCHY='0',USE_RERANKER='0',TOP_K='12',RUN_BUDGET_SECONDS='300',MAX_WORKERS='3')
        run=subprocess.run([sys.executable,'-B',str(harness),'--questions',str(ROOT/'demo_questions.csv'),
            '--kb',str(ROOT/'Knowledge-base'),'--output',str(output)],cwd=ROOT/'work',env=environment,
            capture_output=True,text=True,encoding='utf-8',timeout=130)
        assert run.returncode==0,run.stderr
        assert 'hybrid_ready' in run.stderr,run.stderr
        with (ROOT/'demo_questions.csv').open(encoding='utf-8-sig',newline='') as f:
            ids=[r['id'] for r in csv.DictReader(f)]
        with output.open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f);rows=list(reader);assert reader.fieldnames==['id','answer']
        assert [r['id'] for r in rows]==ids and len(ids)==len(set(ids))
        assert all(r['answer']=='ТЕСТ ФОРМАТА: ответ.' for r in rows)
        return dict(kind='synthetic-not-model-quality',rows=len(rows),hybrid_ready=True,
                    passed=True,seconds=round(time.monotonic()-started,3),sha256=metadata['sha256'])
    finally:
        for path in paths:path.unlink(missing_ok=True)


if __name__=='__main__':print(json.dumps(check(),indent=2))
