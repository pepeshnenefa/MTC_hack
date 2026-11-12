"""Run the submitted ZIP parent/worker against a local synthetic HTTP server."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from build_structured_query_submission import files

ROOT=Path(__file__).resolve().parents[1]


def check():
    archive=ROOT/'dist/energytool-context-structured-query-v1.zip'
    expected=files()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==6 and set(z.namelist())==set(expected)
        assert all(z.read(n)==b for n,b in expected.items())
    manifest=json.loads(archive.with_suffix('.manifest.json').read_text())
    assert manifest['sha256']==hashlib.sha256(archive.read_bytes()).hexdigest()
    with (ROOT/'demo_questions.csv').open(encoding='utf-8-sig',newline='') as f:
        ids=[r['id'] for r in csv.DictReader(f)]
    calls=[];mode='normal'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            planning='Восстанови текущую задачу клиента' in body['messages'][0]['content']
            calls.append('plan' if planning else 'answer')
            assert self.headers['Authorization']=='Bearer local-test'
            if planning and mode=='http_failure':
                self.send_response(503);self.send_header('Content-Length','0');self.end_headers();return
            if planning:
                turns=json.loads(body['messages'][1]['content'])
                text=json.dumps(dict(goal=turns[-1]['text'][:250],facts=[],queries=[]))
                if mode=='invalid_json':text='invalid'
                if mode=='null_content':text=None
            else:text='ТЕСТ ФОРМАТА: ответ.'
            raw=json.dumps({'choices':[{'message':{'content':text},'finish_reason':'stop'}]}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    paths=[];reports=[]
    try:
        for suffix in ('.py','.csv'):
            path=ROOT/'work'/('structured-query-check-'+uuid.uuid4().hex+suffix)
            with path.open('x',encoding='utf-8'):pass
            paths.append(path)
        harness,output=paths
        harness.write_text('import sys\nsys.path.insert(0, '+repr(str(archive))+')\nimport solution\nsolution.main()\n',encoding='utf-8')
        environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8',
            'LLM_BASE_URL':'http://127.0.0.1:'+str(server.server_port),'LLM_API_KEY':'local-test',
            'LLM_MODEL':'local-test-model','USE_RERANKER':'0','TOP_K':'12',
            'MAX_WORKERS':'3','RUN_BUDGET_SECONDS':'120'}
        for mode in ('normal','http_failure','invalid_json','null_content'):
            calls.clear();started=time.monotonic()
            run=subprocess.run([sys.executable,'-B',str(harness),'--questions',str(ROOT/'demo_questions.csv'),
                '--kb',str(ROOT/'Knowledge-base'),'--output',str(output)],cwd=ROOT/'work',env=environment,
                capture_output=True,text=True,encoding='utf-8',timeout=130)
            assert run.returncode==0,run.stderr
            with output.open(encoding='utf-8',newline='') as f:
                reader=csv.DictReader(f);rows=list(reader);assert reader.fieldnames==['id','answer']
            assert [r['id'] for r in rows]==ids and len(ids)==len(set(ids))
            assert all(r['answer']=='ТЕСТ ФОРМАТА: ответ.' for r in rows)
            assert calls.count('plan')==len(ids) and calls.count('answer')==len(ids)
            status='ready' if mode=='normal' else 'fallback'
            assert run.stderr.count('"status": "'+status+'"')==len(ids),run.stderr
            reports.append(dict(mode=mode,rows=len(rows),calls=len(calls),
                                seconds=round(time.monotonic()-started,3),passed=True))
    finally:
        server.shutdown();server.server_close();thread.join()
        for path in paths:path.unlink(missing_ok=True)
    return dict(kind='synthetic-http-not-model-quality',sha256=manifest['sha256'],runs=reports)


if __name__=='__main__':print(json.dumps(check(),indent=2))
