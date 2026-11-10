"""Exercise the immutable submitted ZIP through a real local HTTP endpoint."""
import ast
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'dist/energytool-context-decision-v1.zip'
OUT=ROOT/'work/decision-http-audit'

def run_case(mode):
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            planner='client_facts' in body['messages'][0]['content']
            stage='planner' if planner else 'answer'
            status=200;reason='stop';content='Проверочный ответ, не оценка качества.'
            if planner:
                content=json.dumps({'client_facts':[],'checks':[],'action':'answer','missing':''})
            if mode=='invalid_plan' and planner:content='not JSON'
            if mode=='null_content' and not planner:content=None;reason='length'
            if mode in ('null_once','strict_recovery') and not planner and not any(c['stage']=='answer' for c in calls):content=None;reason='length'
            if mode=='empty_content' and not planner:
                if sum(c['stage']=='answer' for c in calls)==0:content='';reason='length'
            if mode=='partial_failure' and not planner and body['messages'][1]['content'].split('\n\nСЛУЖЕБНАЯ')[0].endswith('broken'):
                content=None;reason='length'
            if mode=='auth_error':status=401
            if mode=='planner_503' and planner:status=503
            if mode in ('strict_roles','strict_recovery') and any(a['role']==b['role']=='user' for a,b in zip(body['messages'],body['messages'][1:])):status=400
            calls.append({'stage':stage,'status':status,'roles':[m['role'] for m in body['messages']],
                          'max_tokens':body['max_tokens']})
            if mode=='slow_planner' and planner:time.sleep(26)
            if mode=='deadline':time.sleep(4)
            response={'choices':[{'message':{'content':content},'finish_reason':reason}]} if status==200 else {'error':{'message':'synthetic error'}}
            if mode=='missing_choices' and not planner and sum(c['stage']=='answer' for c in calls)==1:response={'choices':[]}
            raw=json.dumps(response,ensure_ascii=False).encode('utf-8')
            if mode=='invalid_response' and not planner and sum(c['stage']=='answer' for c in calls)==1:raw=b'{not-json'
            try:
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with zipfile.ZipFile(ARCHIVE) as z:z.extractall(root)
            data=root/'data';data.mkdir()
            questions=[('001','normal'),('002','broken')] if mode=='partial_failure' else [('001','normal')]
            with (data/'questions.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.writer(f);w.writerow(['id','question']);w.writerows(questions)
            with zipfile.ZipFile(data/'knowledge-base.zip','w') as z:z.writestr('test.md','# Test\nnormal broken')
            environment={k:v for k,v in os.environ.items() if k in ('SYSTEMROOT','WINDIR','PATH','TEMP','TMP','COMSPEC')}
            environment.update(LLM_BASE_URL=f'http://127.0.0.1:{server.server_port}',LLM_API_KEY='synthetic',
                LLM_MODEL='synthetic',RUN_BUDGET_SECONDS='2' if mode=='deadline' else '45',
                MAX_WORKERS='1',PYTHONIOENCODING='utf-8',NO_PROXY='127.0.0.1,localhost')
            started=time.monotonic()
            result=subprocess.run([sys.executable,'solution.py'],cwd=root,env=environment,
                                  capture_output=True,text=True,encoding='utf-8',timeout=55)
            path=root/'submission.csv'
            rows=list(csv.reader(path.open(encoding='utf-8',newline=''))) if path.exists() else []
            valid=(bool(rows) and rows[0]==['id','answer'] and len(rows)==len(questions)+1
                   and all(len(r)==2 and r[0]==q[0] and 0<len(r[1])<=10000 for r,q in zip(rows[1:],questions)))
            return {'mode':mode,'exit_code':result.returncode,'seconds':round(time.monotonic()-started,2),
                    'csv_contract_valid':valid,'rows':rows[1:],'calls':calls,
                    'stdout':result.stdout.strip(),'stderr':result.stderr.strip()}
    finally:server.shutdown();server.server_close()

def main():
    global ARCHIVE, OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('case',nargs='?')
    parser.add_argument('--archive',type=Path,default=ARCHIVE)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--expect-v2',action='store_true')
    args=parser.parse_args();ARCHIVE=args.archive;OUT=args.output_dir
    OUT.mkdir(parents=True,exist_ok=True)
    if args.case:
        report=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
        result=run_case(args.case)
        report['results']=[result if r['mode']==args.case else r for r in report['results']]
        (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=True));return
    with zipfile.ZipFile(ARCHIVE) as z:
        source=z.read('solution.py').decode('utf-8')
        system=next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets))
        metadata={'sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),'zip_integrity':z.testzip(),
                  'files':z.namelist(),'replacement_characters_in_system':system.count('\ufffd'),
                  'system_starts_correctly':system.startswith('Ты консультант')}
    results=[]
    for mode in ['normal','invalid_plan','empty_content','null_once','null_content','partial_failure','auth_error','planner_503','strict_roles','strict_recovery','missing_choices','invalid_response','deadline','slow_planner']:
        result=run_case(mode);results.append(result)
        print(json.dumps({k:result[k] for k in ('mode','exit_code','seconds','csv_contract_valid')}) ,flush=True)
        (OUT/'results.json').write_text(json.dumps({'archive':metadata,'results':results},ensure_ascii=False,indent=2),encoding='utf-8')
    if args.expect_v2:
        by={r['mode']:r for r in results}
        assert all(r['csv_contract_valid'] for r in results)
        for mode in ('normal','invalid_plan','empty_content','null_once','strict_roles','strict_recovery','missing_choices','invalid_response','planner_503','slow_planner'):
            assert by[mode]['exit_code']==0,mode
            assert all('технической ошибки' not in row[1] for row in by[mode]['rows']),mode
        assert len(by['auth_error']['calls'])==1
        assert len([c for c in by['planner_503']['calls'] if c['stage']=='planner'])==1
        assert len([c for c in by['null_content']['calls'] if c['stage']=='answer'])==2
        assert 'degraded' in by['partial_failure']['stdout']
        assert by['null_content']['exit_code']==by['auth_error']['exit_code']==by['deadline']['exit_code']==2
        assert by['slow_planner']['seconds']<20
        print('All v2 HTTP regression assertions passed.')

if __name__=='__main__':main()
