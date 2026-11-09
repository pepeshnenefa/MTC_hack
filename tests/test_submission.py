import csv,importlib.util,json,tempfile,threading,unittest,subprocess,sys,os,zipfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
spec=importlib.util.spec_from_file_location('solution',ROOT/'submission/solution.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_full_document(self):
  items=m.chunks([('a.md','# Title\n'+'intro '*200+'\n## Монтаж\nПанели устанавливаются на фасадах.')])
  self.assertIn('фасадах',m.Search(items).find('фасадах')[0]['text'])
 def test_dialogue(self):
  q=m.query('Пользователь: три склада\nАссистент: неверный факт\nПользователь: какие приборы?')
  self.assertIn('три склада',q);self.assertNotIn('неверный',q)
 def test_csv_mock_api(self):
  class Handler(BaseHTTPRequestHandler):
   def do_POST(self):
    data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
    assert data['messages'][-1]['content']
    body=json.dumps({'choices':[{'message':{'content':'Ответ, с "кавычками"\nи переносом'}}]}).encode()
    self.send_response(200);self.end_headers();self.wfile.write(body)
   def log_message(self,*args):pass
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with tempfile.TemporaryDirectory() as d:
    d=Path(d)
    with (d/'questions.csv').open('w',encoding='utf-8-sig',newline='') as f:
     w=csv.writer(f);w.writerow(['id','question']);w.writerows([['001','Панели?'],['x,2','Пользователь: вопрос\nПользователь: фасад?']])
    with zipfile.ZipFile(d/'kb.zip','w') as z:z.writestr('a.md','# Панели\nМонтаж на фасаде.')
    env=dict(os.environ,DATA_DIR=str(d),LLM_BASE_URL='http://127.0.0.1:'+str(server.server_port),LLM_API_KEY='test-only',LLM_MODEL='test-model',USE_RERANKER='0')
    subprocess.run([sys.executable,str(ROOT/'submission/solution.py')],cwd=d,env=env,check=True,capture_output=True)
    with (d/'submission.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    self.assertEqual([r['id'] for r in rows],['001','x,2']);self.assertEqual(rows[0]['answer'],'Ответ, с "кавычками" и переносом')
  finally:server.shutdown();server.server_close()
if __name__=='__main__':unittest.main()
