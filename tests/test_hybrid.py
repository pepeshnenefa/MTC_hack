import sys,time,unittest,math,os,tempfile
from unittest.mock import patch
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'submission'))
from hybrid import embed,prepare,HybridSearch,vector
from retrieval import Search,chunks

class HybridTests(unittest.TestCase):
 def data(self):return Search(chunks([('a.md','# A\nВертикальное размещение.'),('b.md','# B\nДругое описание.')]))
 def test_reordered_embedding_response(self):
  class API:
   def post(self,endpoint,body):return {'data':[{'index':1,'embedding':[0,2]},{'index':0,'embedding':[3,0]}]}
  self.assertEqual(embed(API(),'model',['a','b']),[[1.,0.],[0.,1.]])
 def test_invalid_vectors(self):
  for v in [[],[0,0],[float('nan'),1],[float('inf'),1]]:
   with self.assertRaises(ValueError):vector(v)
 def test_missing_or_duplicate_indices_rejected(self):
  for data in [[],[{'index':0,'embedding':[1]}]*2]:
   class API:
    def post(self,*args):return {'data':data}
   with self.assertRaises(ValueError):embed(API(),'m',['a','b'])
 def test_dense_finds_no_word_overlap_with_supplied_vectors(self):
  lex=self.data();q='Стенка для модулей?'
  h=HybridSearch(lex,[[1,0],[0,1]],{q:[1,0]})
  self.assertEqual(h.find(q,1)[0]['source'],'a.md')
 def test_unknown_query_uses_lexical(self):
  lex=self.data();h=HybridSearch(lex,[[1,0],[0,1]],{})
  self.assertEqual(h.find('Другое'),lex.find('Другое'))
 def test_failure_returns_same_lexical_instance(self):
  class API:
   def post(self,*args):raise TimeoutError('synthetic')
  lex=self.data()
  self.assertIs(prepare(lex,['q'],API(),'m',time.monotonic()+10),lex)
 def test_dimension_mismatch(self):
  with self.assertRaises(ValueError):HybridSearch(self.data(),[[1,0],[0,1]],{'q':[1]})
 def test_preparation_once_and_queries_deduplicated(self):
  calls=[]
  class API:
   def post(self,endpoint,body):
    calls.append(body)
    return {'data':[{'index':i,'embedding':[1,i+1]} for i,_ in enumerate(body['input'])]}
  h=prepare(self.data(),['q','q'],API(),'m',time.monotonic()+10)
  self.assertIsInstance(h,HybridSearch);self.assertEqual(len(calls),2)
  h.find('q');h.find('q');self.assertEqual(len(calls),2)
 def test_expired_budget_skips_requests(self):
  class API:
   def post(self,*args):raise AssertionError('Must not call')
  lex=self.data();self.assertIs(prepare(lex,['q'],API(),'m',time.monotonic()-1),lex)

class EntryPointTests(unittest.TestCase):
 def test_worker_initializes_hybrid_before_generation(self):
  import solution
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'questions.csv').write_text('id,question\n1,Вопрос\n',encoding='utf-8')
   kb=root/'kb';kb.mkdir();(kb/'a.md').write_text('# Заголовок\nТекст',encoding='utf-8')
   config={'LLM_BASE_URL':'http://synthetic.invalid','LLM_API_KEY':'test','LLM_MODEL':'test','USE_HYBRID':'1'}
   def prepared(search,questions,api,model,deadline,**kwargs):return search
   response={'choices':[{'message':{'content':'Ответ'},'finish_reason':'stop'}]}
   with patch.dict(os.environ,config,clear=True),patch.object(sys,'argv',['solution.py','--worker','--questions',str(root/'questions.csv'),'--kb',str(kb),'--output',str(root/'out.csv')]),patch.object(solution,'prepare_hybrid',side_effect=prepared) as prepare_call,patch.object(solution.API,'post',return_value=response):
    solution.main()
    self.assertEqual(prepare_call.call_count,1)
    self.assertEqual(prepare_call.call_args.args[1],['Вопрос'])

if __name__=='__main__':unittest.main()

