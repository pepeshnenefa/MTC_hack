import sys,os,json,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
mode=sys.argv[1]
sys.path.insert(0,str(ROOT/('work/baseline-before-context-v1' if mode=='baseline' else 'submission')))
import solution
from batch_io import read_questions
questions={r['id']:r['question'] for r in read_questions(ROOT/'demo_questions.csv')}
cases=json.loads((ROOT/'tests/retrieval_cases.json').read_text(encoding='utf-8'))
started=time.perf_counter();all_items=solution.chunks(solution.documents(ROOT/'Knowledge-base'));search=solution.Search(all_items)
index_ms=(time.perf_counter()-started)*1000
results=[]
with patch.dict(os.environ,{'LLM_BASE_URL':'http://synthetic.invalid','LLM_API_KEY':'synthetic','LLM_MODEL':'synthetic','TOP_K':'12','USE_RERANKER':'0'},clear=True):
 api=solution.API()
 for case in cases:
  question=case.get('question',questions.get(case['id']));capture=[]
  def post(endpoint,body):
   capture.append(body);return {'choices':[{'message':{'content':'Тест формирования запроса'},'finish_reason':'stop'}]}
  api.post=post
  started=time.perf_counter();found=search.find(question,k=12 if mode=='baseline' else 36)
  api.answer(question,found);prompt=capture[-1]['messages'][-1]['content']
  delivered=[c for c in found if c['text'] in prompt and c['source'] in prompt]
  hits=[]
  for g in case['groups']:
   hits.append(any(c['source'].replace('\\','/').endswith(g['source']) and c['heading'] in g['headings'] for c in delivered))
  results.append({'id':case['id'],'groups_hit':hits,'delivered':len(delivered),'prompt_characters':len(prompt),'latency_ms':round((time.perf_counter()-started)*1000,2)})
summary={'mode':mode,'index_ms':round(index_ms,2),'groups_hit':sum(sum(r['groups_hit']) for r in results),'groups_total':sum(len(r['groups_hit']) for r in results),'complete_cases':sum(all(r['groups_hit']) for r in results),'cases_total':len(results),'misses':[r for r in results if not all(r['groups_hit'])],'cases':results}
(ROOT/'work'/('context-comparison-'+mode+'.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='cases'},ensure_ascii=False))
