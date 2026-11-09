import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'submission'))
import solution
from hierarchy import HierarchicalSearch
from batch_io import read_questions
from context import pack_context
search=HierarchicalSearch(solution.documents(ROOT/'Knowledge-base'))
rows={r['id']:r['question'] for r in read_questions(ROOT/'demo_questions.csv')}
cases=json.loads((ROOT/'tests/retrieval_cases.json').read_text(encoding='utf-8'))
report=[]
for case in cases:
 q=case.get('question',rows.get(case['id']));started=time.perf_counter()
 sources=search.find(q,36);context,packing=pack_context(q,sources,12)
 hits=[any(c['source'].endswith(g['source']) and c['heading'] in g['headings'] and c['text'] in context for c in search.items) for g in case['groups']]
 report.append({'id':case['id'],'routed':search.route(q)[0],'hits':hits,'chars':len(context),'latency_ms':round((time.perf_counter()-started)*1000,1),'dropped':len(packing['dropped'])})
result={'groups_hit':sum(sum(r['hits']) for r in report),'groups_total':sum(len(r['hits']) for r in report),'max_chars':max(r['chars'] for r in report),'cases':report}
(ROOT/'work/hierarchy-evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
