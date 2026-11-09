"""Offline evidence-section coverage, not an LLM score. Fixtures never ship."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
from batch_io import read_questions

def evaluate(module_path,output):
    spec=importlib.util.spec_from_file_location('retrieval_under_test',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    started=time.perf_counter();items=module.chunks(module.documents(ROOT/'Knowledge-base'));search=module.Search(items)
    indexing_ms=round((time.perf_counter()-started)*1000,2)
    questions={r['id']:r['question'] for r in read_questions(ROOT/'demo_questions.csv')}
    cases=json.loads((ROOT/'tests/retrieval_cases.json').read_text(encoding='utf-8'))
    details=[]
    for case in cases:
        question=case.get('question',questions.get(case['id']))
        started=time.perf_counter();found=search.find(question,k=12);ms=(time.perf_counter()-started)*1000
        ranks=[]
        for group in case['groups']:
            ranks.append(next((i+1 for i,c in enumerate(found) if c['source'].replace('\\','/').endswith(group['source']) and c['heading'] in group['headings']),None))
        details.append({'id':case['id'],'question':question,'ranks':ranks,'latency_ms':round(ms,2),
                        'top':[{'source':c['source'],'heading':c['heading']} for c in found]})
    metrics={}
    for k in [3,6,12]:
        metrics[str(k)]={'groups_hit':sum(r is not None and r<=k for d in details for r in d['ranks']),
                         'groups_total':sum(len(d['ranks']) for d in details),
                         'cases_complete':sum(all(r is not None and r<=k for r in d['ranks']) for d in details),'cases_total':len(details)}
    result={'metric':'manually selected evidence-section coverage, not answer accuracy','indexing_ms':indexing_ms,'sections':len(items),'metrics':metrics,'cases':details,'partitions':{name:{'cases':len(part),'groups_total':sum(len(d['ranks']) for d in part),'groups_hit_12':sum(r is not None for d in part for r in d['ranks']),'complete_12':sum(all(r is not None for r in d['ranks']) for d in part)} for name,part in [('demo',[d for d in details if d['id'].isdigit()]),('additional',[d for d in details if not d['id'].isdigit()])]}}
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'sections':len(items),'indexing_ms':indexing_ms,'metrics':metrics},ensure_ascii=False))
    print('Misses:',[(d['id'],d['ranks']) for d in details if any(r is None for r in d['ranks'])])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--module',default=str(ROOT/'submission/solution.py'));p.add_argument('--output',required=True);a=p.parse_args();evaluate(a.module,a.output)
