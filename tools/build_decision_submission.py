"""Add the same two-stage experiment to an immutable known baseline ZIP."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def build(base, output, mode='decision'):
    base=Path(base);output=Path(output)
    if base.resolve()==output.resolve() or output.exists():
        raise ValueError('Never overwrite an experiment or baseline')
    with zipfile.ZipFile(base) as z:
        files={n:z.read(n) for n in z.namelist()}
        if len(files)!=len(z.namelist()) or any('/' in n or '\\' in n or not(n.endswith('.py') or n=='requirements.txt') for n in files):
            raise ValueError('Invalid baseline entries')
    code=files['solution.py'].decode('utf-8')
    if mode=='decision_v2':
        code=code.replace('\r\n','\n').replace('\r','\n')
        # Normalize old CRCRLF files without changing executable statements.
        while '\n\n\n' in code:code=code.replace('\n\n\n','\n\n')
    if mode=='intent':
        if set(files)!={'solution.py','batch_io.py','retrieval.py','context.py','requirements.txt'}:
            raise ValueError('Intent experiment requires context-v1 baseline')
        old="        context,report=pack_context(question,sources,limit=setting('TOP_K',12,1,100))"
        new="        from intent import append_evidence\n        context,report=pack_context(question,[s for s in sources if not s.get('intent_extra')],limit=setting('TOP_K',12,1,100))\n        context=append_evidence(context,sources)"
        call='lambda question: api.answer(question,search.find(question,k=min(100,max(36,top_k*3))))'
        if code.count(old)!=1 or code.count(call)!=1:raise ValueError('Unexpected baseline')
        code=code.replace(old,new).replace(call,'lambda question: intent_answer(api,search,question,k=min(100,max(36,top_k*3)),top_k=top_k)')
        code='from intent import answer as intent_answer\n'+code
    elif mode=='decision_v2':
        start=code.index('        def extract(result):')
        end=code.index('        needs_recovery=',start)
        code=code[:start]+('        from decision_v2 import prepare, extract, recover_payload, request_answer\n'
            '        payload=prepare(self,payload,context,question)\n'
            '        original,reason=request_answer(self,payload)\n')+code[end:]
        start=code.index('            retry=dict(payload)')
        end=code.index('            try:',start)
        code=code[:start]+'            retry=recover_payload(payload)\n'+code[end:]
        code=code.replace("for attempt in range(3):", "for attempt in range(getattr(self,'max_attempts',3)):")
        code=code.replace('or attempt==2:', "or attempt+1>=getattr(self,'max_attempts',3):")
        code=code.replace('if attempt==2:', "if attempt+1>=getattr(self,'max_attempts',3):")
        old="print(json.dumps({'rows':len(rows),'failed':failures,'output':args.output}))"
        new="print(json.dumps({'rows':len(rows),'failed':failures,'output':args.output, 'status':'degraded' if failures else 'complete', 'error_types':dict(Counter(e for e in errors if e))}))"
        if old not in code:raise ValueError('Missing summary anchor')
        code=code.replace(old,new)
    elif mode=='decision':
        anchor='        def extract(result):'
        if code.count(anchor)!=1 or 'from decision import' in code:
            raise ValueError('Unexpected baseline code')
        code=code.replace(anchor,'        from decision import prepare\n        payload=prepare(self,payload,context,question)\n'+anchor)
    else:raise ValueError('Unknown mode')
    files['solution.py']=code.encode('utf-8')
    files[mode+'.py']=(ROOT/'submission'/(mode+'.py')).read_bytes()
    for name,content in files.items():
        if name.endswith('.py'):compile(content.decode('utf-8'),name,'exec')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,content in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,content)
    manifest={'archive':output.name,'base':base.name,
        'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),
        'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'files':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()},
        'status':'prepared_not_submitted','mode':mode,'score':None}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['decision','decision_v2','intent'],default='decision')
    a=p.parse_args();print(json.dumps(build(a.base,a.output,a.mode),indent=2))
