"""Add the same two-stage experiment to an immutable known baseline ZIP."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def build(base, output):
    base=Path(base);output=Path(output)
    if base.resolve()==output.resolve() or output.exists():
        raise ValueError('Never overwrite an experiment or baseline')
    with zipfile.ZipFile(base) as z:
        files={n:z.read(n) for n in z.namelist()}
        if len(files)!=len(z.namelist()) or any('/' in n or '\\' in n or not(n.endswith('.py') or n=='requirements.txt') for n in files):
            raise ValueError('Invalid baseline entries')
    code=files['solution.py'].decode('utf-8')
    anchor='        def extract(result):'
    if code.count(anchor)!=1 or 'from decision import' in code:
        raise ValueError('Unexpected baseline code')
    code=code.replace(anchor,'        from decision import prepare\n        payload=prepare(self,payload,context,question)\n'+anchor)
    files['solution.py']=code.encode('utf-8')
    files['decision.py']=(ROOT/'submission/decision.py').read_bytes()
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
        'status':'prepared_not_submitted','score':None}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.base,a.output),indent=2))
