"""Exact context-v1 plus a bounded, stateless structured query stage."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA256 = '5cd8328cfed2e3ff379e41a92b010f9990b77a738c2c7b54c8b039bb4c8bcab0'


def files():
    base = ROOT/'dist/energytool-context-v1.zip'
    if hashlib.sha256(base.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError('Requires exact context-v1')
    with zipfile.ZipFile(base) as z:
        result = {n:z.read(n) for n in z.namelist()}
        if len(z.namelist()) != 5 or set(result) != {'solution.py','retrieval.py','context.py','batch_io.py','requirements.txt'}:
            raise ValueError('Unexpected baseline entries')
    code = result['solution.py']
    def replace(old, new):
        nonlocal code
        if code.count(old) != 1:
            raise ValueError('Unexpected baseline anchor')
        code = code.replace(old, new)
    replace(b'from context import pack_context', b'from context import pack_context\nfrom structured_query import answer as structured_answer, append_context')
    replace(b'    def answer(self,question,sources):',
            b"    def answer(self,question,sources):\n        extras=[s for s in sources if s.get('structured_extra')]\n        sources=[s for s in sources if not s.get('structured_extra')]")
    anchor = b"        context,report=pack_context(question,sources,limit=setting('TOP_K',12,1,100))"
    replace(anchor, anchor+b'\n        context=append_context(context,extras)')
    replace(b'api.answer(question,search.find(question,k=min(100,max(36,top_k*3))))',
            b'structured_answer(api,search,question,k=min(100,max(36,top_k*3)),top_k=top_k)')
    result['solution.py'] = code
    result['structured_query.py'] = (ROOT/'submission/structured_query.py').read_bytes()
    for name,data in result.items():
        if name.endswith('.py'): compile(data,name,'exec')
    return result


def build(output):
    output = Path(output)
    manifest_path = output.with_suffix('.manifest.json')
    if output.exists() or manifest_path.exists():
        raise ValueError('Refusing to overwrite existing artifact')
    if not output.parent.is_dir():
        raise ValueError('Output directory must exist')
    content = files()
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(content.items()):
            info=zipfile.ZipInfo(name,(2026,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    assert output.stat().st_size < 128_000_000
    manifest=dict(archive=output.name,base_sha256=BASE_SHA256,
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  files={n:hashlib.sha256(b).hexdigest() for n,b in content.items()},
                  changed_baseline_files=['solution.py'],status='prepared_not_submitted',score=None)
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'dist/energytool-context-structured-query-v1.zip')
    print(json.dumps(build(parser.parse_args().output),indent=2))
