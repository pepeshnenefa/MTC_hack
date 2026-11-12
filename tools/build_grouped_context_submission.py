"""Preserve exact semantic07; change only the presentation of selected context."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dist/energytool-unified-v1-hybrid-semantic07.zip'
SHA='4d1b307e7a786c29d74824c453a3e21249841cbb04474e44e2e5c4bbe2579f8c'
OUTPUT=ROOT/'dist/energytool-unified-v1-hybrid-grouped-v1.zip'


def files():
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==SHA
    with zipfile.ZipFile(BASE) as z:
        assert z.testzip() is None
        content={n:z.read(n) for n in z.namelist()}
        assert len(content)==len(z.namelist())==9
    anchor=b"    return '\\n\\n'.join(e[1] for e in order),{'selected':selected,'included':included,'dropped':dropped,'characters':used}"
    assert content['context.py'].count(anchor)==1
    replacement=(b"    from grouped_context import format_context\n"
        b"    return format_context('\\n\\n'.join(e[1] for e in order),"
        b"{'selected':selected,'included':included,'dropped':dropped,'characters':used},max_chars)")
    content['context.py']=content['context.py'].replace(anchor,replacement)
    content['grouped_context.py']=(ROOT/'submission/grouped_context.py').read_bytes()
    for n,data in content.items():
        assert '/' not in n and (n.endswith('.py') or n=='requirements.txt')
        if n.endswith('.py'):compile(data,n,'exec')
    return content


def build():
    manifest=OUTPUT.with_suffix('.manifest.json')
    if OUTPUT.exists() or manifest.exists():raise ValueError('Refusing to overwrite artifact')
    content=files()
    with zipfile.ZipFile(OUTPUT,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,data in sorted(content.items()):
            info=zipfile.ZipInfo(n,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    assert OUTPUT.stat().st_size<128000000
    metadata=dict(archive=OUTPUT.name,base_sha256=SHA,sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        files={n:hashlib.sha256(b).hexdigest() for n,b in content.items()},
        changed_baseline_files=['context.py'],status='prepared_not_submitted',score=None)
    manifest.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


if __name__=='__main__':print(json.dumps(build(),indent=2))
