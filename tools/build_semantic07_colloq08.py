"""Build semantic07 with W_COLLOQUIAL 0.5 -> 0.8 and nothing else (same packing as semantic07).

    python3 -B tools/build_semantic07_colloq08.py [--out PATH]

Checks the base hash, changes one line of config.py, keeps file order, dates and compression,
refuses to overwrite. Every other file stays byte-identical to the base archive.
"""
import argparse, hashlib, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'dist/energytool-unified-v1-hybrid-semantic07.zip'
BASE_SHA = '4d1b307e7a786c29d74824c453a3e21249841cbb04474e44e2e5c4bbe2579f8c'

parser = argparse.ArgumentParser()
parser.add_argument('--out', default=str(ROOT / 'dist/energytool-unified-v1-hybrid-semantic07-colloq08.zip'))
out = Path(parser.parse_args().out)
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA, 'base archive changed'
if out.exists():
    raise SystemExit('Refusing to overwrite ' + str(out))
with zipfile.ZipFile(BASE) as z:
    content = {n: z.read(n) for n in z.namelist()}
old, new = b"'W_COLLOQUIAL': 0.5,", b"'W_COLLOQUIAL': 0.8,"
assert content['config.py'].count(old) == 1 and content['config.py'].count(b'W_COLLOQUIAL') == 1
content['config.py'] = content['config.py'].replace(old, new)
for n, data in content.items():
    assert '/' not in n and (n.endswith('.py') or n == 'requirements.txt')
    if n.endswith('.py'):
        compile(data, n, 'exec')
with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for n, data in content.items():
        info = zipfile.ZipInfo(n, (2026, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, data)
with zipfile.ZipFile(out) as z, zipfile.ZipFile(BASE) as b:
    assert z.testzip() is None and z.namelist() == b.namelist()
    assert [n for n in z.namelist() if z.read(n) != b.read(n)] == ['config.py']
print(out.name, hashlib.sha256(out.read_bytes()).hexdigest())
