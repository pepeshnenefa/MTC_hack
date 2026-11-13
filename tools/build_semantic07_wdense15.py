"""Build semantic07 with W_DENSE 3.0 -> 1.5; every other byte of the archive stays as in semantic07.

    python3 -B tools/build_semantic07_wdense15.py

Base: the exact dist/energytool-unified-v1-hybrid-semantic07.zip (61.019 on the platform).
Only the W_DENSE line of config.py changes; file order, timestamps and compression follow the
base archive. Existing artifacts are never overwritten. ET_W_DENSE in the environment would still
override the packaged value at run time, as for every setting (config.reload).
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'dist/energytool-unified-v1-hybrid-semantic07.zip'
BASE_SHA = '4d1b307e7a786c29d74824c453a3e21249841cbb04474e44e2e5c4bbe2579f8c'
OUTPUT = ROOT / 'dist/energytool-unified-v1-hybrid-semantic07-wdense15.zip'
OLD = b"    'W_DENSE': 3.0,              # weight of embedding similarity in the weighted policy\n"
NEW = b"    'W_DENSE': 1.5,              # weight of embedding similarity in the weighted policy\n"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if sha(BASE.read_bytes()) != BASE_SHA:
        raise ValueError('Base archive is not the exact semantic07')
    manifest = OUTPUT.with_suffix('.manifest.json')
    if OUTPUT.exists() or manifest.exists():
        raise ValueError('Refusing to overwrite artifact')
    with zipfile.ZipFile(BASE) as base:
        names = base.namelist()
        data = {name: base.read(name) for name in names}
    if data['config.py'].count(OLD) != 1:
        raise ValueError('Expected exactly one W_DENSE line')
    data['config.py'] = data['config.py'].replace(OLD, NEW)
    with zipfile.ZipFile(OUTPUT, 'x', compression=zipfile.ZIP_DEFLATED) as target:
        for name in names:
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, data[name])
    metadata = {'archive': OUTPUT.name, 'base_archive': BASE.name, 'base_sha256': BASE_SHA,
                'sha256': sha(OUTPUT.read_bytes()), 'default_overrides': {'W_DENSE': 1.5},
                'files': {name: sha(data[name]) for name in names},
                'status': 'prepared_not_submitted', 'score': None}
    manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'archive': OUTPUT.name, 'sha256': metadata['sha256']}))


if __name__ == '__main__':
    main()
