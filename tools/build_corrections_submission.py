"""Build a retrieval-only experiment from the exact context-v1 archive."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA256 = '5cd8328cfed2e3ff379e41a92b010f9990b77a738c2c7b54c8b039bb4c8bcab0'
EXPECTED = {'solution.py', 'batch_io.py', 'retrieval.py', 'context.py', 'requirements.txt'}


def build(base, output):
    base, output = Path(base), Path(output)
    if hashlib.sha256(base.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError('Requires exact immutable context-v1 ZIP')
    if output.exists() or output.with_suffix('.manifest.json').exists():
        raise ValueError('Refusing to overwrite an existing artifact')
    with zipfile.ZipFile(base) as archive:
        if set(archive.namelist()) != EXPECTED or len(archive.namelist()) != len(EXPECTED):
            raise ValueError('Unexpected baseline files')
        files = {name: archive.read(name) for name in archive.namelist()}
    # Insert only at the ASCII function signature; keep all baseline bytes intact.
    anchor = b'    def find(self,text,k=12):'
    if files['retrieval.py'].count(anchor) != 1:
        raise ValueError('Unexpected search entry point')
    files['retrieval.py'] = files['retrieval.py'].replace(anchor, anchor +
        b'\n        from corrections import search_text\n        text=search_text(text,dialogue)')
    files['corrections.py'] = (ROOT / 'submission/corrections.py').read_bytes()
    for name, content in files.items():
        if name.endswith('.py'):
            compile(content, name, 'exec')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    if output.stat().st_size > 128_000_000:
        raise ValueError('ZIP exceeds limit')
    manifest = {'archive': output.name, 'base': base.name, 'base_sha256': BASE_SHA256,
                'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'files': {n: hashlib.sha256(b).hexdigest() for n, b in files.items()},
                'changed_baseline_files': ['retrieval.py'],
                'status': 'prepared_not_submitted', 'score': None}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=ROOT / 'dist/energytool-context-v1.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/energytool-context-corrections-v1.zip')
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output), indent=2))
