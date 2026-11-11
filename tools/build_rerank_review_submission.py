"""Build context-rerank-review-v1 without changing the winning baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA256 = '5cd8328cfed2e3ff379e41a92b010f9990b77a738c2c7b54c8b039bb4c8bcab0'
EXPECTED = {'solution.py', 'batch_io.py', 'retrieval.py', 'context.py', 'requirements.txt'}


def files():
    base = ROOT / 'dist/energytool-context-v1.zip'
    if hashlib.sha256(base.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError('Requires exact context-v1 ZIP')
    with zipfile.ZipFile(base) as z:
        if set(z.namelist()) != EXPECTED or len(z.namelist()) != len(EXPECTED):
            raise ValueError('Unexpected baseline entries')
        result = {n: z.read(n) for n in z.namelist()}
    original = result['solution.py']
    start = original.index(b'    def answer(self,question,sources):')
    end = original.index(b'\ndef locate(names):', start)
    result['solution.py'] = (original[:start] +
        b'    def answer(self,question,sources):\n        from rerank_review import answer\n'
        b'        return answer(self,question,sources,SYSTEM)\n\n' + original[end:])
    anchor = b"self.reranker=env('RERANKER_MODEL')"
    if result['solution.py'].count(anchor) != 1:
        raise ValueError('Unexpected model configuration')
    # Reranker default is from the organizer sample; generation ID remains env-only.
    result['solution.py'] = result['solution.py'].replace(anchor,
        b"self.reranker=env('RERANKER_MODEL',default='qwen/qwen3-reranker-8b')")
    result['rerank_review.py'] = (ROOT / 'submission/rerank_review.py').read_bytes()
    for name, data in result.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    return result


def build(output):
    output = Path(output)
    manifest_path = output.with_suffix('.manifest.json')
    if output.exists() or manifest_path.exists():
        raise ValueError('Refusing to overwrite existing artifact')
    if not output.parent.is_dir():
        raise ValueError('Output directory must already exist')
    content = files()
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(content.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    assert output.stat().st_size < 128_000_000
    manifest = dict(archive=output.name, base_sha256=BASE_SHA256,
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    files={n: hashlib.sha256(b).hexdigest() for n, b in content.items()},
                    changed_baseline_files=['solution.py'], status='prepared_not_submitted', score=None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/energytool-context-rerank-review-v1.zip')
    print(json.dumps(build(parser.parse_args().output), indent=2))
