"""Build s07-rerank: the exact semantic07 archive plus the Qwen3 Reranker stage (submission/rerank.py).

Hybrid search (BM25 + bge-m3, M_DENSE_MISMATCH 0.7) still returns the same 36-section pool;
the reranker puts the 12 most relevant sections first, the rest of the pool keeps its order.
Every other file and setting of semantic07 is byte-for-byte unchanged; ET_USE_RERANKER=0 or any
reranker failure gives exactly the semantic07 request to the generation model.

    python3 tools/build_s07_rerank.py                      # dist/energytool-s07-rerank.zip + manifest
    python3 tools/build_s07_rerank.py --output X.zip       # any other place (development)
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'dist/energytool-unified-v1-hybrid-semantic07.zip'
SHA = '4d1b307e7a786c29d74824c453a3e21249841cbb04474e44e2e5c4bbe2579f8c'
OUTPUT = ROOT / 'dist/energytool-s07-rerank.zip'

SOLUTION = [
    (b"from hierarchy import HierarchicalSearch\n",
     b"from hierarchy import HierarchicalSearch\nimport rerank\n"),
    (b"        self.reranker=env('RERANKER_MODEL')\n",
     b"        self.reranker=env('RERANKER_MODEL',default=rerank.DEFAULT_MODEL)\n"),
    (b"            raise ValueError('Missing RERANKER_MODEL')\n",
     b"            raise ValueError('Missing RERANKER_MODEL')\n        self.rerank=rerank.Stage(self)\n"),
    (b"""        if config.get('USE_RERANKER') and sources:
            try:
                data=self.post('/rerank',{'model':self.reranker,'query':query(question),'documents':[s.get('title',s['heading'])+'\\n'+s['text'] for s in sources],'top_n':min(self.reranker_top_n,len(sources))})
                indices=[int(x['index']) for x in data.get('results',data.get('data',[]))]
                selected=[sources[i] for i in dict.fromkeys(indices) if 0<=i<len(sources)]
                if selected:sources=selected
            except Exception as exc:print('Reranker unavailable: '+type(exc).__name__,file=sys.stderr)
""",
     b"""        # s07-rerank: the reranker's choice heads the hybrid pool; on any failure the order is semantic07's.
        sources,rerank_info=self.rerank.order(question,sources)
"""),
    (b"'context_chars':len(context),\n",
     b"'context_chars':len(context),'rerank':rerank_info,\n"),
]

CONFIG = [
    (b"    'USE_RERANKER': 0,\n",
     b"    'USE_RERANKER': 1,           # s07-rerank: Qwen3 Reranker orders the hybrid pool (rerank.py); 0 = exact semantic07\n"
     b"    'RERANK_KEEP': 12,           # sections the reranker places first: the 12 the context is built from\n"
     b"    'RERANK_DOC_CHARS': 6000,    # per-section cut of the reranker input only (longest section now 4914); context keeps full text\n"
     b"    'RERANK_TIMEOUT': 15.0,      # seconds per reranker request; a timeout is not retried\n"
     b"    'RERANK_ATTEMPTS': 2,        # one retry after 408/429/5xx\n"
     b"    'RERANK_RESERVE': 60.0,      # seconds of run budget kept for generation: below it the stage is skipped\n"
     b"    'RERANK_MAX_BAD': 3,         # consecutive failed or slow calls that switch the stage off for the run\n"
     b"    'RERANK_SLOW': 12.0,         # a successful call slower than this counts as a bad call (its result is still used)\n"),
]


def patch(data, replacements, name):
    for old, new in replacements:
        if data.count(old) != 1:
            raise ValueError('Anchor not found exactly once in ' + name + ': ' + old[:60].decode())
        data = data.replace(old, new)
    return data


def base_files():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != SHA:
        raise ValueError('semantic07 archive differs from the platform-scored build')
    with zipfile.ZipFile(BASE) as z:
        return {n: z.read(n) for n in z.namelist()}


def files():
    """Archive content in memory: semantic07 with two patched files and rerank.py added."""
    content = base_files()
    content['solution.py'] = patch(content['solution.py'], SOLUTION, 'solution.py')
    content['config.py'] = patch(content['config.py'], CONFIG, 'config.py')
    content['rerank.py'] = (ROOT / 'submission/rerank.py').read_bytes()
    for n, data in content.items():
        assert '/' not in n and (n.endswith('.py') or n == 'requirements.txt'), n
        if n.endswith('.py'):
            compile(data, n, 'exec')
    return content


def build(output, manifest=True):
    output = Path(output)
    meta = output.with_suffix('.manifest.json')
    if output.exists() or (manifest and meta.exists()):
        raise ValueError('Refusing to overwrite artifact: ' + str(output))
    content = files()
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for n, data in content.items():
            info = zipfile.ZipInfo(n, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    with zipfile.ZipFile(output) as z, zipfile.ZipFile(BASE) as base:
        assert z.testzip() is None
        assert z.namelist() == base.namelist() + ['rerank.py']
        changed = [n for n in base.namelist() if z.read(n) != base.read(n)]
        assert sorted(changed) == ['config.py', 'solution.py'], changed
    metadata = dict(archive=output.name, base=BASE.name, base_sha256=SHA,
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    changed=['config.py', 'solution.py'], added=['rerank.py'],
                    default_overrides={'USE_RERANKER': 1, 'RERANK_KEEP': 12},
                    files={n: hashlib.sha256(b).hexdigest() for n, b in content.items()},
                    status='prepared_not_submitted', score=None)
    if manifest:
        meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(OUTPUT))
    parser.add_argument('--no-manifest', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.output, manifest=not args.no_manifest), indent=2, ensure_ascii=False))
