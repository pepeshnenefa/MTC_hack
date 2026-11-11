"""Compute bge-m3 vectors through the submission code path and cache them for offline tuning.

    LLM_BASE_URL=http://127.0.0.1:8089/v1 python3 tools/embed_cache.py [--holdout]

The cache lives in work/ (ignored by Git, never packaged): the platform recomputes vectors at run time.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
sys.path.insert(0, str(ROOT / 'tools'))
os.environ.setdefault('LLM_API_KEY', 'local-test-key')
os.environ.setdefault('LLM_MODEL', 'unused')
import hybrid  # noqa: E402
import solution  # noqa: E402
import tune_retrieval as T  # noqa: E402
from retrieval import query  # noqa: E402

CACHE = ROOT / 'work/emb/cache.json'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]


def load():
    return json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', action='store_true')
    parser.add_argument('--model', default='baai/bge-m3')
    args = parser.parse_args()
    cache = load()
    items = T.chunks(T.documents())
    texts = [c.get('title', c['heading']) + '\n' + c['text'] for c in items]
    texts += [query(c['question']) for c in T.load_cases(args.holdout)]
    if args.holdout:  # every holdout question, including those without labelled sources
        import csv
        with (ROOT / 'tests/evaluation/holdout-v1/questions.csv').open(encoding='utf-8-sig') as stream:
            texts += [query(row['question']) for row in csv.DictReader(stream)]
    missing = [t for t in dict.fromkeys(texts) if digest(t) not in cache]
    api = solution.API()
    for offset in range(0, len(missing), 16):
        batch = missing[offset:offset + 16]
        for text, vector in zip(batch, hybrid.embed(api, args.model, batch)):
            cache[digest(text)] = [round(v, 5) for v in vector]
        print('embedded', offset + len(batch), 'of', len(missing), flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache), encoding='utf-8')
    print('cache size', len(cache))


if __name__ == '__main__':
    main()
