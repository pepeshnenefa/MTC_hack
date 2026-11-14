"""Local stand for s07-rerank (development only, never packaged).

The 8 GB machine cannot hold the reranker (Qwen3-Reranker-8B, ~5 GB) and the generation model
at once, so the reranker runs first and alone: `prefetch` sends exactly the /rerank requests the
candidate will send (same pool, query and documents, built by the archive code itself) and stores
the answers with their latency. `serve` is the platform stand-in on one address: /embeddings
from a bge-m3 cache, /rerank from the prefetch cache, /chat/completions to the local model.
`report` compares two local_eval runs: answers, included sources and time per question.

    python3 tools/rerank_stand.py prefetch --emb E.json --cache R.json --reranker http://127.0.0.1:8091
    python3 tools/rerank_stand.py serve --emb E.json --cache R.json --port 8090
    python3 tools/rerank_stand.py report --base s07-demo --candidate s07-rerank-demo --out report.md

E.json: {sha256(text): vector} of every text the archive embeds; R.json is written by prefetch.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('s07_rerank_builder', ROOT / 'tools/build_s07_rerank.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
ORDER = ('config', 'retrieval', 'context', 'corrections', 'hybrid', 'hierarchy', 'batch_io', 'rerank', 'solution')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical(body):
    """Cache key of a /rerank request: the same JSON whatever the key order of the sender."""
    return digest(json.dumps(body, sort_keys=True, ensure_ascii=False))


def load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


@contextlib.contextmanager
def archive_modules():
    content = builder.files()
    previous = {n: sys.modules.pop(n, None) for n in ORDER}
    try:
        for name in ORDER:
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / 'dist' / (name + '.py'))
            sys.modules[name] = module
            exec(compile(content[name + '.py'], module.__file__, 'exec'), module.__dict__)
        yield {n: sys.modules[n] for n in ORDER}
    finally:
        for name in ORDER:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


def prefetch(args):
    os.environ.update(LLM_BASE_URL='http://unused.invalid', LLM_API_KEY='local', LLM_MODEL='unused')
    vectors = load(args.emb)
    cache = load(args.cache)
    rows = read_csv(args.questions)
    with archive_modules() as m:
        config, solution, retrieval, hybrid, rerank = m['config'], m['solution'], m['retrieval'], m['hybrid'], m['rerank']
        config.reload()
        top_k = max(1, min(100, int(config.get('TOP_K'))))

        class Embeddings:  # the archive's own hybrid.prepare, vectors from the cache
            def post(self, endpoint, body):
                return {'data': [{'index': i, 'embedding': vectors[digest(t)]} for i, t in enumerate(body['input'])]}

        lexical = retrieval.Search(retrieval.chunks(solution.documents(ROOT / 'Knowledge-base')))
        search = hybrid.prepare(lexical, [r['question'] for r in rows], Embeddings(), 'baai/bge-m3',
                                time.monotonic() + 600, anchor_k=top_k)
        assert isinstance(search, hybrid.HybridSearch), 'embedding cache incomplete'
        api = types.SimpleNamespace(base=args.reranker.rstrip('/') + '/v1', key='local',
                                    reranker=rerank.DEFAULT_MODEL, deadline=None)
        stage = rerank.Stage(api)
        send = stage.post
        latency = {}

        def post(body):
            key = canonical(body)
            if key not in cache:
                started = time.monotonic()
                data = send(body)
                cache[key] = {'response': data, 'seconds': round(time.monotonic() - started, 2),
                              'documents': len(body['documents']), 'chars': sum(map(len, body['documents']))}
                Path(args.cache).write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
            latency[key] = cache[key]['seconds']
            return cache[key]['response']

        stage.post = post
        with config.override(RERANK_TIMEOUT=1800, RERANK_SLOW=1e9, RERANK_ATTEMPTS=1):
            for row in rows:
                pool = search.find(row['question'], k=min(100, max(36, top_k * 3)))
                ordered, info = stage.order(row['question'], pool)
                hybrid_top = {id(s) for s in pool[:top_k]}
                kept = sum(id(s) in hybrid_top for s in ordered[:top_k])
                print(json.dumps({'id': row['id'], 'status': info['status'], 'pool': len(pool),
                                  'seconds': info.get('seconds'), 'kept_from_hybrid_top12': kept,
                                  'picked': info.get('picked')}, ensure_ascii=False), flush=True)
                if info['status'] != 'ok':
                    raise SystemExit('reranker failed: ' + str(info))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, payload):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.reply(200, {'status': 'ok'})

    def do_POST(self):
        server = self.server
        raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        try:
            body = json.loads(raw)
            if self.path.endswith('/embeddings'):
                vectors = [server.vectors.get(digest(t)) for t in body['input']]
                if any(v is None for v in vectors):
                    server.misses['embeddings'] += 1
                    return self.reply(502, {'error': 'embedding not cached'})
                return self.reply(200, {'data': [{'index': i, 'embedding': v} for i, v in enumerate(vectors)]})
            if self.path.endswith('/rerank'):
                hit = server.reranks.get(canonical(body))
                if hit is None:
                    server.misses['rerank'] += 1
                    return self.reply(502, {'error': 'rerank not cached'})
                return self.reply(200, hit['response'])
            request = Request(server.chat + self.path, data=raw, headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=900) as response:
                return self.reply(200, response.read())
        except Exception as exc:
            return self.reply(502, {'error': type(exc).__name__})


def serve(args):
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.vectors, server.reranks, server.chat = load(args.emb), load(args.cache), args.chat.rstrip('/')
    server.misses = {'embeddings': 0, 'rerank': 0}
    print(json.dumps({'port': args.port, 'vectors': len(server.vectors), 'reranks': len(server.reranks)}), flush=True)
    try:
        server.serve_forever()
    finally:
        print(json.dumps({'misses': server.misses}), flush=True)


def report(args):
    golden = {r['id']: r['answer'] for r in read_csv(ROOT / 'demo_output.csv')}
    questions = read_csv(ROOT / 'demo_questions.csv')

    def run(name):
        out = ROOT / 'work/eval' / name
        answers = {r['id']: r['answer'] for r in read_csv(out / 'demo.csv')}
        traces = {}
        for line in (out / 'demo.trace.jsonl').read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            traces[item['question']] = item
        return answers, traces, json.loads((out / 'demo.summary.json').read_text(encoding='utf-8'))

    base, cand = run(args.base), run(args.candidate)
    lines = ['# %s против %s (demo, %d вопросов)' % (args.candidate, args.base, len(questions)), '']
    totals = {'same_sources': 0, 'same_answer': 0, 'overlap': [], 'seconds': ([], []), 'words': ([], []), 'calls': ([], [])}
    for row in questions:
        qid, question = row['id'], row['question']
        a, b = base[1][question], cand[1][question]
        sa, sb = a['sources'], b['sources']
        overlap = len(set(sa) & set(sb))
        totals['overlap'].append(overlap)
        totals['same_sources'] += sa == sb
        totals['same_answer'] += base[0][qid] == cand[0][qid]
        for i, t in enumerate((a, b)):
            totals['seconds'][i].append(t['seconds'])
            totals['calls'][i].append(t['calls'])
        for i, answers in enumerate((base[0], cand[0])):
            totals['words'][i].append(len(answers[qid].split()))
        info = b.get('rerank', {})
        lines += ['## %s' % qid, '', '**Вопрос:** ' + question.replace('\n', ' // '), '',
                  '**Эталон:** ' + golden[qid], '',
                  '**%s** (%s с, %s выз.): %s' % (args.base, a['seconds'], a['calls'], base[0][qid]), '',
                  '**%s** (%s с, %s выз., реранкер %s, выбраны позиции пула %s): %s' % (
                      args.candidate, b['seconds'], b['calls'], info.get('status'), info.get('picked'), cand[0][qid]), '',
                  'Источники: общих %d из %d/%d.' % (overlap, len(sa), len(sb)), '']
        only_a = [s for s in sa if s not in sb]
        only_b = [s for s in sb if s not in sa]
        if only_a:
            lines += ['- только %s: %s' % (args.base, '; '.join(only_a))]
        if only_b:
            lines += ['- только %s: %s' % (args.candidate, '; '.join(only_b))]
        lines += ['- первые три у %s: %s' % (args.base, '; '.join(sa[:3])),
                  '- первые три у %s: %s' % (args.candidate, '; '.join(sb[:3])), '']
    mean = lambda xs: round(sum(xs) / max(1, len(xs)), 1)
    summary = {'questions': len(questions), 'identical_sources': totals['same_sources'],
               'identical_answers': totals['same_answer'], 'mean_shared_sources': mean(totals['overlap']),
               'seconds_per_question': [mean(totals['seconds'][0]), mean(totals['seconds'][1])],
               'run_seconds': [base[2]['seconds'], cand[2]['seconds']],
               'mean_words': [mean(totals['words'][0]), mean(totals['words'][1])],
               'calls': [sum(totals['calls'][0]), sum(totals['calls'][1])]}
    lines[1:1] = ['', '```', json.dumps(summary, ensure_ascii=False, indent=1), '```']
    Path(args.out).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prefetch')
    p.add_argument('--emb', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--reranker', default='http://127.0.0.1:8091')
    p.add_argument('--questions', default=str(ROOT / 'demo_questions.csv'))
    p.set_defaults(func=prefetch)
    p = sub.add_parser('serve')
    p.add_argument('--emb', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--port', type=int, default=8090)
    p.add_argument('--chat', default='http://127.0.0.1:8088')
    p.set_defaults(func=serve)
    p = sub.add_parser('report')
    p.add_argument('--base', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)
