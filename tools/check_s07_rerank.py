"""End-to-end check of dist/energytool-s07-rerank.zip against semantic07, synthetic service, no model.

Both archives run exactly as on the platform (parent + worker) on the 18 organizer demo questions
against one local HTTP stand-in with deterministic embeddings and answers. The request each
question sends to /chat/completions is recorded and compared with semantic07:
- reranker switched off, HTTP 503, HTTP 404, malformed response, timeout: identical to semantic07;
- reranker answering: the context starts with the reranker's choice; one /rerank call per question.
Synthetic similarities and answers: this checks wiring and fallbacks, not answer quality.

    python3 tools/check_s07_rerank.py [--archive dist/energytool-s07-rerank.zip]
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'dist/energytool-unified-v1-hybrid-semantic07.zip'
MARK = 'ДИАЛОГ ИЛИ ВОПРОС:\n'


class Stand:
    def __init__(self):
        self.mode, self.calls, self.chats, self.lock = 'ok', [], {}, threading.Lock()
        stand = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, data):
                raw = json.dumps(data).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert self.headers['Authorization'] == 'Bearer local-test'
                with stand.lock:
                    stand.calls.append(self.path)
                if self.path == '/embeddings':
                    return self.send(200, {'data': [{'index': i, 'embedding': [b + 1 for b in hashlib.sha256(t.encode()).digest()[:8]]}
                                                    for i, t in enumerate(body['input'])]})
                if self.path == '/rerank':
                    mode = stand.mode
                    if mode in ('503', '404'):
                        return self.send(int(mode), {'error': mode})
                    if mode == 'garbage':
                        return self.send(200, {'oops': 1})
                    if mode == 'slow':
                        time.sleep(3)
                    n = len(body['documents'])
                    assert body['top_n'] == min(12, n) and body['model'] == 'qwen/qwen3-reranker-8b'
                    return self.send(200, {'results': [{'index': i, 'relevance_score': i / n} for i in range(n)][::-1][:body['top_n']]})
                assert self.path == '/chat/completions'
                content = body['messages'][1]['content']
                with stand.lock:
                    stand.chats[content.split(MARK, 1)[1]] = body
                return self.send(200, {'choices': [{'message': {'content': 'ТЕСТ: ответ.'}, 'finish_reason': 'stop'}]})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def run(self, archive, mode, folder, **env):
        self.mode, self.calls, self.chats = mode, [], {}
        harness, output = folder / 'harness.py', folder / 'submission.csv'
        harness.write_text('import sys\nsys.path.insert(0, %r)\nimport solution\nsolution.main()\n' % str(archive), encoding='utf-8')
        environment = {k: v for k, v in os.environ.items() if not k.startswith('ET_') and 'RERANK' not in k}
        environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', LLM_BASE_URL='http://127.0.0.1:%d' % self.server.server_port,
                           LLM_API_KEY='local-test', LLM_MODEL='local-test-model', TOP_K='12', MAX_WORKERS='3',
                           RUN_BUDGET_SECONDS='300', **env)
        started = time.monotonic()
        run = subprocess.run([sys.executable, '-B', str(harness), '--questions', str(ROOT / 'demo_questions.csv'),
                              '--kb', str(ROOT / 'Knowledge-base'), '--output', str(output)], cwd=folder, env=environment,
                             capture_output=True, text=True, encoding='utf-8', timeout=240)
        seconds = round(time.monotonic() - started, 2)
        assert run.returncode == 0, run.stderr
        assert 'hybrid_ready' in run.stderr, run.stderr
        with output.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        with (ROOT / 'demo_questions.csv').open(encoding='utf-8-sig', newline='') as stream:
            ids = [r['id'] for r in csv.DictReader(stream)]
        assert reader.fieldnames == ['id', 'answer'] and [r['id'] for r in rows] == ids
        assert all(r['answer'] == 'ТЕСТ: ответ.' for r in rows), rows
        assert len(self.chats) == len(ids)
        return dict(self.chats), {'seconds': seconds, 'rerank_calls': self.calls.count('/rerank'),
                                  'chat_calls': self.calls.count('/chat/completions'),
                                  'events': sorted({json.loads(l)['event'] for l in run.stderr.splitlines() if l.startswith('{"event": "rerank')})}


def check(archive):
    stand = Stand()
    report = {}
    try:
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            original, report['semantic07'] = stand.run(BASE, 'ok', folder)
            assert report['semantic07']['rerank_calls'] == 0
            for mode, env, limit in (('off', {'ET_USE_RERANKER': '0'}, 0), ('503', {}, 2 * (3 + 3)), ('404', {}, 3),
                                     ('garbage', {}, 3 + 3), ('slow', {'ET_RERANK_TIMEOUT': '1'}, 3 + 3)):
                chats, report[mode] = stand.run(archive, 'ok' if mode == 'off' else mode, folder, **env)
                assert chats == original, mode + ': request to the model differs from semantic07'
                assert report[mode]['rerank_calls'] <= limit, (mode, report[mode])
                report[mode]['identical_to_semantic07'] = True
            chats, report['reranked'] = stand.run(archive, 'ok', folder)
            assert report['reranked']['rerank_calls'] == 18 and report['reranked']['chat_calls'] == 18
            changed = sum(chats[q] != original[q] for q in original)
            assert changed > 0
            report['reranked']['questions_with_changed_context'] = changed
    finally:
        stand.server.shutdown()
        stand.server.server_close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default=str(ROOT / 'dist/energytool-s07-rerank.zip'))
    print(json.dumps(check(Path(parser.parse_args().archive).resolve()), indent=2, ensure_ascii=False))
