"""Actual ZIP parent/worker check against a synthetic local HTTP service."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zipfile

from build_rerank_review_submission import files

ROOT = Path(__file__).resolve().parents[1]


def check():
    archive = ROOT / 'dist/energytool-context-rerank-review-v1.zip'
    expected = files()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == 6 and set(z.namelist()) == set(expected)
        assert all(z.read(n) == b for n, b in expected.items())
    manifest = json.loads(archive.with_suffix('.manifest.json').read_text())
    assert manifest['sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with (ROOT/'demo_questions.csv').open(encoding='utf-8-sig', newline='') as f:
        ids = [r['id'] for r in csv.DictReader(f)]
    calls = []
    mode = 'normal'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            editing = self.path != '/rerank' and 'ЧЕРНОВИК (не источник фактов)' in body['messages'][1]['content']
            stage = 'rerank' if self.path == '/rerank' else ('edit' if editing else 'draft')
            calls.append(stage)
            assert self.headers['Authorization'] == 'Bearer local-test'
            if (mode == 'rerank_failure' and stage == 'rerank') or (mode == 'editor_failure' and editing):
                self.send_response(503); self.send_header('Content-Length', '0'); self.end_headers(); return
            if stage == 'rerank':
                data = {'results': [{'index': i} for i in reversed(range(len(body['documents'])))]}
            else:
                data = {'choices': [{'message': {'content': 'ТЕСТ: исправленный ответ.' if editing else 'ТЕСТ: черновик.'}, 'finish_reason': 'stop'}]}
            raw = json.dumps(data).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    paths = []
    reports = []
    try:
        for suffix in ('.py', '.csv'):
            path = ROOT/'work'/('rerank-review-check-'+uuid.uuid4().hex+suffix)
            with path.open('x', encoding='utf-8'):
                pass
            paths.append(path)
        harness, output = paths
        harness.write_text('import sys\nsys.path.insert(0, '+repr(str(archive))+')\nimport solution\nsolution.main()\n', encoding='utf-8')
        environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8',
                       'LLM_BASE_URL': 'http://127.0.0.1:'+str(server.server_port),
                       'LLM_API_KEY': 'local-test', 'LLM_MODEL': 'local-test-model',
                       'RERANKER_MODEL': 'local-test-reranker', 'USE_RERANKER': '1',
                       'USE_ANSWER_REVIEW': '1', 'RUN_BUDGET_SECONDS': '90',
                       'MAX_WORKERS': '3', 'TOP_K': '12', 'RERANKER_TOP_N': '6'}
        for mode in ('normal', 'rerank_failure', 'editor_failure'):
            calls.clear()
            started = time.monotonic()
            run = subprocess.run([sys.executable, '-B', str(harness), '--questions',
                                  str(ROOT/'demo_questions.csv'), '--kb', str(ROOT/'Knowledge-base'),
                                  '--output', str(output)], cwd=ROOT/'work', env=environment,
                                 capture_output=True, text=True, encoding='utf-8', timeout=100)
            assert run.returncode == 0, run.stderr
            with output.open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f); rows = list(reader)
                assert reader.fieldnames == ['id', 'answer']
            assert [r['id'] for r in rows] == ids and len(ids) == len(set(ids))
            assert all(0 < len(r['answer']) <= 10000 for r in rows)
            expected_answer = 'ТЕСТ: черновик.' if mode == 'editor_failure' else 'ТЕСТ: исправленный ответ.'
            assert all(r['answer'] == expected_answer for r in rows), (mode,
                sorted({r['answer'] for r in rows}), run.stderr[-4000:])
            assert all(calls.count(stage) == len(ids) for stage in ('rerank', 'draft', 'edit'))
            reports.append(dict(mode=mode, rows=len(rows), calls=len(calls),
                                seconds=round(time.monotonic()-started, 3), passed=True))
    finally:
        server.shutdown(); server.server_close(); thread.join()
        for path in paths:
            path.unlink(missing_ok=True)
    return dict(kind='synthetic-http-not-model-quality', sha256=manifest['sha256'], runs=reports)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
