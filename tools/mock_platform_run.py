"""End-to-end run of an unmodified submission ZIP against a fake platform proxy. NOT a quality check.

The archive runs as `python solution.py` in a temporary folder with data/questions.csv and
data/knowledge-base.zip, exactly as packaged. A local server answers /embeddings (hash-based vectors)
and /chat/completions. Chat replies rotate between three kinds so that every grounded branch fires:
  1. an answer with the service lines and mode CONFIRMED;
  2. "no data" without evidence (grounded recheck);
  3. an answer with an invented phone and figure (correction call, then hand-over to a specialist).
Checks: CSV contract, no service lines in answers, calls per question, run time. Development tool only.

    python3 tools/mock_platform_run.py --archive dist/<name>.zip
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
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
from batch_io import read_questions, validate_answers  # noqa: E402

REPLIES = [
    'Монтаж выполняется по проекту, сроки зависят от объекта.\nОСНОВАНИЕ: «Монтаж на фасаде.»\nРЕЖИМ: ПОДТВЕРЖДЕНО',
    'В базе нет сведений по этому вопросу, могу передать специалисту.\nОСНОВАНИЕ: НЕТ\nРЕЖИМ: НЕТ ДАННЫХ',
    'Позвоните по телефону +7 999 123-45-67, стоимость 987654 рублей.\nОСНОВАНИЕ: НЕТ\nРЕЖИМ: ПОДТВЕРЖДЕНО',
]
SERVICE = ('ОСНОВАНИЕ', 'РЕЖИМ:', 'ОТВЕТ:')
KEY = 'mock-key'


def vector(text):
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    return [((b % 17) - 8) / 8 + 0.01 for b in digest] * 32  # 1024 dimensions, like bge-m3


def serve():
    state = {'calls': Counter(), 'models': set(), 'replies': 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with lock:
                if self.headers.get('Authorization') != 'Bearer ' + KEY:
                    state['calls']['bad_auth'] += 1
                state['models'].add(body.get('model'))
                if self.path.endswith('/embeddings'):
                    state['calls']['embeddings'] += 1
                    data = {'data': [{'index': i, 'embedding': vector(t)} for i, t in enumerate(body['input'])]}
                else:
                    state['calls']['chat'] += 1
                    correction = len(body['messages']) > 2  # assistant draft + feedback appended
                    if correction:
                        text = REPLIES[2]  # the correction stays wrong -> hand-over
                    else:
                        state['replies'] += 1
                        text = REPLIES[state['replies'] % 3]
                    data = {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
            raw = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def run(archive, questions, kb, output):
    server, state = serve()
    try:
        with tempfile.TemporaryDirectory() as temp:
            app = Path(temp) / 'app'
            with zipfile.ZipFile(archive) as package:
                package.extractall(app)
            data = app / 'data'
            data.mkdir()
            (data / 'questions.csv').write_bytes(Path(questions).read_bytes())
            with zipfile.ZipFile(data / 'knowledge-base.zip', 'w') as target:
                for path in sorted(Path(kb).rglob('*')):
                    if path.is_file() and path.suffix.lower() in ('.md', '.txt'):
                        target.write(path, path.relative_to(kb).as_posix())
            trace_path = Path(temp) / 'trace.jsonl'
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(('ET_', 'LLM_', 'OPENAI_', 'EMB_', 'TRACE_'))}
            env.update(LLM_BASE_URL='http://127.0.0.1:%d' % server.server_port, LLM_API_KEY=KEY,
                       LLM_MODEL='mock-generation-model', TRACE_FILE=str(trace_path))
            started = time.monotonic()
            process = subprocess.run([sys.executable, 'solution.py', '--output', str(output)], cwd=app,
                                     env=env, capture_output=True, text=True, encoding='utf-8', timeout=900)
            seconds = time.monotonic() - started
            trace = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()] \
                if trace_path.exists() else []
    finally:
        server.shutdown()
    if process.returncode != 0:
        raise SystemExit('solution.py failed (%d):\n%s' % (process.returncode, process.stderr[-3000:]))
    rows = read_questions(questions)
    count = validate_answers(output, rows)
    with open(output, encoding='utf-8', newline='') as stream:
        answers = [row['answer'] for row in csv.DictReader(stream)]
    grounding = [t.get('grounding') or {} for t in trace]
    report = {
        'kind': 'mock-platform-run', 'model_quality_evaluated': False,
        'rows': count, 'returncode': process.returncode, 'seconds': round(seconds, 1),
        'service_line_leaks': sum(any(m in a.upper() for m in SERVICE) for a in answers),
        'calls': dict(state['calls']), 'models': sorted(m for m in state['models'] if m),
        'calls_per_question': dict(sorted(Counter(t['calls'] for t in trace).items())),
        'handoffs': sum(bool(g.get('handoff')) for g in grounding),
        'rechecks': sum('recheck_mode' in g for g in grounding),
        'grounding_errors': sum('error' in g for g in grounding),
        'stderr_events': sorted({json.loads(l)['event'] for l in process.stderr.splitlines()
                                 if l.startswith('{') and '"event"' in l}),
    }
    if report['service_line_leaks'] or report['calls'].get('bad_auth') or report['grounding_errors']:
        report['passed'] = False
    else:
        report['passed'] = True
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--questions', type=Path, default=ROOT / 'demo_questions.csv')
    parser.add_argument('--kb', type=Path, default=ROOT / 'Knowledge-base')
    parser.add_argument('--output', type=Path, default=ROOT / 'work' / 'offline-check' / 'mock-platform.csv')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.archive, args.questions, args.kb, args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result['passed'] else 1)
