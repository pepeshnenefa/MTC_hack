"""Local stand-in for the platform proxy: one OpenAI-compatible address for chat and embeddings.

/chat/completions is forwarded to the local generation server; /embeddings is answered from the
bge-m3 cache built by tools/embed_cache.py (or forwarded to a local embedding server on a miss).
Development tool only; never packaged.

    python3 tools/local_proxy.py --port 8090
"""
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = {}


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def forward(self, base, body):
        request = Request(base + self.path, data=body, headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=600) as response:
            return response.read()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        try:
            if self.path.endswith('/embeddings'):
                texts = json.loads(body)['input']
                if all(digest(t) in CACHE for t in texts):
                    payload = json.dumps({'data': [{'index': i, 'embedding': CACHE[digest(t)]} for i, t in enumerate(texts)]}).encode()
                else:
                    payload = self.forward(self.server.embeddings, body)
            else:
                payload = self.forward(self.server.chat, body)
            self.send_response(200)
        except Exception as exc:
            payload = json.dumps({'error': type(exc).__name__}).encode()
            self.send_response(502)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--chat', default='http://127.0.0.1:8088')
    parser.add_argument('--embeddings', default='http://127.0.0.1:8089')
    args = parser.parse_args()
    cache = ROOT / 'work/emb/cache.json'
    if cache.exists():
        CACHE.update(json.loads(cache.read_text(encoding='utf-8')))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.chat, server.embeddings = args.chat, args.embeddings
    server.serve_forever()
