"""Exercise the actual candidate modules in memory, without creating folders."""
import ast
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rr_builder', ROOT / 'tools/build_rerank_review_submission.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@contextlib.contextmanager
def loaded():
    content = builder.files()
    names = ('batch_io', 'retrieval', 'context', 'rerank_review', 'solution')
    previous = {n: sys.modules.pop(n, None) for n in names}
    try:
        for name in names:
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / 'submission' / (name + '.py'))
            sys.modules[name] = module
            exec(compile(content[name + '.py'], module.__file__, 'exec'), module.__dict__)
        yield sys.modules['solution'], sys.modules['rerank_review']
    finally:
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


def response(text='Готовый ответ.', reason='stop'):
    return {'choices': [{'message': {'content': text}, 'finish_reason': reason}]}


def sources():
    return [dict(source=str(i)+'.md', heading='Раздел', title='Раздел',
                 text='Проверочный текст '+str(i), score=20-i) for i in range(15)]


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://test.invalid', 'LLM_API_KEY': 'test',
            'LLM_MODEL': 'test-model', 'RERANKER_MODEL': 'test-reranker',
            'USE_RERANKER': '1', 'USE_ANSWER_REVIEW': '1', 'TOP_K': '12'}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.modules = loaded()
        self.solution, self.rr = self.modules.__enter__()
        self.addCleanup(self.modules.__exit__, None, None, None)
        self.api = self.solution.API(deadline=time.monotonic()+120)

    def test_exact_baseline_and_source(self):
        content = builder.files()
        self.assertEqual(len(content), 6)
        with zipfile.ZipFile(ROOT/'dist/energytool-context-v1.zip') as z:
            for n in ('context.py', 'retrieval.py', 'batch_io.py', 'requirements.txt'):
                self.assertEqual(content[n], z.read(n))
            def system(data):
                return next(ast.literal_eval(n.value) for n in ast.parse(data)
                            .body if isinstance(n, ast.Assign) and
                            any(isinstance(t, ast.Name) and t.id == 'SYSTEM' for t in n.targets))
            self.assertEqual(system(content['solution.py']), system(z.read('solution.py')))
        self.assertEqual(content['rerank_review.py'], (ROOT/'submission/rerank_review.py').read_bytes())

    def test_ranking_keeps_anchors_and_entire_pool(self):
        pool = sources()
        with patch.object(self.rr, 'optional_post', return_value={'results': [{'index': 14}, {'index': 9}]}):
            ranked = self.rr.rerank(self.api, 'Вопрос', pool)
        self.assertEqual(ranked[:5], [*pool[:3], pool[14], pool[9]])
        self.assertEqual(len(ranked), len(pool))
        self.assertEqual({id(s) for s in ranked}, {id(s) for s in pool})

    def test_malformed_rankings_fallback(self):
        pool = sources()
        for data in ({}, {'results': []}, {'results': [{'index': -1}]},
                     {'results': [{'index': True}]}, {'results': [{'index': 99}]},
                     {'results': [{'index': 2}, {'index': 2}]},
                     {'results': [{'index': 2, 'relevance_score': float('nan')}]}):
            with self.subTest(data=data), patch.object(self.rr, 'optional_post', return_value=data):
                self.assertIs(self.rr.rerank(self.api, 'Вопрос', pool), pool)

    def test_editor_receives_same_evidence_and_original_dialogue(self):
        question = 'Пользователь: 20 объектов\nПользователь: Ошибся, 12 объектов'
        with patch.object(self.api, 'post', return_value=response('Черновик.')) as draft, \
             patch.object(self.rr, 'optional_post', side_effect=[{'results': [{'index': 14}]}, response('Исправлено.')]) as optional:
            self.assertEqual(self.api.answer(question, sources()), 'Исправлено.')
        self.assertEqual(draft.call_count, 1)
        self.assertEqual(optional.call_count, 2)
        first = draft.call_args.args[1]['messages'][1]['content']
        edit = optional.call_args_list[1].args[2]['messages']
        self.assertEqual([m['role'] for m in edit], ['system', 'user'])
        self.assertTrue(edit[1]['content'].startswith(first))
        self.assertIn(question, edit[1]['content'])
        self.assertIn('Черновик.', edit[1]['content'])

    def test_editor_failures_keep_draft(self):
        os.environ['USE_RERANKER'] = '0'
        for result in (response(None), response('Обрыв', 'length'), response('<think>секрет'),
                       response('x'*10001), {}, TimeoutError()):
            kwargs = {'side_effect': result} if isinstance(result, Exception) else {'return_value': result}
            with self.subTest(result=type(result).__name__), patch.object(self.api, 'post', return_value=response('Черновик.')), \
                 patch.object(self.rr, 'optional_post', **kwargs):
                self.assertEqual(self.api.answer('Вопрос', sources()), 'Черновик.')

    def test_null_draft_recovers_without_draft_or_extra_review(self):
        os.environ['USE_RERANKER'] = '0'
        with patch.object(self.api, 'post', return_value=response(None)), \
             patch.object(self.rr, 'optional_post', return_value=response('Восстановлено.')) as call:
            self.assertEqual(self.api.answer('Вопрос', sources()), 'Восстановлено.')
        self.assertEqual(call.call_count, 1)
        self.assertNotIn('ЧЕРНОВИК', call.call_args.args[2]['messages'][1]['content'])

    def test_empty_draft_and_failed_recovery_raise(self):
        os.environ['USE_RERANKER'] = '0'
        with patch.object(self.api, 'post', return_value=response(None)), \
             patch.object(self.rr, 'optional_post', side_effect=TimeoutError()), self.assertRaises(ValueError):
            self.api.answer('Вопрос', sources())

    def test_budget_skips_optional_stages(self):
        self.api.deadline = time.monotonic()+10
        with patch.object(self.api, 'post', return_value=response('Ответ.')), \
             patch.object(self.rr, 'optional_post') as call:
            self.assertEqual(self.api.answer('Вопрос', sources()), 'Ответ.')
        call.assert_not_called()

    def test_switches_and_session_isolation(self):
        os.environ.update(USE_RERANKER='0', USE_ANSWER_REVIEW='0')
        with patch.object(self.api, 'post', return_value=response('Ответ.')) as call, \
             patch.object(self.rr, 'optional_post') as optional:
            self.api.answer('Первая сессия', sources())
            self.api.answer('Вторая сессия', sources())
        self.assertNotIn('Первая сессия', str(call.call_args))
        optional.assert_not_called()

    def test_optional_timeout_does_not_mutate_shared_deadline(self):
        before = self.api.deadline
        with patch.object(self.rr, 'urlopen', side_effect=TimeoutError()) as call:
            with self.assertRaises(TimeoutError):
                self.rr.optional_post(self.api, '/rerank', {}, 10, 35)
        self.assertLessEqual(call.call_args.kwargs['timeout'], 10)
        self.assertEqual(before, self.api.deadline)

    def test_real_http_three_requests_and_csv_validation(self):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((self.path, body, self.headers.get('Authorization')))
                if self.path == '/rerank':
                    value = {'results': [{'index': 10}]}
                else:
                    value = response('Итоговый ответ.' if 'ЧЕРНОВИК' in body['messages'][1]['content'] else 'Черновик.')
                raw = json.dumps(value).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.api.base = 'http://127.0.0.1:'+str(server.server_port)
            result = self.api.answer('Вопрос', sources())
            self.assertEqual(result, 'Итоговый ответ.')
            self.assertEqual([x[0] for x in calls], ['/rerank', '/chat/completions', '/chat/completions'])
            self.assertTrue(all(x[2] == 'Bearer test' for x in calls))
            self.assertEqual(calls[1][1]['model'], 'test-model')
            self.assertEqual(calls[2][1]['model'], 'test-model')
            self.assertEqual(self.solution.normalize_answer(result), result)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()
