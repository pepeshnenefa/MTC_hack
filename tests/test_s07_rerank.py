"""s07-rerank: the semantic07 archive plus the Qwen3 Reranker stage. Synthetic data only."""
import contextlib
import importlib.util
import json
import os
import sys
import threading
import time
import types
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('s07_rerank_builder', ROOT / 'tools/build_s07_rerank.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

ORDER = ('config', 'retrieval', 'context', 'corrections', 'hybrid', 'hierarchy', 'batch_io', 'rerank', 'solution')
ENV = {'LLM_BASE_URL': 'http://test.invalid', 'LLM_API_KEY': 'test-key', 'LLM_MODEL': 'test-model', 'TOP_K': '12'}


@contextlib.contextmanager
def loaded(content):
    """Import the archive modules from memory; earlier modules of the same names are restored."""
    names = [n for n in ORDER if n + '.py' in content]
    previous = {n: sys.modules.pop(n, None) for n in names}
    try:
        for name in names:
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / 'dist' / (name + '.py'))
            sys.modules[name] = module
            exec(compile(content[name + '.py'], module.__file__, 'exec'), module.__dict__)
        yield {n: sys.modules[n] for n in names}
    finally:
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


def pool(size=36):
    return [dict(source='doc%02d.md' % i, heading='Раздел %d' % i, title='Продукт / Раздел %d' % i,
                 text='Текст раздела %d.' % i, score=100 - i) for i in range(size)]


class FakeAPI:
    def __init__(self, deadline=None):
        self.base, self.key, self.reranker, self.deadline = 'http://test.invalid', 'test-key', 'test-reranker', deadline


class Archive(unittest.TestCase):
    def test_only_two_files_change_and_stage_is_added(self):
        base, content = builder.base_files(), builder.files()
        self.assertEqual(list(content), list(base) + ['rerank.py'])
        self.assertEqual(sorted(n for n in base if base[n] != content[n]), ['config.py', 'solution.py'])
        self.assertEqual(content['rerank.py'], (ROOT / 'submission/rerank.py').read_bytes())
        for name in ('retrieval.py', 'hybrid.py', 'context.py', 'batch_io.py', 'requirements.txt'):
            self.assertEqual(content[name], base[name])

    def test_search_settings_are_those_of_semantic07(self):
        with patch.dict(os.environ, ENV, clear=True):
            with loaded(builder.base_files()) as old:
                before = dict(old['config'].DEFAULTS)
            with loaded(builder.files()) as new:
                after = dict(new['config'].DEFAULTS)
        added = {k for k in after if k not in before}
        self.assertEqual(added, {'RERANK_KEEP', 'RERANK_DOC_CHARS', 'RERANK_TIMEOUT', 'RERANK_ATTEMPTS',
                                 'RERANK_RESERVE', 'RERANK_MAX_BAD', 'RERANK_SLOW'})
        self.assertEqual({k: v for k, v in after.items() if k in before and v != before[k]}, {'USE_RERANKER': 1})
        self.assertEqual((after['M_DENSE_MISMATCH'], after['W_DENSE'], after['USE_HYBRID'], after['TOP_K']), (0.7, 3.0, 1, 12))
        self.assertEqual(after['RERANK_KEEP'], 12)

    def test_built_archive_matches_memory_content(self):
        archive = ROOT / 'dist/energytool-s07-rerank.zip'
        if not archive.exists():
            self.skipTest('archive not built')
        with zipfile.ZipFile(archive) as z:
            self.assertEqual({n: z.read(n) for n in z.namelist()}, builder.files())


class Ranking(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, ENV, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.modules = loaded(builder.files())
        self.m = self.modules.__enter__()
        self.addCleanup(self.modules.__exit__, None, None, None)
        self.rerank = self.m['rerank']

    def test_score_decides_the_order(self):
        data = {'results': [{'index': 2, 'relevance_score': 0.1}, {'index': 0, 'relevance_score': 0.9},
                            {'index': 1, 'relevance_score': 0.5}]}
        self.assertEqual(self.rerank.ranking(data, 3), ([0, 1, 2], [0.9, 0.5, 0.1]))

    def test_response_order_without_scores_and_data_key(self):
        self.assertEqual(self.rerank.ranking({'data': [{'index': 1}, {'index': 0}]}, 2)[0], [1, 0])

    def test_invalid_responses_are_rejected(self):
        for data in ({}, {'results': []}, {'results': [{'index': 3}]}, {'results': [{'index': -1}]},
                     {'results': [{'index': True}]}, {'results': [{'index': '1'}]},
                     {'results': [{'index': 1}, {'index': 1}]}, {'results': [1]}, [], None):
            with self.assertRaises(ValueError, msg=repr(data)):
                self.rerank.ranking(data, 3)


class Stage(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, ENV, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.modules = loaded(builder.files())
        self.m = self.modules.__enter__()
        self.addCleanup(self.modules.__exit__, None, None, None)
        self.config, self.rerank = self.m['config'], self.m['rerank']
        self.stage = self.rerank.Stage(FakeAPI(deadline=time.monotonic() + 600))
        self.bodies = []

    def reply(self, data):
        def post(body):
            self.bodies.append(body)
            if isinstance(data, Exception):
                raise data
            return data
        self.stage.post = post

    def test_reranker_choice_heads_the_pool_and_the_rest_keeps_hybrid_order(self):
        sources = pool() + [dict(pool(1)[0], source='guard.md', forced=True)]
        choice = [30, 5, 0, 17, 2, 9, 35, 11, 20, 1, 3, 4]
        self.reply({'results': [{'index': i, 'relevance_score': 1 - n / 100} for n, i in enumerate(choice)]})
        ordered, info = self.stage.order('Вопрос', sources)
        self.assertEqual([s['source'] for s in ordered[:12]], ['doc%02d.md' % i for i in choice])
        rest = [i for i in range(36) if i not in choice]
        self.assertEqual([s['source'] for s in ordered[12:36]], ['doc%02d.md' % i for i in rest])
        self.assertEqual(ordered[36]['source'], 'guard.md')
        self.assertEqual(len(ordered), len(sources))
        self.assertEqual((info['status'], info['picked'], info['filled']), ('ok', choice, 0))
        body = self.bodies[0]
        self.assertEqual((body['model'], body['top_n'], len(body['documents'])), ('test-reranker', 12, 36))
        self.assertEqual(body['documents'][3], 'Продукт / Раздел 3\nТекст раздела 3.')
        self.assertEqual(body['query'], self.m['retrieval'].query('Вопрос'))

    def test_fewer_results_are_filled_in_hybrid_order(self):
        self.reply({'results': [{'index': 7, 'relevance_score': 0.9}, {'index': 3, 'relevance_score': 0.8}]})
        ordered, info = self.stage.order('Вопрос', pool())
        self.assertEqual([s['source'] for s in ordered[:12]],
                         ['doc%02d.md' % i for i in [7, 3, 0, 1, 2, 4, 5, 6, 8, 9, 10, 11]])
        self.assertEqual(info['filled'], 10)

    def test_documents_are_cut_for_the_reranker_only(self):
        sources = pool(3)
        sources[0]['text'] = 'я' * 9000
        self.reply({'results': [{'index': 0, 'relevance_score': 1.0}]})
        ordered, _ = self.stage.order('Вопрос', sources)
        self.assertEqual(len(self.bodies[0]['documents'][0]), 6000)
        self.assertEqual(len(ordered[0]['text']), 9000)

    def test_switched_off_gives_the_same_list_without_requests(self):
        self.reply({'results': [{'index': 1}]})
        sources = pool()
        with self.config.override(USE_RERANKER=0):
            ordered, info = self.stage.order('Вопрос', sources)
        self.assertIs(ordered, sources)
        self.assertEqual((info, self.bodies), ({'status': 'off'}, []))

    def test_failure_keeps_the_hybrid_order(self):
        sources = pool()
        for error in (TimeoutError('slow'), ValueError('bad'), HTTPError('u', 503, 'x', {}, None)):
            self.reply(error)
            ordered, info = self.stage.order('Вопрос', sources)
            self.assertIs(ordered, sources)
            self.assertEqual(info['status'], 'fallback')

    def test_three_consecutive_failures_switch_the_stage_off(self):
        self.reply(TimeoutError('slow'))
        for _ in range(3):
            self.stage.order('Вопрос', pool())
        self.reply({'results': [{'index': 1}]})
        ordered, info = self.stage.order('Вопрос', pool())
        self.assertEqual((info['status'], len(self.bodies)), ('disabled', 3))

    def test_a_success_resets_the_failure_count(self):
        for _ in range(3):  # two failures, then a success, three times over
            self.reply(TimeoutError('slow'))
            self.stage.order('Вопрос', pool())
            self.stage.order('Вопрос', pool())
            self.reply({'results': [{'index': 1}]})
            self.assertEqual(self.stage.order('Вопрос', pool())[1]['status'], 'ok')
            self.assertEqual(self.stage.bad, 0)
        self.assertIsNone(self.stage.disabled)

    def test_missing_endpoint_or_key_switches_off_at_once(self):
        for code in (404, 401):
            self.stage = self.rerank.Stage(FakeAPI(deadline=time.monotonic() + 600))
            self.reply(HTTPError('u', code, 'x', {}, None))
            self.stage.order('Вопрос', pool())
            self.assertEqual(self.stage.order('Вопрос', pool())[1]['status'], 'disabled')

    def test_slow_successes_are_used_but_switch_off_after_three(self):
        self.reply({'results': [{'index': 1}]})
        with self.config.override(RERANK_SLOW=-1):
            for _ in range(3):
                self.assertEqual(self.stage.order('Вопрос', pool())[1]['status'], 'ok')
        self.assertEqual(self.stage.order('Вопрос', pool())[1]['status'], 'disabled')

    def test_limits_test_is_the_one_of_context_py(self):
        source = builder.files()['context.py'].decode()
        self.assertIn("re.search(r'" + self.rerank.LIMITS.pattern + "',title.lower())", source)

    def test_limits_sections_of_semantic07_stay_in_the_context(self):
        sources = pool()
        for i, title in ((4, 'Платформа учёта / Что мы не делаем'), (9, 'Платформа учёта / Чего утверждать нельзя'),
                         (30, 'Платформа учёта / Ограничения')):
            sources[i] = dict(sources[i], title=title, source='doc%02d.md' % i)
        choice = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32]
        self.reply({'results': [{'index': i, 'relevance_score': 1 - n / 100} for n, i in enumerate(choice)]})
        ordered, info = self.stage.order('Вопрос', sources)
        semantic07 = self.m['context'].select_sources('Вопрос', sources, 12)
        kept = [s['source'] for s in semantic07 if self.rerank.limits_section(s)]
        self.assertEqual(kept, ['doc04.md', 'doc09.md'])  # doc30 is outside the semantic07 context
        head = [s['source'] for s in ordered[:12]]
        self.assertEqual(head[:10], ['doc%02d.md' % i for i in choice[:10]])
        self.assertEqual(set(head[10:]), {'doc04.md', 'doc09.md'})  # the two weakest picks gave way
        self.assertEqual(info['restored_limits'], 2)
        self.assertEqual(sorted(s['source'] for s in ordered), sorted(s['source'] for s in sources))
        self.assertEqual([s['source'] for s in ordered[12:14]], ['doc31.md', 'doc32.md'])  # displaced picks next
        included = {s['source'] for s in self.m['context'].pack_context('Вопрос', ordered, 12)[1]['included']}
        self.assertTrue({'doc04.md', 'doc09.md'} <= included)

    def test_a_limits_section_already_chosen_is_not_moved(self):
        sources = pool()
        sources[4] = dict(sources[4], title='Платформа учёта / Что мы не делаем')
        choice = [4, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31]
        self.reply({'results': [{'index': i, 'relevance_score': 1 - n / 100} for n, i in enumerate(choice)]})
        ordered, info = self.stage.order('Вопрос', sources)
        self.assertEqual([s['source'] for s in ordered[:12]], ['doc%02d.md' % i for i in choice])
        self.assertEqual(info['restored_limits'], 0)

    def test_skipped_when_the_run_budget_is_nearly_spent(self):
        self.stage = self.rerank.Stage(FakeAPI(deadline=time.monotonic() + 30))
        self.reply({'results': [{'index': 1}]})
        sources = pool()
        ordered, info = self.stage.order('Вопрос', sources)
        self.assertIs(ordered, sources)
        self.assertEqual((info, self.bodies), ({'status': 'skipped', 'reason': 'time'}, []))


class Server:
    """Local OpenAI-compatible stand-in: records every request; /rerank behaviour is chosen per test."""

    def __init__(self, rerank):
        calls = self.calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((self.path, body, self.headers['Authorization']))
                if self.path.endswith('/rerank'):
                    status, data = rerank(body)
                else:
                    status, data = 200, {'choices': [{'message': {'content': 'Ответ.'}, 'finish_reason': 'stop'}]}
                raw = json.dumps(data).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = 'http://127.0.0.1:%d' % self.server.server_port

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Answer(unittest.TestCase):
    """The request to the generation model: identical to semantic07 when the stage is off or fails."""

    def chat_body(self, content, rerank, **env):
        server = Server(rerank)
        try:
            with patch.dict(os.environ, dict(ENV, LLM_BASE_URL=server.url, **env), clear=True):
                with loaded(content) as m:
                    api = m['solution'].API(deadline=time.monotonic() + 600)
                    api.answer('Пользователь: Вопрос', pool())
            return [c for c in server.calls if c[0] == '/chat/completions'][0][1], server.calls
        finally:
            server.close()

    def test_off_and_failures_send_exactly_the_semantic07_request(self):
        original, _ = self.chat_body(builder.base_files(), lambda body: (500, {}))
        off, calls = self.chat_body(builder.files(), lambda body: (500, {}), ET_USE_RERANKER='0')
        self.assertEqual(off, original)
        self.assertEqual([c[0] for c in calls], ['/chat/completions'])
        for status, data in ((503, {}), (404, {}), (200, {'results': [{'index': 99}]}), (200, {'oops': 1})):
            failed, calls = self.chat_body(builder.files(), lambda body: (status, data))
            self.assertEqual(failed, original, (status, data))
            self.assertEqual(calls[-1][0], '/chat/completions')
        self.assertEqual([c[0] for c in calls], ['/rerank', '/chat/completions'])

    def test_retry_once_after_503(self):
        _, calls = self.chat_body(builder.files(), lambda body: (503, {}))
        self.assertEqual([c[0] for c in calls], ['/rerank', '/rerank', '/chat/completions'])
        self.assertTrue(all(c[2] == 'Bearer test-key' for c in calls))

    def test_reranker_choice_is_what_the_model_reads_first(self):
        def rerank(body):
            return 200, {'results': [{'index': i, 'relevance_score': 1 - n / 100}
                                     for n, i in enumerate([33, 34, 35, 0, 1, 2, 3, 4, 5, 6, 7, 8])]}
        body, calls = self.chat_body(builder.files(), rerank, RERANKER_MODEL='platform-reranker')
        self.assertEqual(calls[0][1]['model'], 'platform-reranker')
        context = body['messages'][1]['content']
        self.assertLess(context.index('doc33.md'), context.index('doc00.md'))
        self.assertNotIn('doc20.md', context)
        default, calls = self.chat_body(builder.files(), rerank)
        self.assertEqual(calls[0][1]['model'], 'qwen/qwen3-reranker-8b')
        self.assertEqual(default, body)


if __name__ == '__main__':
    unittest.main()
