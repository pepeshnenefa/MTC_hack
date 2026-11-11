"""Unified solution: one configuration for every weight and switch (synthetic data only)."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
import config  # noqa: E402
import context  # noqa: E402
import hybrid  # noqa: E402
import retrieval  # noqa: E402
import solution  # noqa: E402

DOCS = [
    ('00-reference/02-product-scope-and-limits.md',
     '# Ограничения\n\n## Солнечные электростанции\n\n- Сетевая станция отключается вместе с сетью.\n\n'
     '## Облачная платформа учёта\n\n- Платформа не выставляет счета.\n'),
    ('00-reference/01-numbers-and-indicators.md',
     '# Числа\n\n## Солнечные станции\n\nТиповой диапазон 15-500 кВт.\n'),
    ('20-products/23-solar.md',
     '# Солнечные электростанции под ключ\n\n## Размещение\n\nСкатная кровля подходит лучше всего.\n\n'
     '## Чего утверждать нельзя\n\n- Обещать автономию на сетевой станции.\n\n'
     '## Оборудование\n\nИнвертор и панели поставляются в комплекте.\n'),
    ('20-products/22-askue.md',
     '# Облачная платформа учёта электроэнергии\n\n## Что это\n\nАвтоматический сбор показаний приборов.\n'),
]


def search():
    return retrieval.Search(retrieval.chunks(DOCS))


class ConfigTests(unittest.TestCase):
    def tearDown(self):
        config.reload()

    def test_environment_overrides_defaults_and_rejects_garbage(self):
        with patch.dict(os.environ, {'ET_TOP_K': '5', 'ET_W_DENSE': '2.5', 'ET_PROMPT': 'v2'}):
            config.reload()
            self.assertEqual((config.get('TOP_K'), config.get('W_DENSE'), config.get('PROMPT')), (5, 2.5, 'v2'))
        with patch.dict(os.environ, {'ET_W_DENSE': 'nan'}):
            with self.assertRaises(ValueError):
                config.reload()
        config.reload()
        self.assertEqual(config.get('TOP_K'), config.DEFAULTS['TOP_K'])

    def test_platform_top_k_is_honoured_like_context_v1_unless_switched_off(self):
        with patch.dict(os.environ, {'TOP_K': '30'}):
            config.reload()
            self.assertEqual(config.get('TOP_K'), 30)
        with patch.dict(os.environ, {'TOP_K': '30', 'ET_TOP_K': '7'}):
            config.reload()
            self.assertEqual(config.get('TOP_K'), 7)
        with patch.dict(os.environ, {'TOP_K': '30', 'ET_PLATFORM_TOP_K': '0'}):
            config.reload()
            self.assertEqual(config.get('TOP_K'), config.DEFAULTS['TOP_K'])

    def test_override_is_temporary(self):
        before = config.get('W_HISTORY')
        with config.override(W_HISTORY=0):
            self.assertEqual(config.get('W_HISTORY'), 0)
        self.assertEqual(config.get('W_HISTORY'), before)


class GuardrailTests(unittest.TestCase):
    def test_limits_of_the_active_product_are_added_last_and_only_when_enabled(self):
        index = search()
        question = 'Скатная кровля подходит для панелей лучше всего?'
        with config.override(LIMITS_INJECT=0, TOP_K=1, SUPPORT_EXTRAS=0, PRIMARY=1):
            _, report = context.pack_context(question, index.find(question, k=36), limit=1)
            self.assertFalse(any(s.get('forced') for s in report['included']))
            self.assertEqual(len(report['included']), 1)
        with config.override(LIMITS_INJECT=1, TOP_K=1, SUPPORT_EXTRAS=0, PRIMARY=1):
            text, report = context.pack_context(question, index.find(question, k=36), limit=1)
            self.assertIn('отключается вместе с сетью', text)
            self.assertNotIn('не выставляет счета', text)  # another product's limits stay out
            self.assertTrue(report['included'][-1].get('forced'))
        with config.override(LIMITS_INJECT=2, TOP_K=1, SUPPORT_EXTRAS=0, PRIMARY=1):
            text, _ = context.pack_context(question, index.find(question, k=36), limit=1)
            self.assertIn('Обещать автономию', text)

    def test_numbers_reference_only_when_the_client_gives_a_number(self):
        index = search()
        with config.override(NUMBERS_INJECT=1, TOP_K=1, SUPPORT_EXTRAS=0, PRIMARY=1):
            plain, _ = context.pack_context('Солнечная станция на кровле?', index.find('Солнечная станция на кровле?', k=36), limit=1)
            numeric, _ = context.pack_context('Солнечная станция на 60 кВт на кровле?',
                                              index.find('Солнечная станция на 60 кВт на кровле?', k=36), limit=1)
        self.assertNotIn('15-500', plain)
        self.assertIn('15-500', numeric)

    def test_guardrail_is_not_duplicated_when_already_retrieved(self):
        index = search()
        question = 'Солнечная сетевая станция отключается вместе с сетью?'
        with config.override(LIMITS_INJECT=1, TOP_K=12):
            text, _ = context.pack_context(question, index.find(question, k=36), limit=12)
        self.assertEqual(text.count('Сетевая станция отключается вместе с сетью'), 1)


class FusionTests(unittest.TestCase):
    def test_dense_similarity_can_rescue_a_section_without_shared_words(self):
        index = search()
        question = 'Куда крепить модули?'
        target = next(i for i, c in enumerate(index.items) if c['heading'] == 'Размещение')
        dense = [0.1] * len(index.items)
        dense[target] = 0.9
        with config.override(W_DENSE=0):
            self.assertNotEqual((index.find(question, k=1, dense=dense) or [{'heading': None}])[0]['heading'], 'Размещение')
        with config.override(W_DENSE=3):
            self.assertEqual(index.find(question, k=1, dense=dense)[0]['heading'], 'Размещение')

    def test_weighted_policy_uses_config_weight_and_falls_back_for_unknown_question(self):
        index = search()
        vectors = [[1.0, 0.0] for _ in index.items]
        target = next(i for i, c in enumerate(index.items) if c['heading'] == 'Размещение')
        vectors[target] = [0.0, 1.0]
        fused = hybrid.HybridSearch(index, vectors, {'Куда крепить модули?': [0.0, 1.0]}, policy='weighted')
        with config.override(W_DENSE=3):
            self.assertEqual(fused.find('Куда крепить модули?', k=1)[0]['heading'], 'Размещение')
            self.assertEqual(fused.find('Вопрос без вектора про инвертор', k=3), index.find('Вопрос без вектора про инвертор', k=3))

    def test_parallel_embedding_keeps_order(self):
        class API:
            def post(self, endpoint, body):
                return {'data': [{'index': i, 'embedding': [float(len(text)), 1.0]} for i, text in enumerate(body['input'])]}
        texts = ['т' * n for n in range(1, 41)]
        self.assertEqual(hybrid.embed(API(), 'm', texts, workers=3), hybrid.embed(API(), 'm', texts, workers=1))


class TransportTests(unittest.TestCase):
    ENV = {'LLM_BASE_URL': 'http://proxy.invalid/v1', 'LLM_API_KEY': 'k', 'LLM_MODEL': 'm'}

    def run_with(self, failures):
        import http.client
        import io
        import json
        import socket
        errors = {'reset': ConnectionResetError('reset'), 'dropped': http.client.RemoteDisconnected('closed'),
                  'read-timeout': socket.timeout('timed out'), 'partial': http.client.IncompleteRead(b'{')}
        calls = []

        def transport(request, timeout):
            calls.append(1)
            if len(calls) <= len(failures):
                raise errors[failures[len(calls) - 1]]
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'Готовый ответ'}, 'finish_reason': 'stop'}]}).encode())
        with patch.dict(os.environ, self.ENV, clear=True), patch.object(solution, 'urlopen', side_effect=transport), \
                patch.object(solution.time, 'sleep'):
            answer = solution.API().answer('Вопрос?', [{'source': 'a', 'heading': 'h', 'text': 'текст'}])
        return answer, len(calls)

    def test_connection_dropped_while_reading_is_retried_not_turned_into_a_stub(self):
        for kind in ('reset', 'dropped', 'read-timeout', 'partial'):
            with self.subTest(kind=kind):
                self.assertEqual(self.run_with([kind]), ('Готовый ответ', 2))

    def test_repeated_drops_are_still_bounded(self):
        answer, calls = self.run_with(['dropped'] * 4)
        self.assertEqual(answer, 'Готовый ответ')
        self.assertEqual(calls, 5)


class ThinkingSwitchTests(unittest.TestCase):
    ENV = {'LLM_BASE_URL': 'http://proxy.invalid/v1', 'LLM_API_KEY': 'k', 'LLM_MODEL': 'm'}

    def tearDown(self):
        solution.API.template_arguments_rejected = False
        config.reload()

    def test_default_request_has_no_template_arguments(self):
        import io
        import json
        seen = []

        def transport(request, timeout):
            seen.append(json.loads(request.data))
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'Ответ'}, 'finish_reason': 'stop'}]}).encode())
        with patch.dict(os.environ, self.ENV, clear=True), patch.object(solution, 'urlopen', side_effect=transport):
            solution.API().answer('Вопрос?', [{'source': 'a', 'heading': 'h', 'text': 'текст'}])
        self.assertNotIn('chat_template_kwargs', seen[0])

    def test_rejected_switch_is_dropped_once_and_answers_survive(self):
        import io
        import json
        from urllib.error import HTTPError
        seen = []

        def transport(request, timeout):
            body = json.loads(request.data)
            seen.append(body)
            if 'chat_template_kwargs' in body:
                raise HTTPError(request.full_url, 400, 'Bad Request', {}, io.BytesIO(b'{}'))
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'Ответ'}, 'finish_reason': 'stop'}]}).encode())
        with patch.dict(os.environ, dict(self.ENV, ET_THINKING='on'), clear=True), \
                patch.object(solution, 'urlopen', side_effect=transport), patch.object(solution.time, 'sleep'):
            api = solution.API()
            first = api.answer('Вопрос?', [{'source': 'a', 'heading': 'h', 'text': 'текст'}])
            second = api.answer('Ещё вопрос?', [{'source': 'a', 'heading': 'h', 'text': 'текст'}])
        self.assertEqual((first, second), ('Ответ', 'Ответ'))
        self.assertEqual(['chat_template_kwargs' in body for body in seen], [True, False, False])


class DefaultBehaviourTests(unittest.TestCase):
    def test_defaults_are_context_v1_generation_with_lexical_search(self):
        config.reload()
        self.assertEqual((config.get('PROMPT'), config.get('TOP_K'), config.get('LONG_WORDS'), config.get('USE_HYBRID'),
                          config.get('W_COLLOQUIAL'), config.get('DIALOG_FORMAT')), ('full', 12, 0, 0, 0.0, 'raw'))
        self.assertEqual(solution.user_message('Вопрос?', 'ДОК'), 'ДОКУМЕНТЫ:\nДОК\n\nДИАЛОГ ИЛИ ВОПРОС:\nВопрос?')


class PromptTests(unittest.TestCase):
    def test_structured_dialogue_separates_last_client_turn_and_labels_assistant(self):
        dialogue = 'Пользователь: У нас склад.\nАссистент: Станция продолжит работать.\nПользователь: Это точно?'
        with config.override(DIALOG_FORMAT='structured', PROMPT='v2'):
            message = solution.user_message(dialogue, 'ДОК')
        history, last = message.split('ПОСЛЕДНЯЯ РЕПЛИКА КЛИЕНТА')
        self.assertIn('Консультант: Станция продолжит работать.', history)
        self.assertIn('не являются источником фактов', history)
        self.assertIn('Это точно?', last)
        self.assertNotIn('У нас склад', last)
        with config.override(DIALOG_FORMAT='raw', PROMPT='short'):
            self.assertIn('ДИАЛОГ ИЛИ ВОПРОС:\n' + dialogue, solution.user_message(dialogue, 'ДОК'))

    def test_prompt_variants_are_selectable_and_full_is_context_v1_text(self):
        with config.override(PROMPT='full'):
            self.assertTrue(solution.system_prompt().startswith('Ты консультант МТС Energytool.'))
            self.assertNotIn('2–4 предложения', solution.user_message('Вопрос?', 'ДОК'))
        with config.override(PROMPT='v2'):
            self.assertIn('по этому же продукту', solution.system_prompt())
        with config.override(PROMPT='v2', USE_EXAMPLES=0):
            self.assertNotIn('ПРИМЕРЫ ФОРМЫ', solution.system_prompt())

    def test_no_demo_reference_answer_is_embedded_in_any_prompt(self):
        import csv
        path = ROOT / 'demo_output.csv'
        if not path.exists():
            self.skipTest('demo data is local only')
        prompts = ' '.join([solution.SYSTEM, solution.EXAMPLES, solution.SYSTEM_V2, solution.EXAMPLES_V2,
                            solution.SYSTEM_FULL, solution.REMINDER, solution.REMINDER_V2])
        with path.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for sentence in row['answer'].split('. '):
                if len(sentence) > 40:
                    self.assertNotIn(sentence, prompts)


if __name__ == '__main__':
    unittest.main()
