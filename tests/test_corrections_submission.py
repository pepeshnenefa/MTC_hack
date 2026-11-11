"""Correction behavior and exact-baseline integration; no real model calls."""
import contextlib
import csv
import importlib
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_corrections_submission import build
sys.path.insert(0, str(ROOT / 'submission'))
from corrections import clean_turns


@contextlib.contextmanager
def loaded(archive):
    names = ('solution', 'retrieval', 'context', 'batch_io', 'corrections')
    previous = {name: sys.modules.pop(name, None) for name in names}
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(archive) as package:
            package.extractall(temp)
        sys.path.insert(0, temp)
        try:
            yield importlib.import_module('solution')
        finally:
            sys.path.remove(temp)
            for name in names:
                sys.modules.pop(name, None)
                if previous[name] is not None:
                    sys.modules[name] = previous[name]


class CorrectionRules(unittest.TestCase):
    def test_numeric_preserves_other_parameters(self):
        result, changed = clean_turns(['У нас 20 объектов и 30 счетчиков.',
                                      'Ошибся, у нас 12 объектов. Можно начать?'])
        self.assertTrue(changed)
        self.assertNotIn('20', result[0])
        self.assertIn('30 счетчиков', result[0])
        self.assertIn('12 объектов', result[1])

    def test_literal_correction(self):
        result, changed = clean_turns(['У нас магазин, потребление высокое.',
                                      'Поправка: не магазин, а склад. Какие данные нужны?'])
        self.assertTrue(changed)
        self.assertNotIn('магазин', ' '.join(result))
        self.assertIn('потребление высокое', result[0])
        self.assertIn('склад', result[1])

    def test_repeated_corrections_and_followup(self):
        result, changed = clean_turns(['Есть 20 объектов.', 'Ошибся, 12 объектов.',
                                      'Исправляю: 15 объектов.', 'Что подготовить?'])
        self.assertTrue(changed)
        self.assertNotIn('20', ' '.join(result))
        self.assertNotIn('12', ' '.join(result))
        self.assertIn('15 объектов', result[2])
        self.assertEqual(result[-1], 'Что подготовить?')

    def test_decimal_and_unit_inflection(self):
        result, changed = clean_turns(['Мощность 120,5 кВт.', 'Поправка: 150.5 кВт.'])
        self.assertTrue(changed)
        self.assertNotIn('120,5', result[0])
        self.assertEqual(result[1], 'Поправка: 150.5 кВт.')
        result, changed = clean_turns(['Есть 21 счетчик.', 'Ошибся, 22 счетчика.'])
        self.assertTrue(changed)

    def test_no_cross_session_state(self):
        clean_turns(['Есть 20 объектов.', 'Ошибся, 12 объектов.'])
        self.assertEqual(clean_turns(['Есть 20 объектов.']), (['Есть 20 объектов.'], False))

    def test_ambiguous_and_non_corrections_unchanged(self):
        scenarios = [
            ['У нас 20 объектов.', 'Начнем с 12 объектов.'],
            ['У нас 20 объектов.', 'Ошибся? Может быть 12 объектов?'],
            ['У нас 20 объектов.', 'Поправка: если будет 12 объектов, подойдет?'],
            ['У нас 20 объектов.', 'Поправка: планируем 12 объектов.'],
            ['У нас 20 объектов.', 'Поправка: 12 или 15 объектов.'],
            ['У нас 20 объектов.', 'Поправка: от 12 до 15 объектов.'],
            ['У нас 20 объектов.', 'Поправка: 12 объектов и 30 счетчиков.'],
            ['В Москве 20 объектов, в Казани 30 объектов.', 'Ошибся, 12 объектов.'],
            ['У нас не 20 объектов.', 'Поправка: 12 объектов.'],
            ['У нас магазин.', 'Не магазин, а склад.'],
            ['У нас магазин и еще один магазин.', 'Поправка: не магазин, а склад.'],
            ['У нас магазин.', 'Поправка: не только магазин, а склад.'],
            ['У нас магазин.', 'Поправка: не завод, а склад.'],
            ['У нас 20 объектов.', 'Поправка: 20 объектов.'],
            ['В Москве 20 объектов.', 'Ошибся, в Казани 12 объектов.'],
            ['В Москве 20 объектов.', 'Ошибся, 12 объектов.'],
            ['У нас 20 объектов.', 'Поправка: у филиала 12 объектов.'],
            ['У нас 20 объектов.', 'Поправка: 12 объектов в другом городе.'],
            ['Есть магазин?', 'Поправка: не магазин, а склад.'],
        ]
        for turns in scenarios:
            with self.subTest(turns=turns):
                self.assertEqual(clean_turns(turns), (turns, False))


class ArchiveIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = ROOT / 'dist/energytool-context-v1.zip'
        cls.candidate = Path(cls.temp.name) / 'candidate.zip'
        cls.manifest = build(cls.base, cls.candidate)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_baseline_and_package(self):
        with zipfile.ZipFile(self.base) as base, zipfile.ZipFile(self.candidate) as candidate:
            self.assertEqual(set(candidate.namelist()), set(base.namelist()) | {'corrections.py'})
            self.assertIsNone(candidate.testzip())
            for name in base.namelist():
                if name != 'retrieval.py':
                    self.assertEqual(base.read(name), candidate.read(name), name)
            for name in candidate.namelist():
                self.assertNotIn('/', name)
                self.assertTrue(name.endswith('.py') or name == 'requirements.txt')
        with self.assertRaises(ValueError):
            build(self.base, self.candidate)
        with self.assertRaises(ValueError):
            build(self.candidate, Path(self.temp.name) / 'wrong.zip')

    def test_correction_reaches_search_not_generation(self):
        question = 'Пользователь: У нас магазин.\nАссистент: Уточните объект.\nПользователь: Поправка: не магазин, а склад.'
        with loaded(self.candidate) as solution:
            retrieval = sys.modules['retrieval']
            seen = []
            search = retrieval.Search(retrieval.chunks([('test.md', '# Объекты\nМагазин склад.')]))
            original_query = search._query
            def capture(text):
                seen.append(text)
                return original_query(text)
            search._query = capture
            sources = search.find(question)
            self.assertNotIn('магазин', seen[0])
            self.assertIn('склад', seen[0])
            payloads = []
            with patch.dict(os.environ, {'LLM_BASE_URL': 'http://test.invalid', 'LLM_API_KEY': 'test',
                                         'LLM_MODEL': 'test-model', 'USE_RERANKER': '0'}):
                api = solution.API()
                def post(endpoint, payload):
                    payloads.append(payload)
                    return {'choices': [{'message': {'content': 'Тестовый ответ.'}, 'finish_reason': 'stop'}]}
                api.post = post
                self.assertEqual(api.answer(question, sources), 'Тестовый ответ.')
            self.assertEqual(len(payloads), 1)
            self.assertTrue(payloads[0]['messages'][-1]['content'].endswith(question))

    def test_assistant_correction_ignored(self):
        question = 'Пользователь: У нас 20 объектов.\nАссистент: Поправка: 12 объектов.\nПользователь: Что подготовить?'
        with loaded(self.candidate) as solution:
            from corrections import search_text
            self.assertEqual(search_text(question, sys.modules['retrieval'].dialogue), question)

    def test_obsolete_number_no_longer_boosts_document(self):
        question = 'Пользователь: У нас 120 объектов.\nПользователь: Ошибся, у нас 150 объектов.'
        docs = [('old.md', '# Старое\n120 объектов.'), ('new.md', '# Новое\n150 объектов.')]
        scores = []
        for archive in (self.base, self.candidate):
            with loaded(archive) as solution:
                search = solution.Search(solution.chunks(docs))
                scores.append({s['source']: s['score'] for s in search.find(question)})
        self.assertLess(scores[1]['old.md'], scores[0]['old.md'])
        self.assertEqual(scores[1]['new.md'], scores[0]['new.md'])

    def test_extracted_runner_csv_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(self.candidate) as package:
                package.extractall(root)
            harness = root / 'offline.py'
            harness.write_text('''import solution
class Stub:
    def __init__(self, **kwargs): pass
    def answer(self, question, sources):
        return 'ТЕСТ ФОРМАТА: ' + str(len(sources))
solution.API = Stub
solution.main()
''', encoding='utf-8')
            result = subprocess.run([sys.executable, str(harness), '--questions',
                str(ROOT / 'demo_questions.csv'), '--kb', str(ROOT / 'Knowledge-base'),
                '--output', str(root / 'submission.csv')], cwd=root,
                capture_output=True, text=True, encoding='utf-8', timeout=60,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'MAX_WORKERS': '3', 'RUN_BUDGET_SECONDS': '50'})
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / 'submission.csv').open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, ['id', 'answer'])
            with (ROOT / 'demo_questions.csv').open(encoding='utf-8-sig', newline='') as stream:
                ids = [r['id'] for r in csv.DictReader(stream)]
            self.assertEqual([r['id'] for r in rows], ids)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(0 < len(r['answer']) <= 10000 for r in rows))

    def test_demo_sources_and_context_equal(self):
        with (ROOT / 'demo_questions.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        results = []
        for archive in (self.base, self.candidate):
            with loaded(archive) as solution:
                search = solution.Search(solution.chunks(solution.documents(ROOT / 'Knowledge-base')))
                results.append([(search.find(r['question'], k=36),
                                 solution.pack_context(r['question'], search.find(r['question'], k=36)))
                                for r in rows])
        self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
