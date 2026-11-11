"""prompt-v1: answer form, cleaning and bounded regeneration. No network."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'submission'))
import solution

# These tests describe the prompt-v1 behaviour, now selected by switches rather than by default.
BASE = {'LLM_BASE_URL': 'http://synthetic.invalid', 'LLM_API_KEY': 'synthetic', 'LLM_MODEL': 'synthetic',
        'ET_PROMPT': 'short', 'ET_LONG_WORDS': '95'}
SOURCES = [{'source': 'a.md', 'heading': 'h', 'text': 'Факт.'}]


def reply(content, reason='stop'):
    return {'choices': [{'message': {'content': content}, 'finish_reason': reason}]}


def run(replies):
    calls = []
    queue = list(replies)
    def post(api, endpoint, body):
        calls.append(body)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    with patch.dict(os.environ, BASE, clear=True), patch.object(solution.API, 'post', post):
        return solution.API().answer('Вопрос?', SOURCES), calls


class CleanTests(unittest.TestCase):
    def test_reasoning_markdown_and_labels_removed(self):
        self.assertEqual(solution.clean_answer('<think>план</think>**Да**, можно.'), 'Да, можно.')
        self.assertEqual(solution.clean_answer('план без открытия</think> Ответ: Да.'), 'Да.')
        self.assertEqual(solution.clean_answer('- Первое.\n- Второе.'), 'Первое. Второе.')
        self.assertEqual(solution.clean_answer('<think>незакрыто'), '')
        self.assertEqual(solution.clean_answer(None), '')
        self.assertEqual(solution.clean_answer([{'type': 'text', 'text': 'Да.'}]), 'Да.')


class GenerationTests(unittest.TestCase):
    def test_short_answer_single_call_with_reminder(self):
        answer, calls = run([reply('Да, можно.')])
        self.assertEqual(answer, 'Да, можно.')
        self.assertEqual(len(calls), 1)
        self.assertEqual([m['role'] for m in calls[0]['messages']], ['system', 'user'])
        self.assertTrue(calls[0]['messages'][1]['content'].endswith(solution.REMINDER))

    def test_null_content_is_regenerated(self):
        answer, calls = run([reply(None), reply('Да.')])
        self.assertEqual((answer, len(calls)), ('Да.', 2))

    def test_malformed_response_is_regenerated(self):
        answer, calls = run([{'choices': []}, reply('Да.')])
        self.assertEqual((answer, len(calls)), ('Да.', 2))

    def test_long_answer_replaced_only_by_shorter_complete_one(self):
        long = ' '.join(['слово'] * (95 + 5))
        answer, calls = run([reply(long), reply('Короткий ответ.')])
        self.assertEqual((answer, len(calls)), ('Короткий ответ.', 2))
        answer, _ = run([reply(long), reply(long + ' ещё')])
        self.assertEqual(answer, long)
        answer, _ = run([reply(long), reply('Обрыв', 'length')])
        self.assertEqual(answer, long)

    def test_no_content_twice_is_an_error(self):
        with self.assertRaises(ValueError):
            run([reply(None), reply('')])

    def test_prompt_examples_are_not_demo_answers(self):
        import csv
        root = Path(__file__).resolve().parents[1]
        with (root / 'demo_output.csv').open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                self.assertNotIn(row['answer'][:60], solution.SYSTEM)


if __name__ == '__main__':
    unittest.main()
