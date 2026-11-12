"""Deterministic checks next to the question: arithmetic only, synthetic data only."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
import checks  # noqa: E402
import config  # noqa: E402
import solution  # noqa: E402


def found(text, inverted=True):
    return [(q['dimension'], q['value'], q['period']) for q in checks.quantities(text, inverted=inverted)]


class QuantityTests(unittest.TestCase):
    def test_digits_units_and_multipliers_are_brought_to_one_unit(self):
        self.assertEqual(found('Мощность 4 МВт, в месяц уходит около 1,8 млн кВт·ч'),
                         [('power', 4000.0, None), ('energy', 1800000.0, 'month')])
        self.assertEqual(found('0,6 МВт'), [('power', 600.0, None)])
        self.assertEqual(found('от 1 000 000 кВт·ч в месяц'), [('energy', 1000000.0, 'month')])
        self.assertEqual(found('2,5 тыс. кВт*ч в сутки'), [('energy', 2500.0, 'day')])
        self.assertEqual(found('В год выходит 9 млн кВтч'), [('energy', 9000000.0, 'year')])

    def test_numbers_written_in_words_and_colloquial_order(self):
        self.assertEqual(found('панелей киловатт на восемь'), [('power', 8.0, None)])
        self.assertEqual(found('полтора мегаватта'), [('power', 1500.0, None)])
        self.assertEqual(found('сто двадцать пять киловатт'), [('power', 125.0, None)])
        self.assertEqual(found('тысяч двенадцать киловатт-часов в месяц'), [('energy', 12000.0, 'month')])
        self.assertEqual(found('киловатт-часов тысяч двенадцать'), [('energy', 12000.0, None)])
        self.assertEqual(found('миллион киловатт-часов'), [('energy', 1000000.0, None)])

    def test_ranges_and_lists_share_the_unit(self):
        self.assertEqual(found('от 15 до 500 кВт', inverted=False), [('power', 15.0, None), ('power', 500.0, None)])
        self.assertEqual(found('15--500 кВт', inverted=False), [('power', 15.0, None), ('power', 500.0, None)])
        self.assertEqual(found('Пороги 15, 60 и 150 кВт', inverted=False),
                         [('power', 15.0, None), ('power', 60.0, None), ('power', 150.0, None)])
        self.assertEqual(found('от 750 кВт, но менее 20 МВт', inverted=False), [('power', 750.0, None), ('power', 20000.0, None)])

    def test_numbers_without_a_unit_and_other_units_are_ignored(self):
        self.assertEqual(found('на одной 400 кВт, на другой 450'), [('power', 400.0, None)])
        self.assertEqual(found('распоряжение от 26.09.2026 до 150 кВт', inverted=False), [('power', 150.0, None)])
        self.assertEqual(found('70 руб за ватт, 10 кВ и 0,4 кВ, 12 месяцев, у нас три цеха'), [])
        self.assertEqual(found('670 кВт часто хватает'), [('power', 670.0, None)])
        # The colloquial order is read in client turns only.
        self.assertEqual(found('мощность в киловаттах на 3 площадках', inverted=False), [])


class NumberCheckTests(unittest.TestCase):
    def tearDown(self):
        config.reload()

    def lines(self, question, *texts):
        with config.override(NUMBER_CHECK=1):
            return checks.number_check(question, [{'text': t} for t in texts])

    def test_value_below_the_range_is_stated_as_smaller(self):
        line, = self.lines('Хотим панелей киловатт на восемь.', 'Типовой проект — от 15 до 500 кВт.')
        self.assertIn('8 кВт из вопроса', line)
        self.assertIn('МЕНЬШЕ, чем 15 кВт, 500 кВт', line)
        self.assertNotIn('БОЛЬШЕ', line)

    def test_megawatts_and_millions_are_compared_in_one_unit(self):
        power, energy = self.lines('Пользователь: Что даст рынок?\nАссистент: Назовите мощность.\n'
                                   'Пользователь: Мощность 4 МВт, в месяц около 1,8 млн кВт·ч.',
                                   'От 750 кВт, но менее 20 МВт — через посредника. Ориентир — от 1 000 000 кВт·ч в месяц.')
        self.assertIn('4 МВт (= 4000 кВт)', power)
        self.assertIn('БОЛЬШЕ, чем 750 кВт; МЕНЬШЕ, чем 20 МВт', power)
        self.assertIn('1,8 млн кВт·ч в месяц из вопроса — это БОЛЬШЕ, чем 1 млн кВт·ч в месяц', energy)

    def test_yearly_consumption_is_converted_to_the_monthly_reference(self):
        line, = self.lines('В год у нас выходит 9 млн кВт·ч.', 'Ориентир — от 1 000 000 кВт·ч в месяц.')
        self.assertIn('около 750 тыс. кВт·ч в месяц', line)
        self.assertIn('МЕНЬШЕ, чем 1 млн кВт·ч в месяц', line)

    def test_equal_value_assistant_numbers_and_quoted_wordings(self):
        line, = self.lines('Пользователь: У нас ровно 750 кВт.\nАссистент: Лимит 15 кВт.\nПользователь: Подходим?',
                           'Ниже 750 кВт выход невозможен. Нельзя говорить «от 5 кВт».')
        self.assertIn('РАВНО 750 кВт', line)
        self.assertNotIn('5 кВт', line.replace('750 кВт', ''))  # quoted wording skipped
        self.assertNotIn('15 кВт', line)                        # assistant turns are not the client's numbers

    def test_latest_client_turn_wins_per_dimension(self):
        lines = self.lines('Пользователь: У нас 500 кВт и 300 тыс. кВт·ч в месяц.\nАссистент: Понятно.\n'
                           'Пользователь: Поднял акт — там 720 кВт.', 'Порог 670 кВт. Ориентир 1 000 000 кВт·ч в месяц.')
        self.assertEqual(len(lines), 2)
        self.assertIn('720 кВт из вопроса — это БОЛЬШЕ, чем 670 кВт', lines[0])
        self.assertIn('300 тыс. кВт·ч в месяц из вопроса', lines[1])
        self.assertNotIn('500 кВт', ' '.join(lines))

    def test_last_scope_reads_only_the_last_client_turn(self):
        early = ('Пользователь: У нас 500 кВт и 300 тыс. кВт·ч в месяц.\nАссистент: Лимит 15 кВт.\n'
                 'Пользователь: А сколько длится подключение?')
        documents = ('Порог 670 кВт. Ориентир 1 000 000 кВт·ч в месяц.',)
        self.assertEqual(len(self.lines(early, *documents)), 2)  # default scope keeps the earlier quantities
        with config.override(NUMBER_CHECK=1, NUMBER_CHECK_SCOPE='last'):
            sources = [{'text': t} for t in documents]
            self.assertEqual(checks.number_check(early, sources), [])
            self.assertEqual(checks.block(early, sources), '')
            # A quantity in the last turn is compared; the earlier one of another dimension is not carried over.
            line, = checks.number_check(early + ' У нас, напомню, 720 кВт.', sources)
            self.assertIn('720 кВт из вопроса — это БОЛЬШЕ, чем 670 кВт', line)
            # A plain question is its own last turn; a trailing assistant turn does not hide the client's.
            self.assertEqual(len(checks.number_check('У нас 8 кВт. Подойдёт?', [{'text': 'От 15 до 500 кВт.'}])), 1)
            self.assertEqual(len(checks.number_check('Пользователь: У нас 8 кВт.\nАссистент: Уточню.', [{'text': 'От 15 до 500 кВт.'}])), 1)

    def test_no_line_without_client_quantity_or_comparable_document_value(self):
        self.assertEqual(self.lines('Сколько стоит платформа учёта?', 'От 15 до 500 кВт.'), [])
        self.assertEqual(self.lines('У нас 300 кВт.', 'Платформа ведёт учёт в кВт·ч.'), [])


class RequestTests(unittest.TestCase):
    def tearDown(self):
        config.reload()

    def test_defaults_add_nothing_to_the_request(self):
        config.reload()
        self.assertEqual(checks.block('У нас 8 кВт.', [{'text': 'От 15 до 500 кВт.'}]), '')
        self.assertEqual(solution.user_message('Вопрос?', 'ДОК'), 'ДОКУМЕНТЫ:\nДОК\n\nДИАЛОГ ИЛИ ВОПРОС:\nВопрос?')
        self.assertEqual(solution.system_prompt(), solution.SYSTEM_FULL)

    def test_checks_follow_the_dialogue_and_assistant_reminder_needs_an_assistant_turn(self):
        dialogue = 'Пользователь: У нас 3 МВт.\nАссистент: Прямой выход возможен.\nПользователь: Значит, сбыт не нужен?'
        with config.override(NUMBER_CHECK=1, ASSISTANT_CHECK=1, PROMPT='v4'):
            notes = checks.block(dialogue, [{'text': 'От 750 кВт, но менее 20 МВт — через посредника.'}])
            message = solution.user_message(dialogue, 'ДОК', notes)
            self.assertIn('СВЕРКА ЧИСЕЛ', notes)
            self.assertIn('ПРОВЕРКА ДИАЛОГА', notes)
            self.assertTrue(message.endswith(notes))
            self.assertNotIn('клиент', notes.lower())  # that word made the model speak about the client in the third person
            self.assertLess(message.index(dialogue), message.index('СВЕРКА ЧИСЕЛ'))
            self.assertNotIn('ПРОВЕРКА ДИАЛОГА', checks.block('У нас 3 МВт. Можно на опт?', [{'text': 'От 750 кВт.'}]))
            self.assertTrue(solution.system_prompt().startswith(solution.SYSTEM_FULL))
            self.assertIn('ПЕРЕД ОТВЕТОМ ПРОВЕРЬ', solution.system_prompt())

    def test_no_product_fact_lives_in_the_checks(self):
        source = (ROOT / 'submission/checks.py').read_text(encoding='utf-8')
        code = source.split('"""', 2)[2]  # everything after the module docstring
        # Numerals, units and multipliers only: no threshold of the knowledge base is written in code.
        for figure in ('750', '670', '2675', 'руб', 'оптов', 'солнеч'):
            self.assertFalse(figure in code, figure)


if __name__ == '__main__':
    unittest.main()
