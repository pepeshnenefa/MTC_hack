"""Numeric validation engine: parsing, validation of sources, aggregation rules and verdicts.

Synthetic data only. The extraction call is replaced by a fixed JSON object: these tests cover
everything the code decides. What the model extracts on real questions is measured separately
(tools/numeric_regression.py, tests/evaluation/numeric-regression-v1).
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
import config  # noqa: E402
import numeric_engine as ne  # noqa: E402
import solution  # noqa: E402

DOCS = [
    {'source': 'a.md', 'heading': 'Условия', 'title': 'Склад / Условия', 'text':
        'Пост ставим на участках от 40 кВт, но менее 2 МВт.\n\n'
        'Мощности участков складываются, если они запитаны от одного щита.\n\n'
        'Типовой комплект — от 15 до 500 кВт.\n\n'
        'Ориентир загрузки — от 1 000 000 кВт·ч в месяц.'},
    {'source': 'b.md', 'heading': 'Сроки', 'title': 'Склад / Сроки', 'text':
        'Заявку подают не позднее чем за 10 рабочих\nдней до начала периода.\n\n'
        'Рассрочка на срок до 5 лет.\n\nАрхив — не менее года. Формулировка «от 5 кВт» запрещена.'},
]


def found(text, client=True):
    return [(q['dimension'], q['value'], q['period']) for q in ne.quantities(text, client=client)]


def run(question, extraction, docs=DOCS, scope='last'):
    """Engine result for a canned extraction; block numbers follow ne.blocks()."""
    return ne.run(question, docs, lambda messages: json.dumps(extraction, ensure_ascii=False), scope=scope)


def value(id_, quote, turn=1, **extra):
    return dict({'id': id_, 'turn': turn, 'quote': quote, 'unit': '', 'period': '', 'object': 'объект', 'own': True, 'replaced': False}, **extra)


def block_no(docs, needle, families):
    return next(b['no'] for b in ne.blocks(docs, families) if needle in b['text'])


class ParsingTests(unittest.TestCase):
    def test_decimals_millions_units_and_word_numerals(self):
        self.assertEqual(found('по акту 0,7 МВт, за год около четырёх миллионов киловатт-часов'),
                         [('power', 700.0, None), ('energy', 4000000.0, 'year')])
        self.assertEqual(found('18 500 кВт'), [('power', 18500.0, None)])
        self.assertEqual(found('в месяц уходит около 1,8 млн кВт·ч'), [('energy', 1800000.0, 'month')])
        self.assertEqual(found('панелей киловатт на восемь'), [('power', 8.0, None)])
        self.assertEqual(found('полтора мегаватта'), [('power', 1500.0, None)])
        self.assertEqual(found('670 кВт часто хватает'), [('power', 670.0, None)])

    def test_durations_and_dates(self):
        self.assertEqual(found('через четыре рабочих дня'), [('workdays', 4.0, None)])
        self.assertEqual(found('в течение 42 месяцев'), [('months', 42.0, None)])
        self.assertEqual(found('растянуть лет на семь'), [('months', 84.0, None)])
        self.assertEqual(found('не позднее чем за 10 рабочих\n   дней', client=False), [('workdays', 10.0, None)])
        self.assertEqual(found('архив не менее года, рассрочка до 5 лет, около полугода', client=False),
                         [('months', 12.0, None), ('months', 60.0, None), ('months', 6.0, None)])
        # A calendar year is a date, not a duration; a period word after a unit is not a quantity.
        self.assertEqual(found('с 1 сентября 2026 года лимит до 150 кВт, 25 000 ₽ в год', client=False), [('power', 150.0, None)])
        self.assertEqual(found('с десяти вечера до шести утра, 10 кВ, 70 руб за ватт'), [])

    def test_a_half_is_read_and_a_multiplier_without_its_numeral_is_not(self):
        self.assertEqual(found('мощность двенадцать с половиной тысяч киловатт'), [('power', 12500.0, None)])
        self.assertEqual(found('один цех на полмегаватта, второй на пол-киловатта'), [('power', 500.0, None), ('power', 0.5, None)])
        self.assertEqual(found('около 3 с половиной мегаватт'), [('power', 3500.0, None)])
        # "тысяч киловатт" has lost its numeral: reading it as 1000 kW turned 12.5 MW into 1 MW.
        self.assertEqual(found('тысяч киловатт'), [])
        self.assertEqual(found('млн кВт·ч в год'), [])
        self.assertEqual(found('около тысячи киловатт'), [('power', 1000.0, None)])
        self.assertEqual(found('миллион киловатт-часов в год'), [('energy', 1000000.0, 'year')])

    def test_client_lists_keep_every_object_in_any_order(self):
        self.assertEqual(found('по документам 300, 280 и 240 киловатт'), [('power', 300.0, None), ('power', 280.0, None), ('power', 240.0, None)])
        self.assertEqual(found('на 8 и на 7 мегаватт'), [('power', 8000.0, None), ('power', 7000.0, None)])
        self.assertEqual(found('в 2 и на 7 мегаватт'), [('power', 7000.0, None)])
        # Documents stay strict: a list or range must grow towards the unit.
        self.assertEqual(found('в 2025 и 150 кВт', client=False), [('power', 150.0, None)])
        self.assertEqual(found('от 15 до 500 кВт', client=False), [('power', 15.0, None), ('power', 500.0, None)])

    def test_gate_reads_the_last_client_turn_only_by_default(self):
        early = 'Пользователь: У нас 500 кВт.\nАссистент: Понятно.\nПользователь: А сколько длится подключение?'
        self.assertEqual(ne.gate(early), set())
        self.assertEqual(ne.gate(early, 'dialog'), {'power'})
        self.assertEqual(ne.gate('Растянуть на 42 месяца можно?'), {'duration'})
        self.assertEqual(ne.gate('У нас два ската и 32 дома'), set())


class BlockTests(unittest.TestCase):
    def test_blocks_hold_bounds_of_the_clients_kind_and_the_sum_rule(self):
        blocks = ne.blocks(DOCS, {'power'})
        texts = [b['text'] for b in blocks]
        self.assertEqual(len(blocks), 3)
        self.assertEqual(ne.blocks([{'source': 'c.md', 'title': 'T', 'text': 'Пример: станция 100 кВт окупилась за 6,1 года.'}], {'power', 'duration'}), [])
        listed, = ne.blocks([{'source': 'c.md', 'title': 'T', 'text': '- Работаем по будням.\n- Пост ставим на участках\n  от 40 кВт.\n- Гарантия есть.'}], {'power'})
        self.assertEqual(listed['text'], '- Пост ставим на участках\n  от 40 кВт.')
        self.assertTrue(any('складываются' in t for t in texts))
        self.assertFalse(any('рабочих' in t or 'кВт·ч' in t for t in texts))
        self.assertEqual([b['no'] for b in blocks], [1, 2, 3])
        # A figure inside «…» is a quoted wording and never becomes a bound.
        self.assertEqual(ne.blocks([{'source': 'c.md', 'title': 'T', 'text': 'Формулировка «от 5 кВт» запрещена.'}], {'power'}), [])


class VerdictTests(unittest.TestCase):
    def tearDown(self):
        config.reload()

    def test_fail_below_a_minimum_with_unit_conversion_and_sources(self):
        question = 'У нас участок на 0,03 МВт. Пост поставите?'
        result = run(question, {'check': True, 'values': [value('v1', 'участок на 0,03 МВт')],
                                'limits': [{'block': 1, 'quote': 'от 40 кВт', 'kind': 'min', 'values': ['v1']}]})
        self.assertEqual(result['status'], ne.FAIL)
        check, = result['checks']
        self.assertEqual((check['client'], check['relation'], check['limit']['low']), (30.0, 'below', 40.0))
        self.assertIn('30 кВт из вопроса — это МЕНЬШЕ, чем 40 кВт: условие «не меньше 40 кВт» НЕ выполнено', result['note'])
        # Sources of both numbers are kept: turn and offsets for the client, file, section and offsets for the bound.
        stored, = result['values']
        self.assertEqual((stored['turn'], question[stored['span'][0]:stored['span'][1]]), (1, '0,03 МВт'))
        self.assertEqual((check['limit']['source'], check['limit']['title'], check['limit']['text']), ('a.md', 'Склад / Условия', '40 кВт'))
        self.assertEqual(result['calls'], 1)

    def test_pass_and_boundary_values(self):
        limit = [{'block': 1, 'quote': 'от 40 кВт', 'kind': 'min', 'values': ['v1']}]
        self.assertEqual(run('У нас 55 кВт. Подходим?', {'check': True, 'values': [value('v1', '55 кВт')], 'limits': limit})['status'], ne.PASS)
        # A value exactly on the bound is usually the bound retold: no verdict, the request stays as it was.
        equal = run('У нас ровно 40 кВт. Подходим?', {'check': True, 'values': [value('v1', '40 кВт')], 'limits': limit})
        self.assertEqual((equal['status'], equal['note']), (ne.NO_CHECK, ''))
        # "менее 2 МВт" as a maximum: the bound itself is already outside.
        upper = [{'block': 1, 'quote': '2 МВт', 'kind': 'max', 'values': ['v1']}]
        self.assertEqual(run('У нас 2 МВт. Подходим?', {'check': True, 'values': [value('v1', '2 МВт')], 'limits': upper})['status'], ne.FAIL)
        self.assertEqual(run('У нас 1,9 МВт. Подходим?', {'check': True, 'values': [value('v1', '1,9 МВт')], 'limits': upper})['status'], ne.PASS)

    def test_the_first_bound_decides_and_neighbouring_steps_are_listed(self):
        steps = [{'block': 1, 'quote': 'от 40 кВт', 'kind': 'min'}, {'block': 1, 'quote': '2 МВт', 'kind': 'min'}]
        result = run('У нас 55 кВт. Подходим?', {'intent': 'threshold', 'values': [value('v1', '55 кВт')], 'limits': steps})
        self.assertEqual((result['status'], [c['verdict'] for c in result['checks']]), (ne.PASS, [ne.PASS, ne.FAIL]))
        self.assertEqual(result['note'].count('\n- '), 2)

    def test_next_step_of_the_same_section_is_compared_when_the_minimum_is_reached(self):
        steps = [{'source': 's.md', 'heading': 'Ступени', 'title': 'Склад / Ступени', 'text':
                  '- Ниже 40 кВт пост не ставим.\n- От 40 кВт ставим через подрядчика.\n- От 900 кВт ставим сами.'},
                 {'source': 's.md', 'heading': 'Другое', 'title': 'Склад / Другое', 'text': 'Скидка действует от 100 кВт.'}]
        first = [{'block': 1, 'quote': 'Ниже 40 кВт', 'kind': 'min'}]
        reached = run('У нас 0,5 МВт. Пост поставите?', {'intent': 'threshold', 'values': [value('v1', '0,5 МВт')], 'limits': first}, docs=steps)
        self.assertEqual((reached['status'], [(c['limit']['low'], c['verdict'], c['limit']['next_step']) for c in reached['checks']]),
                         (ne.PASS, [(40.0, ne.PASS, False), (900.0, ne.FAIL, True)]))
        self.assertIn('500 кВт из вопроса — это МЕНЬШЕ, чем 900 кВт: условие «не меньше 900 кВт» НЕ выполнено', reached['note'])
        # Not reached: the next step says nothing new. Another section's bound is never a step.
        below = run('У нас 30 кВт. Пост поставите?', {'intent': 'threshold', 'values': [value('v1', '30 кВт')], 'limits': first}, docs=steps)
        self.assertEqual((below['status'], len(below['checks'])), (ne.FAIL, 1))
        self.assertNotIn('100 кВт', reached['note'])

    def test_range_and_the_word_before_the_number_overrules_a_wrong_kind(self):
        no = block_no(DOCS, 'Типовой', {'power'})
        above = run('Хотим комплект на 0,65 мегаватта. Типовой?', {'check': True, 'values': [value('v1', '0,65 мегаватта')],
                                                                  'limits': [{'block': no, 'quote': 'от 15 до 500 кВт', 'kind': 'range'}]})
        self.assertEqual((above['status'], above['checks'][0]['relation']), (ne.FAIL, 'above'))
        self.assertIn('ВЫШЕ диапазона (больше, чем 500 кВт): условие «от 15 кВт до 500 кВт» НЕ выполнено', above['note'])
        # The extractor called "до 5 лет" a minimum; the document wording decides.
        late = run('Рассрочку лет на семь дадите?', {'check': True, 'values': [value('v1', 'лет на семь')],
                                                    'limits': [{'block': block_no(DOCS, 'Рассрочка', {'duration'}), 'quote': 'до 5 лет', 'kind': 'min'}]})
        self.assertEqual((late['status'], late['checks'][0]['limit']['kind'], late['checks'][0]['relation']), (ne.FAIL, 'max', 'above'))
        self.assertTrue(any(r.get('corrected_to') == 'max' for r in late['rejected']))
        fits = run('Платить 42 месяца можно?', {'check': True, 'values': [value('v1', '42 месяца')],
                                               'limits': [{'block': block_no(DOCS, 'Рассрочка', {'duration'}), 'quote': 'до 5 лет', 'kind': 'max'}]})
        self.assertEqual(fits['status'], ne.PASS)
        self.assertIn('3,5 года (= 42 мес.) из вопроса — это МЕНЬШЕ, чем 5 лет (= 60 мес.): условие «не больше 5 лет (= 60 мес.)» выполнено', fits['note'])

    def test_yearly_consumption_is_compared_with_a_monthly_bound(self):
        no = block_no(DOCS, 'Ориентир', {'energy'})
        limit = [{'block': no, 'quote': 'от 1 000 000 кВт·ч в месяц', 'kind': 'min'}]
        low = run('За год сжигаем около девяти миллионов киловатт-часов. Дотягиваем?',
                  {'check': True, 'values': [value('v1', 'девяти миллионов киловатт-часов')], 'limits': limit})
        self.assertEqual((low['status'], low['checks'][0]['client']), (ne.FAIL, 750000.0))
        self.assertIn('9 млн кВт·ч в год (около 750 тыс. кВт·ч в месяц) из вопроса — это МЕНЬШЕ, чем 1 млн кВт·ч в месяц', low['note'])
        high = run('В год выходит 14,4 млн кВт·ч. Дотягиваем?', {'check': True, 'values': [value('v1', '14,4 млн кВт·ч')], 'limits': limit})
        self.assertEqual((high['status'], high['checks'][0]['client']), (ne.PASS, 1200000.0))

    def test_missing_period_and_inexact_units_need_clarification(self):
        no = block_no(DOCS, 'Ориентир', {'energy'})
        limit = [{'block': no, 'quote': '1 000 000 кВт·ч', 'kind': 'min'}]
        unknown = run('Потребляем 5 млн кВт·ч. Дотягиваем?', {'check': True, 'values': [value('v1', '5 млн кВт·ч')], 'limits': limit})
        self.assertEqual((unknown['status'], unknown['checks'][0]['reason']), (ne.CLARIFY, 'period'))
        self.assertIn('не назван период', unknown['note'])
        # A period the extractor names is accepted only when the consultant's previous turn asked about it.
        dialog = 'Пользователь: Думаем про склад.\nАссистент: Сколько потребляете в месяц?\nПользователь: Тысяч шестьсот киловатт-часов. Дотягиваем?'
        told = run(dialog, {'check': True, 'values': [value('v1', 'шестьсот киловатт-часов', turn=3, period='месяц')], 'limits': limit})
        self.assertEqual((told['status'], told['values'][0]['period'], told['values'][0]['period_inferred']), (ne.FAIL, 'month', True))
        invented = run('Потребляем 5 млн кВт·ч. Дотягиваем?', {'check': True, 'values': [value('v1', '5 млн кВт·ч', period='месяц')], 'limits': limit})
        self.assertEqual(invented['status'], ne.CLARIFY)
        days = run('Подадим заявку за 12 дней. Успеем?', {'check': True, 'values': [value('v1', '12 дней')],
                                                        'limits': [{'block': block_no(DOCS, 'рабочих', {'duration'}), 'quote': '10 рабочих дней', 'kind': 'min'}]})
        self.assertEqual((days['status'], days['checks'][0]['reason']), (ne.CLARIFY, 'units'))
        # "не позднее чем за N" is a minimum lead time even when the extractor calls it a maximum.
        work = run('До начала периода четыре рабочих дня. Успеем?', {'check': True, 'values': [value('v1', 'четыре рабочих дня')],
                                                                   'limits': [{'block': block_no(DOCS, 'рабочих', {'duration'}), 'quote': '10 рабочих дней', 'kind': 'max'}]})
        self.assertEqual((work['status'], work['checks'][0]['relation'], work['checks'][0]['limit']['kind']), (ne.FAIL, 'below', 'min'))


class AggregationTests(unittest.TestCase):
    QUESTION = 'У нас два участка: на первом 25 кВт, на втором 0,03 МВт, оба от одного щита. Пост поставите?'
    VALUES = [value('v1', 'на первом 25 кВт', object='участок 1'), value('v2', 'на втором 0,03 МВт', object='участок 2')]
    LIMIT = [{'block': 1, 'quote': 'от 40 кВт', 'kind': 'min', 'values': ['v1', 'v2']}]

    def test_sum_needs_a_document_rule_and_the_clients_words(self):
        result = run(self.QUESTION, {'check': True, 'values': self.VALUES, 'limits': self.LIMIT,
                                     'sum': {'mode': 'yes', 'rule_block': 2, 'client_words': 'оба от одного щита'}})
        check, = result['checks']
        self.assertEqual((result['status'], check['mode'], check['client']), (ne.PASS, 'sum', 55.0))
        self.assertIn('25 кВт + 30 кВт = 55 кВт', result['note'])
        self.assertIn('оба от одного щита', result['note'])
        # The evidence is looked up by code: the client's clause must share words with the rule, whatever
        # fragment the extractor cites. Without such a clause, with a negated one, or without a "yes"
        # from the extractor, the values are not added.
        silent = 'У нас два участка: на первом 25 кВт, на втором 0,03 МВт. Пост поставите?'
        denied = 'У нас два участка: на первом 25 кВт, на втором 0,03 МВт, от одного щита они не запитаны. Пост поставите?'
        for question, basis in ((silent, {'mode': 'yes', 'rule_block': 2, 'client_words': 'У нас два участка'}),
                                (denied, {'mode': 'yes', 'rule_block': 2, 'client_words': 'от одного щита'}),
                                (self.QUESTION, {})):
            doubt = run(question, {'check': True, 'values': self.VALUES, 'limits': self.LIMIT, 'sum': basis})
            self.assertEqual((doubt['status'], doubt['checks'][0]['reason']), (ne.CLARIFY, 'aggregation'))
            self.assertIn('спроси про это условие', doubt['note'])
        # No block states a rule of adding: nothing to add by.
        no_rule = [dict(DOCS[0], text='Пост ставим на участках от 40 кВт, но менее 2 МВт.')]
        self.assertEqual(run(self.QUESTION, {'check': True, 'values': self.VALUES, 'limits': self.LIMIT, 'sum': {'mode': 'yes'}}, docs=no_rule)['status'], ne.CLARIFY)

    def test_objects_compared_one_by_one_when_the_client_denies_the_condition(self):
        question = 'Два участка в разных городах, никак не соединены: 0,9 МВт и 30 кВт. Каждый пройдёт?'
        values = [value('v1', '0,9 МВт', object='участок 1'), value('v2', '30 кВт', object='участок 2')]
        result = run(question, {'check': True, 'values': values, 'limits': self.LIMIT, 'sum': {'mode': 'no', 'rule_block': 2, 'client_words': 'никак не соединены'}},
                     docs=[dict(DOCS[0], text=DOCS[0]['text'].replace('запитаны от одного щита', 'соединены общей линией'))])
        check, = result['checks']
        self.assertEqual((result['status'], check['mode'], check['relation']), (ne.FAIL, 'each', 'mixed'))
        self.assertIn('900 кВт — БОЛЬШЕ, чем 40 кВт: условие «не меньше 40 кВт» выполнено; 30 кВт — МЕНЬШЕ, чем 40 кВт: условие «не меньше 40 кВт» НЕ выполнено', result['note'])

    def test_same_outcome_either_way_needs_no_basis_and_one_fragment_can_list_objects(self):
        question = 'Три корпуса: 300, 280 и 240 киловатт. Пост поставите?'
        listed = run(question, {'check': True, 'values': [value('v1', '300, 280 и 240 киловатт', object='корпуса')], 'limits': self.LIMIT[:1]})
        check, = listed['checks']
        self.assertEqual((listed['status'], check['mode'], len(check['values']), check['client']), (ne.PASS, 'any', 3, 820.0))
        self.assertEqual({v['object'] for v in listed['values']}, {'корпуса #1', 'корпуса #2', 'корпуса #3'})

    def test_bare_number_takes_the_unit_named_by_the_extractor_and_a_client_total_is_dropped(self):
        question = 'На одной площадке 20 кВт, на другой 25, вместе 45. От одного щита. Пост поставите?'
        values = [value('v1', '20 кВт', object='площадка 1'), value('v2', 'на другой 25', unit='кВт', object='площадка 2'),
                  value('v3', 'вместе 45', unit='кВт', object='итог')]
        result = run(question, {'check': True, 'values': values, 'limits': [{'block': 1, 'quote': 'от 40 кВт', 'kind': 'min'}],
                                'sum': {'mode': 'yes', 'block': 2, 'quote': 'От одного щита'}})
        check, = result['checks']
        self.assertEqual((result['status'], check['client'], len(check['values'])), (ne.PASS, 45.0, 2))
        self.assertTrue(next(v for v in result['values'] if v['id'] == 'v2')['unit_inferred'])

    VOLTAGE_DOCS = [dict(DOCS[0], text=DOCS[0]['text'].replace(
        'Мощности участков складываются, если они запитаны от одного щита.',
        'Мощности участков складываются только при электрической связности между ними хотя бы по 0,4 кВ.'))]

    def test_a_link_at_the_rules_voltage_level_is_the_condition_and_a_shared_stem_is_not(self):
        values = [value('v1', '25 и 20 кВт', object='участки')]
        linked = run('Два участка, 25 и 20 кВт, между ними наша кабельная перемычка на 0,4 кВ. Пост поставите?',
                     {'intent': 'threshold', 'values': values, 'limits': self.LIMIT,
                      'sum': {'mode': 'yes', 'rule_block': 2, 'client_words': 'между ними наша кабельная перемычка на 0,4 кВ'}},
                     docs=self.VOLTAGE_DOCS)
        self.assertEqual((linked['status'], linked['checks'][0]['mode'], linked['checks'][0]['client']), (ne.PASS, 'sum', 45.0))
        # Both sites on 0,4 кВ says nothing about a link between them; "электричество" is not "электрическая связность".
        for question in ('Два участка, 25 и 20 кВт, оба на 0,4 кВ. Пост поставите?',
                         'Два участка, 25 и 20 кВт, оба покупают электричество у одного поставщика. Пост поставите?'):
            doubt = run(question, {'intent': 'threshold', 'values': values, 'limits': self.LIMIT,
                                   'sum': {'mode': 'yes', 'rule_block': 2, 'client_words': 'оба'}}, docs=self.VOLTAGE_DOCS)
            self.assertEqual((doubt['status'], doubt['checks'][0]['reason']), (ne.CLARIFY, 'aggregation'), question)


class CorrectionTests(unittest.TestCase):
    LIMIT = [{'block': 1, 'quote': 'от 40 кВт', 'kind': 'min'}]

    def test_a_correction_replaces_the_earlier_value(self):
        dialog = 'Пользователь: У нас участок 60 кВт.\nАссистент: Уточните по документам.\nПользователь: Исправляю: 35 кВт. Проходим?'
        values = [value('v1', '60 кВт', turn=1), value('v2', '35 кВт', turn=3)]
        result = run(dialog, {'check': True, 'values': values, 'limits': self.LIMIT})
        self.assertEqual((result['status'], result['checks'][0]['client'], result['checks'][0]['values']), (ne.FAIL, 35.0, ['v2']))
        self.assertNotIn('60 кВт', result['note'])
        # Inside one turn the extractor's flag decides which of the two numbers stands.
        one = 'Сначала думал, что 60 кВт, но это ошибка, по паспорту 35 кВт. Проходим?'
        flagged = run(one, {'check': True, 'values': [value('v1', '60 кВт', replaced=True), value('v2', '35 кВт')], 'limits': self.LIMIT})
        self.assertEqual((flagged['status'], flagged['checks'][0]['client']), (ne.FAIL, 35.0))

    def test_the_latest_turn_wins_when_it_reads_as_a_correction_whatever_the_flags_say(self):
        dialog = ('Пользователь: Думаем о комплекте, киловатт на шестьсот.\nАссистент: Какая площадь?\n'
                  'Пользователь: Посчитали заново, ошибся: не шестьсот, а четыреста пятьдесят киловатт. Это типовой?')
        # The extractor flagged the wrong number as withdrawn.
        values = [value('v1', 'не шестьсот, а четыреста пятьдесят киловатт', turn=3, replaced=True), value('v2', 'киловатт на шестьсот', turn=1)]
        result = run(dialog, {'intent': 'threshold', 'values': values, 'limits': [{'block': 3, 'quote': 'от 15 до 500 кВт', 'kind': 'range'}]})
        self.assertEqual((result['status'], result['checks'][0]['client']), (ne.PASS, 450.0))

    def test_corrections_without_the_usual_words_and_with_another_object_name(self):
        # "старую цифру назвал … по свежему акту": the extractor named the objects differently, it is one object.
        dialog = ('Пользователь: У нас участок, мощность 60 кВт.\nАссистент: Тогда пост ставим.\n'
                  'Пользователь: Подождите, я старую цифру назвал. По свежему акту у нас 0,035 МВт. Тогда что?')
        values = [value('v1', 'мощность 60 кВт', turn=1, object='участок'), value('v2', 'по свежему акту у нас 0,035 МВт', turn=3, object='по свежему акту')]
        result = run(dialog, {'intent': 'threshold', 'values': values, 'limits': self.LIMIT})
        self.assertEqual((result['status'], result['checks'][0]['client'], result['checks'][0]['mode']), (ne.FAIL, 35.0, 'single'))
        # "пересчитал … , не шестьдесят" and a bare "не 60" both withdraw the earlier number.
        for last in ('Инженер пересчитал: реально выйдет тридцать пять киловатт, не шестьдесят. Проходим?',
                     'Реально будет 35 кВт, не 60. Проходим?'):
            dialog = 'Пользователь: У нас участок 60 кВт.\nАссистент: Пост ставим.\nПользователь: ' + last
            quote = 'тридцать пять киловатт' if 'тридцать' in last else '35 кВт'
            again = run(dialog, {'intent': 'threshold', 'values': [value('v1', '60 кВт', turn=1), value('v2', quote, turn=3)], 'limits': self.LIMIT})
            self.assertEqual((again['status'], again['checks'][0]['client']), (ne.FAIL, 35.0), last)
        # Two objects of one turn with a correcting word stay two objects.
        dialog = 'Пользователь: Участок 30 кВт.\nАссистент: Мало.\nПользователь: Уточню: участков два, второй 20 кВт, оба от одного щита. Теперь?'
        two = run(dialog, {'intent': 'threshold', 'values': [value('v1', '30 кВт', turn=1, object='участок 1'),
                                                               value('v2', 'второй 20 кВт', turn=3, object='участок 2')],
                           'limits': self.LIMIT, 'sum': {'mode': 'yes', 'rule_block': 2}})
        self.assertEqual((two['status'], two['checks'][0]['mode'], two['checks'][0]['client']), (ne.PASS, 'sum', 50.0))

    def test_a_number_without_unit_is_kept_and_doubt_makes_values_alternatives(self):
        question = ('Пользователь: Хотим пост.\nАссистент: Какая мощность?\n'
                    'Пользователь: Тут путаница: в расчёте проектировщика 30 кВт, а в письме подрядчика — 50. Какая цифра правильная, не знаю. Поставите?')
        values = [value('v1', 'в расчёте проектировщика 30 кВт', turn=3, object='расчёт проектировщика'),
                  value('v2', 'в письме подрядчика — 50', turn=3, object='письмо подрядчика')]
        result = run(question, {'intent': 'threshold', 'values': values, 'limits': self.LIMIT})
        self.assertEqual((result['status'], result['checks'][0]['reason']), (ne.CLARIFY, 'contradiction'))
        self.assertIn('(30 кВт и 50 кВт)', result['note'])
        self.assertTrue(next(v for v in result['values'] if v['id'] == 'v2')['unit_inferred'])

    def test_contradictory_values_need_clarification(self):
        one = 'По договору 60 кВт, а в акте стоит 35 кВт. Проходим?'
        result = run(one, {'check': True, 'values': [value('v1', '60 кВт'), value('v2', '35 кВт')], 'limits': self.LIMIT})
        self.assertEqual((result['status'], result['checks'][0]['reason']), (ne.CLARIFY, 'contradiction'))
        self.assertIn('названы разные значения одной величины (60 кВт и 35 кВт)', result['note'])
        # A later turn that does not read as a correction does not silently win.
        dialog = 'Пользователь: У нас участок 60 кВт.\nАссистент: Что ещё важно?\nПользователь: Так при наших 35 кВт пост поставите?'
        later = run(dialog, {'check': True, 'values': [value('v1', '60 кВт', turn=1), value('v2', '35 кВт', turn=3)], 'limits': self.LIMIT})
        self.assertEqual(later['status'], ne.CLARIFY)
        # The same number repeated is not a contradiction; someone else's figure is not the client's value.
        same = run('Пользователь: У нас 60 кВт.\nАссистент: Да?\nПользователь: При наших 60 кВт поставите? Соседи говорят, нужно 100 кВт.',
                   {'check': True, 'values': [value('v1', '60 кВт', turn=1), value('v2', 'наших 60 кВт', turn=3),
                                              value('v3', '100 кВт', turn=3, own=False, object='слова соседей')], 'limits': self.LIMIT})
        self.assertEqual((same['status'], same['checks'][0]['client']), (ne.PASS, 60.0))
        self.assertFalse(next(v for v in same['values'] if v['id'] == 'v3')['own'])


class ValidationTests(unittest.TestCase):
    def test_anything_unconfirmed_yields_no_check_and_no_note(self):
        question = 'У нас участок на 30 кВт. Пост поставите?'
        good_value = [value('v1', '30 кВт')]
        good_limit = [{'block': 1, 'quote': 'от 40 кВт', 'kind': 'min'}]
        cases = {
            'extractor declined': {'check': False, 'values': good_value, 'limits': good_limit},
            'invented client number': {'check': True, 'values': [value('v1', '300 кВт')], 'limits': good_limit},
            'invented bound': {'check': True, 'values': good_value, 'limits': [{'block': 1, 'quote': 'от 25 кВт', 'kind': 'min'}]},
            'bound from an unknown block': {'check': True, 'values': good_value, 'limits': [{'block': 9, 'quote': '40 кВт на участке', 'kind': 'min'}]},
            'number that the document does not present as a bound': {'check': True, 'values': good_value, 'limits': [{'block': 1, 'quote': 'станция 100 кВт', 'kind': 'min'}]},
            'another intent': {'intent': 'price', 'values': good_value, 'limits': good_limit},
            'a quantity of the last turn was skipped': {'check': True, 'values': [value('v1', '30 кВт')], 'limits': good_limit, 'question': 'У нас участок на 30 кВт, хотя по паспорту 55 кВт. Пост поставите?'},
            'bound of another kind': {'check': True, 'values': good_value, 'limits': [{'block': 1, 'quote': 'складываются', 'kind': 'min'}]},
            'no values': {'check': True, 'values': [], 'limits': good_limit},
            # The extractor calls it someone else's and the sentence is about someone else.
            'only a foreign value': {'check': True, 'values': [value('v1', '30 кВт', own=False)], 'limits': good_limit,
                                     'question': 'У соседей участок на 30 кВт. Нам пост поставите?'},
        }
        examples = DOCS + [{'source': 'c.md', 'heading': 'Примеры', 'title': 'Склад / Примеры', 'text': 'Пример: станция 100 кВт окупилась.'}]
        for name, extraction in cases.items():
            result = run(extraction.pop('question', question), extraction, docs=examples)
            self.assertEqual((name, result['status'], result['note']), (name, ne.NO_CHECK, ''))
        # A wrong block number is repaired when the fragment stands word for word in another block.
        moved = run(question, {'intent': 'threshold', 'values': good_value, 'limits': [{'block': 9, 'quote': 'от 40 кВт', 'kind': 'min'}]})
        self.assertEqual((moved['status'], moved['checks'][0]['limit']['block']), (ne.FAIL, 1))
        # Someone else's figure is never the client's value, even when the extractor says so.
        heard = run('Нам сказали, что у соседей участок на 30 кВт. Это много?', {'intent': 'threshold', 'values': good_value, 'limits': good_limit})
        self.assertEqual((heard['status'], heard['values'][0]['hearsay']), (ne.NO_CHECK, True))
        # "Not the client's" without reported speech next to it may be a second value of the client: no verdict.
        unclear = run('У нас участок на 30 кВт, а в акте стоит 55 кВт. Пост поставите?',
                      {'intent': 'threshold', 'values': [value('v1', '30 кВт', object='участок'), value('v2', '55 кВт', own=False, object='акт')],
                       'limits': good_limit})
        self.assertEqual((unclear['status'], unclear['reason']), (ne.NO_CHECK, 'ownership of a value is unclear'))
        self.assertFalse(run('По договору у нас 30 кВт. Пост поставите?', {'intent': 'threshold', 'values': good_value, 'limits': good_limit})['values'][0]['hearsay'])
        for raw in ('не JSON', '', '[1, 2]', '{"check": true', None):
            result = ne.run(question, DOCS, lambda messages: raw)
            self.assertEqual((result['status'], result['note']), (ne.NO_CHECK, ''))

    def test_json_is_read_through_reasoning_and_code_fences(self):
        self.assertEqual(ne.parse_json('<think>{"a": 1}</think>\n```json\n{"check": true, "x": "}{"}\n```'), {'check': True, 'x': '}{'})
        self.assertIsNone(ne.parse_json('{"check": '))

    def test_no_model_call_without_a_quantity_or_without_a_bound_of_that_kind(self):
        calls = []
        llm = lambda messages: calls.append(1) or '{}'  # noqa: E731
        self.assertEqual(ne.run('Сколько стоит пост?', DOCS, llm)['status'], ne.NO_CHECK)
        self.assertEqual(ne.run('У нас 30 кВт. Подойдёт?', [{'source': 'c.md', 'title': 'T', 'text': 'Работаем по будням.'}], llm)['status'], ne.NO_CHECK)
        self.assertEqual(calls, [])

    def test_extraction_prompt_holds_blocks_numbered_turns_and_no_product_fact(self):
        dialog = 'Пользователь: У нас 30 кВт.\nАссистент: Понятно.\nПользователь: А 35 кВт подойдёт?'
        system, user = ne.extraction_messages(dialog, ne.blocks(DOCS, {'power'}))
        self.assertIn('[1] Склад / Условия', user['content'])
        self.assertIn('(1) Клиент: У нас 30 кВт.', user['content'])
        self.assertIn('(2) Консультант: Понятно.', user['content'])
        self.assertIn('ничего не считаешь', system['content'])
        code = (ROOT / 'submission/numeric_engine.py').read_text(encoding='utf-8').split('"""', 2)[2]
        for figure in ('750', '670', '2675', 'оптов', 'солнеч', 'руб'):
            self.assertFalse(figure in code, figure)
        self.assertNotIn('клиент', ne.HEADER.lower())  # that word made the model answer in the third person


# A scale of steps split over two sections, as documents often state it.
LADDER = [
    {'source': 'p.md', 'heading': 'Не подходит', 'title': 'Пост / Не подходит, если', 'text': '- Мощность участка ниже 40 кВт.'},
    {'source': 'q.md', 'heading': 'Ступени', 'title': 'Пороги / Ступени', 'text':
        '- Ниже 40 кВт пост не ставим.\n'
        '- От 40 кВт, но менее 2 МВт пост ставим через подрядчика, напрямую поставить не можем.\n'
        '- От 2 МВт возможна прямая установка нашей бригадой.'},
]
FIRST_STEP = [{'block': 1, 'quote': 'ниже 40 кВт', 'kind': 'min', 'values': ['v1']}]


class StepAndSilenceTests(unittest.TestCase):
    """Rules added after holdout-v4: which step the client asks about, and figures the extractor left out."""

    def test_the_clients_words_choose_the_step_of_the_scale(self):
        # Only the lowest step is cited; the scale is completed from the other section and the client's
        # "напрямую, без подрядчика" points at the top step.
        asked = run('У нас участок 0,5 МВт. Хотим, чтобы поставили напрямую, без подрядчика. Так можно?',
                    {'intent': 'threshold', 'values': [value('v1', '0,5 МВт')], 'limits': FIRST_STEP}, docs=LADDER)
        self.assertEqual((asked['status'], [(c['limit']['low'], c['verdict'], c['limit']['asked']) for c in asked['checks']]),
                         (ne.FAIL, [(2000.0, ne.FAIL, True), (40.0, ne.PASS, False)]))
        self.assertTrue(asked['note'].split('\n')[1].startswith('- 500 кВт из вопроса — это МЕНЬШЕ, чем 2 МВт'))
        # Words that point at no step leave the extractor's order.
        plain = run('У нас участок 0,5 МВт. Подходим?', {'intent': 'threshold', 'values': [value('v1', '0,5 МВт')], 'limits': FIRST_STEP}, docs=LADDER)
        self.assertEqual((plain['status'], [c['limit']['low'] for c in plain['checks']]), (ne.PASS, [40.0]))

    def test_a_scale_joins_neighbouring_steps_only(self):
        blocks = ne.blocks([{'source': 's.md', 'title': 'Станция / Отдача', 'text': 'Отдавать в сеть можно до 150 кВт, раньше лимит был 15 кВт.'},
                            {'source': 's.md', 'title': 'Станция / Типовой', 'text': 'Типовой проект — от 15 до 500 кВт.'}], {'power'})
        cap = {'kind': 'max', 'dimension': 'power', 'low': 150.0, 'high': 150.0, 'period': None, 'values': ['v1']}
        self.assertEqual(ne.ladder([cap], blocks), [])  # "15 … 150" overlaps the cap, "15–500" does not touch it
        steps = ne.ladder([{'kind': 'min', 'dimension': 'power', 'low': 40.0, 'high': 40.0, 'period': None, 'values': ['v1']}],
                          ne.blocks(LADDER, {'power'}))
        self.assertEqual(sorted((l['kind'], l['low'], l['high']) for l in steps), [('min', 2000.0, 2000.0), ('range', 40.0, 2000.0)])

    def test_a_claim_or_duration_with_the_clients_figure_and_a_question_about_it_is_checked(self):
        no = block_no(DOCS, 'Рассрочка', {'duration'})
        limit = [{'block': no, 'quote': 'до 5 лет', 'kind': 'max'}]
        asked = run('Финдиректор спрашивает: можно ли растянуть рассрочку на 84 месяца?',
                    {'intent': 'duration', 'values': [value('v1', 'на 84 месяца')], 'limits': limit})
        self.assertEqual(asked['status'], ne.FAIL)
        # No question about the figure, or a bound that shares no word with the question: no verdict.
        for question in ('Рассрочку на 84 месяца оформите быстро?', 'Нам 84 месяца можно?'):
            self.assertEqual(run(question, {'intent': 'duration', 'values': [value('v1', '84 месяца')], 'limits': limit})['status'], ne.NO_CHECK, question)

    def test_a_relayed_figure_of_the_client_is_the_clients_and_a_relayed_bound_is_not(self):
        limit = [{'block': 1, 'quote': 'от 40 кВт', 'kind': 'min'}]
        relayed = run('Главный энергетик говорит, что мощность у нас 30 кВт. Пост поставите?',
                      {'intent': 'threshold', 'values': [value('v1', '30 кВт')], 'limits': limit})
        self.assertEqual((relayed['status'], relayed['values'][0]['hearsay']), (ne.FAIL, False))
        text = 'Энергетик говорит, что у нас пост ставят только от 25 кВт.'
        self.assertTrue(ne._reported(text, text.find('25')))
        # Two figures the extractor gave one object name: the client named two values of it.
        two = run('Энергетик говорит, что мощность у нас 30 кВт, а директор уверен, что 0,05 МВт. Пост поставите?',
                  {'intent': 'threshold', 'values': [value('v1', '30 кВт', object='мощность'), value('v2', '0,05 МВт', own=False, object='мощность')],
                   'limits': limit})
        self.assertEqual((two['status'], two['checks'][0]['reason']), (ne.CLARIFY, 'contradiction'))
        # The only figure of its kind, the client speaks of themselves: the extractor's "foreign" is overruled.
        days = run('Мы отнесли заявку за двенадцать рабочих дней до начала периода. Успели?',
                   {'intent': 'threshold', 'values': [value('v1', 'за двенадцать рабочих дней', own=False)],
                    'limits': [{'block': block_no(DOCS, 'рабочих', {'duration'}), 'quote': 'не позднее чем за 10 рабочих дней', 'kind': 'min'}]})
        self.assertEqual((days['status'], days['values'][0]['owner']), (ne.PASS, 'only value'))
        self.assertIsNone(ne.CORRECTION.search('по договору у нас 800 квт'))  # "договор" is not "оговорился"
        self.assertTrue(ne.CORRECTION.search('я оговорился'))

    def test_code_reads_the_clients_figure_the_extractor_left_out(self):
        no = block_no(DOCS, 'Архив', {'duration'})
        limit = [{'block': no, 'quote': 'не менее года', 'kind': 'min'}]
        dialog = ('Пользователь: Архив у нас за восемнадцать месяцев.\nАссистент: Этого хватит.\n'
                  'Пользователь: Я погорячился: айтишники говорят, данные есть всего за восемь месяцев. Хватит?')
        read = run(dialog, {'intent': 'threshold', 'values': [value('v1', 'Хватит', turn=3)], 'limits': limit})
        self.assertEqual((read['status'], read['checks'][0]['client'], read['values'][-1]['owner']), (ne.FAIL, 8.0, 'code'))
        # Without the correction the figure is someone else's statement and is not read.
        hearsay = dialog.replace('Я погорячился: айтишники', 'Айтишники')
        self.assertEqual(run(hearsay, {'intent': 'threshold', 'values': [value('v1', 'Хватит', turn=3)], 'limits': limit})['status'], ne.NO_CHECK)

    def test_daily_consumption_is_compared_with_a_monthly_bound_only_when_every_month_length_agrees(self):
        limit = [{'block': block_no(DOCS, 'Ориентир', {'energy'}), 'quote': 'от 1 000 000 кВт·ч в месяц', 'kind': 'min'}]
        high = run('В сутки уходит тысяч сорок киловатт-часов. Дотягиваем?',
                   {'intent': 'threshold', 'values': [value('v1', 'тысяч сорок киловатт-часов')], 'limits': limit})
        self.assertEqual(high['status'], ne.PASS)
        self.assertIn('40 тыс. кВт·ч в сутки (за месяц от 1,12 млн кВт·ч до 1,24 млн кВт·ч) из вопроса — это БОЛЬШЕ', high['note'])
        edge = run('В сутки уходит 34 000 кВт·ч. Дотягиваем?', {'intent': 'threshold', 'values': [value('v1', '34 000 кВт·ч')], 'limits': limit})
        self.assertEqual((edge['status'], edge['checks'][0]['reason']), (ne.CLARIFY, 'period'))

    def test_without_a_valid_citation_a_verdict_needs_every_bound_to_agree(self):
        cited = [{'block': block_no(DOCS, 'складываются', {'power'}), 'quote': 'Мощности участков складываются', 'kind': 'min'}]
        low = run('У нас участок на 10 кВт. Пост поставите?', {'intent': 'threshold', 'values': [value('v1', '10 кВт')], 'limits': cited})
        self.assertEqual(low['status'], ne.FAIL)  # below "от 40 кВт, но менее 2 МВт" and below "от 15 до 500 кВт"
        split = run('У нас участок на 30 кВт. Пост поставите?', {'intent': 'threshold', 'values': [value('v1', '30 кВт')], 'limits': cited})
        self.assertEqual((split['status'], split['note']), (ne.NO_CHECK, ''))  # one bound passes, the other does not


class IntegrationTests(unittest.TestCase):
    def tearDown(self):
        config.reload()

    def api(self, replies):
        api = solution.API.__new__(solution.API)
        api.deadline, api.model, api.temperature, api.max_tokens = None, 'm', 0, 500
        api.sent = []

        def post(endpoint, body, attempts=None):
            api.sent.append(body)
            return {'choices': [{'message': {'content': replies[len(api.sent) - 1]}, 'finish_reason': 'stop'}]}
        api.post = post
        return api

    def test_engine_is_off_by_default_and_the_request_is_unchanged(self):
        config.reload()
        api = self.api(['Ответ.'])
        self.assertEqual(api.answer('У нас участок на 30 кВт. Пост поставите?', DOCS), 'Ответ.')
        self.assertEqual(len(api.sent), 1)
        self.assertNotIn('ПРОВЕРКА ЧИСЕЛ', api.sent[0]['messages'][1]['content'])

    def test_engine_adds_one_extraction_call_and_the_verdict_note(self):
        extraction = json.dumps({'check': True, 'values': [value('v1', '30 кВт')], 'limits': [{'block': 1, 'quote': 'от 40 кВт', 'kind': 'min'}]}, ensure_ascii=False)
        with config.override(NUMERIC_ENGINE=1):
            api = self.api([extraction, 'Нет, не поставим.'])
            self.assertEqual(api.answer('У нас участок на 30 кВт. Пост поставите?', DOCS), 'Нет, не поставим.')
            self.assertEqual(len(api.sent), 2)
            self.assertEqual(api.sent[0]['temperature'], 0)
            final = api.sent[1]['messages'][1]['content']
            # The note ends the request and carries the document's own sentence as the source of the bound.
            self.assertTrue(final.endswith('условие «не меньше 40 кВт» НЕ выполнено (раздел «Условия»: «Пост ставим на участках от 40 кВт, но менее 2 МВт»).'))
            self.assertIn('не пересчитывай его и не утверждай обратного', final)
            # No quantity in the last turn: no extraction call, and the request equals the default one.
            plain = self.api(['Ответ.'])
            plain.answer('Сколько стоит пост?', DOCS)
            self.assertEqual(len(plain.sent), 1)
            reference = self.api(['Ответ.'])
            with config.override(NUMERIC_ENGINE=0):
                reference.answer('Сколько стоит пост?', DOCS)
            self.assertEqual(plain.sent[0], reference.sent[0])

    def test_engine_failure_or_short_time_never_costs_the_answer(self):
        import time
        with config.override(NUMERIC_ENGINE=1):
            broken = self.api(['Ответ.'])
            original = broken.post

            def post(endpoint, body, attempts=None):
                if attempts == 2:
                    raise OSError('connection reset')
                return original(endpoint, body)
            broken.post = post
            self.assertEqual(broken.answer('У нас участок на 30 кВт. Пост поставите?', DOCS), 'Ответ.')
            self.assertNotIn('ПРОВЕРКА ЧИСЕЛ', broken.sent[0]['messages'][1]['content'])
            late = self.api(['Ответ.'])
            late.deadline = time.monotonic() + 60
            self.assertEqual(late.answer('У нас участок на 30 кВт. Пост поставите?', DOCS), 'Ответ.')
            self.assertEqual(len(late.sent), 1)


if __name__ == '__main__':
    unittest.main()
