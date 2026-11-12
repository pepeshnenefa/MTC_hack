"""Grounded answers: regression tests for three failures (synthetic knowledge base only).

1. Mixing products: a price, term or property of one product applied to another.
2. Hallucination: figures, contacts and channels that are in no document.
3. Over-caution: a refusal or an invented caveat where the documents answer plainly.

Each failure has a case that must be caught and a mirror case that must be left alone.
"""
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
import config  # noqa: E402
import context  # noqa: E402
import grounding  # noqa: E402
import retrieval  # noqa: E402
import solution  # noqa: E402

# Invented facts in the layout of the real base: product files, a multi-product reference, general pages.
DOCS = [
    ('20-products/23-solar.md',
     '# Солнечные электростанции под ключ\n\n## Сроки реализации\n\nТиповой срок реализации станции около 2 месяцев '
     'от подписания до запуска.\n\n## Оплата\n\nДоступна рассрочка на срок до 5 лет.\n\n'
     '## Размещение\n\nОриентир 70 руб за ватт относится к станции на кровле. Размещение на земле требует фундамента.\n'),
    ('20-products/22-askue.md',
     '# Облачная платформа учёта электроэнергии\n\n## Что это\n\nПлатформа собирает показания приборов автоматически. '
     'Минимального количества приборов учёта нет.\n\n## Сколько это стоит\n\nОриентир стоимости 25 000 руб в год за прибор учёта.\n\n'
     '## Как посмотреть до покупки\n\nДоступен демонстрационный кабинет.\n'),
    ('00-reference/02-product-scope-and-limits.md',
     '# Область применения\n\n## Солнечные электростанции\n\n- Частные жилые дома не входят в целевой сегмент.\n\n'
     '## Облачная платформа учёта\n\n- Платформа не выставляет счета.\n\n## Когда требуется специалист\n\n'
     'Специалист нужен, если сведений в базе недостаточно.\n'),
    ('20-products/20-portfolio.md',
     '# Обзор продуктов\n\n## Навигация по типу компании\n\n### Ретейл, общепит\n\nМагазинам подходит учёт по объектам.\n'),
    ('00-reference/03-calculation-and-requests.md',
     '# Расчёт по объекту\n\n## Обратный звонок\n\nОбратный звонок — способ продолжить общение. Для заявки нужны ФИО, '
     'телефон и название компании.\n'),
]
ENV = {'LLM_BASE_URL': 'http://proxy.invalid/v1', 'LLM_API_KEY': 'k', 'LLM_MODEL': 'm'}


def items():
    return retrieval.chunks(DOCS)


def section(heading):
    return next(s for s in items() if s['heading'] == heading)


def registry():
    return grounding.Registry(items())


def formatted(evidence, mode, answer):
    return 'ОСНОВАНИЕ: %s\nРЕЖИМ: %s\nОТВЕТ: %s' % (evidence, mode, answer)


class EntityTests(unittest.TestCase):
    def test_product_comes_from_file_title_or_reference_heading_never_from_deep_headings(self):
        self.assertEqual(grounding.section_products(section('Сроки реализации')), {'solar'})
        self.assertEqual(grounding.section_products(section('Облачная платформа учёта')), {'metering'})
        # "Ретейл" is search vocabulary of the metering topic, but the section is a general page.
        self.assertEqual(grounding.section_products(section('Ретейл, общепит')), set())
        self.assertEqual(grounding.section_products(section('Обратный звонок')), set())

    def test_names_are_read_from_the_knowledge_base(self):
        names = registry().names
        self.assertEqual(names['solar'], 'Солнечные электростанции под ключ')
        self.assertEqual(names['metering'], 'Облачная платформа учёта электроэнергии')
        self.assertEqual(grounding.label(section('Сроки реализации'), registry()), '[Относится к: Солнечные электростанции под ключ]')
        self.assertEqual(grounding.label(section('Обратный звонок'), registry()), '[Общие сведения]')

    def test_focus_is_the_product_named_last_then_history_then_evidence(self):
        both = 'По платформе учёта дают демо. А по солнечной станции так же можно?'
        self.assertEqual(grounding.focus(both), ({'metering', 'solar'}, 'solar'))
        dialog = 'Пользователь: Нужна платформа учёта на 35 счётчиков.\nАссистент: Хорошо.\nПользователь: А за два месяца уложитесь?'
        self.assertEqual(grounding.focus(dialog), ({'metering'}, 'metering'))
        evidence = [section('Сроки реализации'), section('Оплата'), section('Обратный звонок')]
        self.assertEqual(grounding.focus('Сколько ждать запуска?', evidence), ({'solar'}, 'solar'))
        self.assertEqual(grounding.focus('Сколько ждать запуска?', []), (set(), None))
        # The search vocabulary matches inside words ("запуСКАТься" ~ скат кровли); the product check must not.
        self.assertEqual(grounding.focus('Можно не возиться с расчётами и сразу запускаться на оптовый рынок?'), ({'wholesale'}, 'wholesale'))


class ContextTests(unittest.TestCase):
    def tearDown(self):
        config.reload()
        grounding.REGISTRY = None

    def test_sections_of_other_products_are_capped_and_replaced_by_the_product_in_question(self):
        solar = [section('Сроки реализации'), section('Оплата'), section('Размещение')]
        metering = [section('Что это'), section('Сколько это стоит'), section('Как посмотреть до покупки')]
        selected = [metering[0]] + solar
        arranged = grounding.arrange(selected, selected + metering[1:], {'metering'}, 1)
        self.assertEqual([s['heading'] for s in arranged], ['Что это', 'Сроки реализации', 'Сколько это стоит', 'Как посмотреть до покупки'])
        # Unknown product, or both products named: nothing is removed.
        self.assertEqual(grounding.arrange(selected, selected, set(), 1), selected)
        self.assertEqual(grounding.arrange(selected, selected, {'metering', 'solar'}, 0), selected)

    def test_a_section_that_mentions_the_product_in_question_is_not_foreign(self):
        bridge = dict(section('Оплата'), text='Приборы учёта станции заводятся в платформу учёта.')
        self.assertFalse(grounding.off_product(bridge, {'metering'}))
        self.assertTrue(grounding.off_product(section('Оплата'), {'metering'}))
        self.assertFalse(grounding.off_product(section('Обратный звонок'), {'metering'}))

    def test_tags_appear_only_when_switched_on(self):
        question = 'Нужна платформа учёта. За сколько запустите?'
        sources = [section('Что это'), section('Сроки реализации')]
        plain, _ = context.pack_context(question, sources, limit=12)
        self.assertNotIn('[Относится к', plain)
        self.assertTrue(plain.startswith('20-products/22-askue.md / '))
        with config.override(GROUNDED_CONTEXT=1):
            tagged, report = context.pack_context(question, sources, limit=12)
        self.assertTrue(tagged.startswith(grounding.NOTE))
        self.assertIn('[Относится к: Облачная платформа учёта электроэнергии] 20-products/22-askue.md / ', tagged)
        self.assertIn('[Относится к: Солнечные электростанции под ключ] 20-products/23-solar.md / ', tagged)
        self.assertLessEqual(report['characters'], int(config.get('CONTEXT_CHARS')))


class FormatTests(unittest.TestCase):
    def test_three_lines_are_split_and_modes_recognised(self):
        parsed = grounding.parse(formatted('«Минимального количества приборов учёта нет.»', 'ПОДТВЕРЖДЕНО', 'Да, подключаем.'))
        self.assertEqual((parsed['answer'], parsed['mode'], parsed['evidence'], parsed['formatted']),
                         ('Да, подключаем.', 'confirmed', 'Минимального количества приборов учёта нет.', True))
        self.assertEqual(grounding.parse(formatted('НЕТ', 'НЕТ ДАННЫХ', 'Таких сведений нет.'))['mode'], 'insufficient')
        self.assertEqual(grounding.parse(formatted('НЕТ', 'НЕТ ДАННЫХ', 'Таких сведений нет.'))['evidence'], '')
        self.assertEqual(grounding.parse(formatted('x y z', 'Неоднозначно.', 'Зависит от мощности.'))['mode'], 'ambiguous')
        # The cleaned completion arrives as one line.
        one_line = 'ОСНОВАНИЕ: Доступна рассрочка на срок до 5 лет. РЕЖИМ: ПОДТВЕРЖДЕНО ОТВЕТ: Да, рассрочка есть.'
        self.assertEqual(grounding.parse(one_line)['answer'], 'Да, рассрочка есть.')

    def test_answer_first_layout_keeps_the_answer_and_reads_the_service_lines(self):
        completion = 'Да, подключаем: минимума нет. Сколько у вас объектов?\nОСНОВАНИЕ: Минимального количества приборов учёта нет.\nРЕЖИМ: ПОДТВЕРЖДЕНО'
        parsed = grounding.parse(completion)
        self.assertEqual(parsed, {'answer': 'Да, подключаем: минимума нет. Сколько у вас объектов?', 'mode': 'confirmed',
                                  'evidence': 'Минимального количества приборов учёта нет.', 'formatted': True})
        # Cut off inside the service lines: the answer is intact.
        self.assertEqual(grounding.parse('Срока у нас нет. ОСНОВАНИЕ: Типовой срок реал')['answer'], 'Срока у нас нет.')
        with config.override(PROMPT='grounded_first'):
            self.assertIn('ровно три строки', solution.system_prompt())
            self.assertTrue(solution.user_message('Вопрос?', 'текст').endswith(solution.FORMAT_REMINDER_FIRST))
        with config.override(PROMPT='grounded'):
            self.assertIn('ПОСЛЕ ОТВЕТА', solution.system_prompt())
            self.assertNotIn('ровно три строки', solution.system_prompt())
        config.reload()

    def test_service_lines_never_reach_the_client_even_when_the_format_is_broken(self):
        self.assertEqual(grounding.parse('Да, подключаем.'), {'answer': 'Да, подключаем.', 'mode': None, 'evidence': '', 'formatted': False})
        broken = grounding.parse('Да, подключаем. ОСНОВАНИЕ: что-то РЕЖИМ: ПОДТВЕРЖДЕНО')
        self.assertEqual(broken['answer'], 'Да, подключаем.')
        self.assertEqual(grounding.parse('ОСНОВАНИЕ: что-то РЕЖИМ: ПОДТВЕРЖДЕНО')['answer'], '')
        self.assertEqual(grounding.parse('')['answer'], '')
        self.assertEqual(grounding.parse(None)['answer'], '')

    def test_default_request_is_untouched_and_grounded_instruction_extends_the_base_one(self):
        self.assertEqual(solution.system_prompt(), solution.SYSTEM_FULL)
        message = solution.user_message('Вопрос?', 'текст')
        self.assertEqual(message, 'ДОКУМЕНТЫ:\nтекст\n\nДИАЛОГ ИЛИ ВОПРОС:\nВопрос?')
        with config.override(PROMPT='grounded'):
            self.assertTrue(solution.system_prompt().startswith(solution.SYSTEM_FULL))
            self.assertIn('НЕТ ДАННЫХ', solution.system_prompt())
            self.assertTrue(solution.user_message('Вопрос?', 'текст').endswith(solution.FORMAT_REMINDER))
            self.assertEqual(solution.user_message('Вопрос?', 'текст', plain=True), message)
        self.assertEqual(solution.payload_context({'messages': [{}, {'content': message}]}), 'текст')


class EvidenceTests(unittest.TestCase):
    def test_quote_is_found_despite_wraps_and_markup_and_an_invented_one_is_not(self):
        sources = items()
        source, share = grounding.locate('Типовой срок реализации станции около 2 месяцев от подписания до запуска.', sources)
        self.assertEqual((source['heading'], share), ('Сроки реализации', 1.0))
        wrapped = dict(section('Оплата'), text='Доступна **рассрочка**\n    на срок до 5 лет.')
        self.assertEqual(grounding.locate('Доступна рассрочка на срок до 5 лет', [wrapped])[1], 1.0)
        self.assertLess(grounding.locate('Платформа учёта внедряется за две недели под ключ.', sources)[1], 0.5)
        self.assertEqual(grounding.locate('да', sources), (None, 0.0))
        # A section named instead of quoted is resolved to that section (its product can then be checked).
        named = '[Относится к: Солнечные электростанции под ключ] 20-products/23-solar.md / Солнечные электростанции под ключ / Сроки реализации'
        self.assertEqual(grounding.locate(named, sources)[0]['heading'], 'Сроки реализации')


class HallucinationTests(unittest.TestCase):
    """Failure 2: what is in no document must be caught; what is there must pass."""

    def check(self, answer, question='Вопрос?', sources=None):
        return grounding.ungrounded(answer, question, items() if sources is None else sources)

    def test_invented_contacts_are_caught(self):
        found = self.check('Адрес офиса: улица Мира, 35. Часы работы: 9:00–18:00, телефон +7 495 123-45-67, сайт energy.example.ru.')
        kinds = [kind for kind, _ in found]
        self.assertEqual(kinds.count('contact'), 4)
        self.assertIn(('contact', 'улица Мира, 35'), found)

    def test_invented_channel_is_caught_and_a_documented_one_is_not(self):
        self.assertEqual(self.check('Обратитесь в отдел продаж.'), [('channel', 'отдел продаж')])
        self.assertEqual(self.check('Напишите в техническую поддержку.')[0][0], 'channel')
        with_channel = items() + [dict(section('Обратный звонок'), text='Показание отправляется в Telegram-бот.')]
        self.assertEqual(self.check('Снимок отправляется в Telegram-бот.', sources=with_channel), [])

    def test_invented_figure_is_caught(self):
        self.assertEqual([kind for kind, _ in self.check('Платформа запускается за 14 дней и стоит 480 000 руб.')], ['figure', 'figure'])

    def test_figures_of_the_documents_and_of_the_client_pass(self):
        self.assertEqual(self.check('Срок около 2 месяцев, рассрочка до 5 лет, ориентир 25 000 руб в год за прибор.'), [])
        self.assertEqual(self.check('Ориентир — 25 тыс. руб в год за прибор.'), [])
        client = 'У нас сто восемьдесят киловатт и 35 счётчиков. Запишите: Иванов, +7 900 000-00-00.'
        self.assertEqual(self.check('Для 180 кВт и 35 счётчиков оформим заявку, телефон +7 900 000-00-00 записан.', client), [])

    def test_arithmetic_on_the_client_figures_passes(self):
        self.assertEqual(self.check('Вместе 300 + 280 + 240 = 820 кВт.', 'Три корпуса: 300, 280 и 240 киловатт.'), [])
        self.assertEqual(self.check('Это около 12,6 млн руб.', 'Станция на 180 кВт, по 70 рублей за ватт — сколько выйдет?'), [])
        self.assertEqual(self.check('Это 0,75 млн кВт·ч в месяц.', 'За год тратим 9 млн кВт·ч.'), [])
        self.assertEqual(self.check('1,6 МВт — это 1600 кВт.', 'По акту 1,6 МВт.'), [])

    def test_list_numbers_are_not_figures(self):
        self.assertEqual(self.check('Нужно знать: 1. Тип приборов. 2. Схему сбора. 3. Число объектов.'), [])


class ValidationTests(unittest.TestCase):
    QUESTION = 'Пользователь: Нужна платформа учёта на 35 счётчиков.\nАссистент: Подходит.\nПользователь: Запуск за два месяца — уложитесь?'

    def issues(self, completion, question=None, sources=None):
        parsed = grounding.parse(completion)
        return [code for code, _ in grounding.validate(question or self.QUESTION, parsed, items() if sources is None else sources, registry())]

    # Failure 1: mixing products.
    def test_confirmed_answer_resting_on_another_product_is_caught(self):
        completion = formatted('Типовой срок реализации станции около 2 месяцев от подписания до запуска.', 'ПОДТВЕРЖДЕНО',
                               'Да, запуск занимает около 2 месяцев.')
        self.assertEqual(self.issues(completion), ['other_product'])
        _, detail = grounding.validate(self.QUESTION, grounding.parse(completion), items(), registry())[0]
        self.assertEqual(detail, ('Солнечные электростанции под ключ', 'Облачная платформа учёта электроэнергии'))

    def test_the_same_quote_is_fine_when_the_client_asks_about_that_product(self):
        completion = formatted('Типовой срок реализации станции около 2 месяцев от подписания до запуска.', 'ПОДТВЕРЖДЕНО',
                               'Да, типовая станция запускается примерно за 2 месяца.')
        self.assertEqual(self.issues(completion, 'Солнечную станцию за два месяца запустите?'), [])

    def test_explaining_where_the_known_figure_belongs_is_not_a_transfer(self):
        completion = formatted('Типовой срок реализации станции около 2 месяцев от подписания до запуска.', 'НЕТ ДАННЫХ',
                               'Срок около 2 месяцев относится к солнечной станции; для платформы учёта ориентира по сроку у нас нет, его назовёт специалист.')
        self.assertEqual(self.issues(completion), [])

    # Failure 2: hallucination at the level of the evidence.
    def test_invented_evidence_is_caught(self):
        completion = formatted('Платформа учёта внедряется за две недели под ключ.', 'ПОДТВЕРЖДЕНО', 'Да, уложимся.')
        self.assertEqual(self.issues(completion), ['quote_missing'])
        self.assertEqual(self.issues(formatted('НЕТ', 'ПОДТВЕРЖДЕНО', 'Да, уложимся.')), ['no_evidence'])

    def test_honest_no_data_answer_passes(self):
        completion = formatted('НЕТ', 'НЕТ ДАННЫХ', 'Срока внедрения платформы учёта у нас нет; могу подключить специалиста.')
        self.assertEqual(self.issues(completion), [])

    # Failure 3: over-caution.
    def test_refusal_with_confirming_evidence_is_caught(self):
        completion = formatted('Минимального количества приборов учёта нет.', 'ПОДТВЕРЖДЕНО',
                               'К сожалению, не могу подтвердить, что шесть счётчиков подойдут, уточните у специалиста.')
        self.assertEqual(self.issues(completion, 'У нас шесть счётчиков, платформа учёта подойдёт?'), ['refusal_despite_evidence'])

    def test_direct_confirmed_answer_passes(self):
        completion = formatted('Минимального количества приборов учёта нет.', 'ПОДТВЕРЖДЕНО',
                               'Да, подойдёт: минимального количества приборов нет.')
        self.assertEqual(self.issues(completion, 'У нас шесть счётчиков, платформа учёта подойдёт?'), [])

    def test_plain_answer_is_checked_without_any_quote(self):
        # Failure 1 without the format: the figure stands in the documents only under another product,
        # and the client naming it himself ("за два месяца") does not make it a fact about his product.
        self.assertEqual(self.issues('Да, запуск занимает около 2 месяцев.'), ['other_product'])
        self.assertEqual(self.issues('Да, типовая станция запускается за 2 месяца.', 'Солнечную станцию за два месяца запустите?'), [])
        # Both products named by the client: figures of either are his subject.
        both = 'По солнечной станции срок два месяца, а платформу учёта когда запустите?'
        self.assertEqual(grounding.foreign('Станция — около 2 месяцев.', both, items()), [])
        # A figure that the product in question has too is not foreign.
        shared = items() + [dict(section('Что это'), text='Пилот занимает около 2 месяцев.')]
        self.assertEqual(self.issues('Пилот займёт около 2 месяцев.', sources=shared), [])
        self.assertEqual(grounding.measures('С 2025 года срок около 2 месяцев, цена 25 тыс. руб и 70 руб за ватт.'),
                         {(2.0, 'months'), (25000.0, 'money'), (70.0, 'money')})
        self.assertEqual(self.issues('Офис работает с 9:00 до 18:00.'), ['contact'])

    def test_feedback_names_the_findings_and_carries_no_ready_made_opening(self):
        text = grounding.feedback([('contact', 'улица Мира, 35'), ('other_product', ('Солнечные станции', 'Платформа учёта'))], True)
        self.assertIn('«улица Мира, 35»', text)
        self.assertIn('«Солнечные станции»', text)
        self.assertIn('ОСНОВАНИЕ и РЕЖИМ', text)
        self.assertNotIn('клиент', text.lower())  # that word in a service note turns the answer into the third person
        self.assertNotIn('ОСНОВАНИЕ', grounding.feedback([('figure', '14 дней')], False))


class FlowTests(unittest.TestCase):
    """The whole answer path with a scripted model."""
    QUESTION = ValidationTests.QUESTION

    def tearDown(self):
        config.reload()
        grounding.REGISTRY = None
        solution.API.template_arguments_rejected = False

    def run_flow(self, completions, question=None, settings=None, sources=None, search=None):
        seen = []
        script = list(completions)

        def transport(request, timeout):
            seen.append(json.loads(request.data))
            content = script.pop(0)
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}).encode())
        env = dict(ENV)
        for name, value in (settings if settings is not None else {'PROMPT': 'grounded', 'GROUNDED_CONTEXT': 1, 'GROUNDED_VALIDATE': 1}).items():
            env['ET_' + name] = str(value)
        with patch.dict(os.environ, env, clear=True), patch.object(solution, 'urlopen', side_effect=transport):
            api = solution.API()
            api.search = search
            grounding.REGISTRY = registry()
            answer = api.answer(question or self.QUESTION, items() if sources is None else sources)
        return answer, seen

    def test_clean_confirmed_answer_costs_one_call_and_only_the_answer_line_is_returned(self):
        question = 'У нас шесть счётчиков, платформа учёта подойдёт?'
        answer, seen = self.run_flow([formatted('Минимального количества приборов учёта нет.', 'ПОДТВЕРЖДЕНО', 'Да, подойдёт.')], question)
        self.assertEqual((answer, len(seen)), ('Да, подойдёт.', 1))
        self.assertIn('[Относится к: Облачная платформа учёта электроэнергии]', seen[0]['messages'][1]['content'])

    def test_cross_product_answer_is_corrected_once(self):
        wrong = formatted('Типовой срок реализации станции около 2 месяцев от подписания до запуска.', 'ПОДТВЕРЖДЕНО', 'Да, уложимся в 2 месяца.')
        right = formatted('НЕТ', 'НЕТ ДАННЫХ', 'Срок около 2 месяцев относится к солнечной станции; по платформе учёта срок назовёт специалист.')
        answer, seen = self.run_flow([wrong, right])
        self.assertEqual(len(seen), 2)
        self.assertTrue(answer.startswith('Срок около 2 месяцев относится к солнечной станции'))
        roles = [m['role'] for m in seen[1]['messages']]
        self.assertEqual(roles, ['system', 'user', 'assistant', 'user'])
        self.assertIn('Солнечные электростанции под ключ', seen[1]['messages'][3]['content'])

    def test_invented_contact_is_corrected_once(self):
        wrong = formatted('НЕТ', 'НЕТ ДАННЫХ', 'Адрес офиса: улица Мира, 35, работаем 9:00–18:00.')
        right = formatted('НЕТ', 'НЕТ ДАННЫХ', 'Адреса офиса у меня нет; можем оформить заявку или обратный звонок.')
        answer, seen = self.run_flow([wrong, right], 'Продиктуйте адрес офиса и часы работы.')
        self.assertEqual((answer, len(seen)), ('Адреса офиса у меня нет; можем оформить заявку или обратный звонок.', 2))

    def test_overcautious_answer_is_corrected_once(self):
        wrong = formatted('Минимального количества приборов учёта нет.', 'ПОДТВЕРЖДЕНО', 'К сожалению, не могу подтвердить это без специалиста.')
        right = formatted('Минимального количества приборов учёта нет.', 'ПОДТВЕРЖДЕНО', 'Да, подойдёт: минимального количества приборов нет.')
        answer, seen = self.run_flow([wrong, right], 'У нас шесть счётчиков, платформа учёта подойдёт?')
        self.assertEqual((answer, len(seen)), ('Да, подойдёт: минимального количества приборов нет.', 2))

    def test_persistent_hard_failure_becomes_a_hand_over_only_at_level_two(self):
        strict = {'PROMPT': 'grounded', 'GROUNDED_CONTEXT': 1, 'GROUNDED_VALIDATE': 2}
        # Failure 1: the evidence stays with another product after the correction.
        wrong = formatted('Типовой срок реализации станции около 2 месяцев от подписания до запуска.', 'ПОДТВЕРЖДЕНО', 'Да, уложимся в 2 месяца.')
        answer, seen = self.run_flow([wrong, wrong], settings=strict)
        self.assertEqual(len(seen), 2)
        self.assertIn('«Солнечные электростанции под ключ»', answer)
        self.assertIn('специалист', answer)
        self.assertNotIn('2 месяц', answer)
        # Failure 2: the invented contact stays.
        contact = formatted('НЕТ', 'НЕТ ДАННЫХ', 'Адрес офиса: улица Мира, 35.')
        answer, _ = self.run_flow([contact, contact], 'Продиктуйте адрес офиса.', strict)
        self.assertNotIn('Мира', answer)
        self.assertIn('специалист', answer)
        # Failure 3 (mirror): a finding code cannot be sure of (an inexact quote) costs no call by default
        # and never turns an answer into a refusal.
        soft = formatted('Платформа подходит всем.', 'ПОДТВЕРЖДЕНО', 'Да, подойдёт.')
        answer, seen = self.run_flow([soft], 'У нас шесть счётчиков, платформа учёта подойдёт?', strict)
        self.assertEqual((answer, len(seen)), ('Да, подойдёт.', 1))
        answer, seen = self.run_flow([soft, soft], 'У нас шесть счётчиков, платформа учёта подойдёт?', dict(strict, GROUNDED_SOFT_RETRY=1))
        self.assertEqual((answer, len(seen)), ('Да, подойдёт.', 2))
        # A clean answer is never replaced.
        clean = formatted('Минимального количества приборов учёта нет.', 'ПОДТВЕРЖДЕНО', 'Да, подойдёт.')
        self.assertEqual(self.run_flow([clean], 'У нас шесть счётчиков, платформа учёта подойдёт?', strict)[0], 'Да, подойдёт.')
        self.assertEqual(grounding.handoff([('quote_missing', 'x'), ('refusal_despite_evidence', '')]), '')

    def test_correction_that_is_no_better_is_discarded(self):
        wrong = formatted('НЕТ', 'НЕТ ДАННЫХ', 'Адрес офиса: улица Мира, 35.')
        answer, seen = self.run_flow([wrong, formatted('НЕТ', 'НЕТ ДАННЫХ', 'Адрес: проспект Ленина, 1.')], 'Продиктуйте адрес офиса.')
        self.assertEqual((answer, len(seen)), ('Адрес офиса: улица Мира, 35.', 2))

    def test_completion_without_an_answer_line_falls_back_to_the_base_instruction(self):
        answer, seen = self.run_flow(['ОСНОВАНИЕ: НЕТ РЕЖИМ: НЕТ ДАННЫХ', 'Срока у нас нет, подключим специалиста.'])
        self.assertEqual((answer, len(seen)), ('Срока у нас нет, подключим специалиста.', 2))
        self.assertEqual(seen[1]['messages'][0]['content'], solution.SYSTEM_FULL)
        self.assertNotIn(solution.FORMAT_REMINDER, seen[1]['messages'][1]['content'])

    def test_unformatted_completion_is_returned_as_a_plain_answer(self):
        answer, seen = self.run_flow(['Срока внедрения у нас нет, подключим специалиста.'])
        self.assertEqual((answer, len(seen)), ('Срока внедрения у нас нет, подключим специалиста.', 1))

    def test_no_data_triggers_a_second_look_and_a_verified_answer_replaces_the_refusal(self):
        class Lexical:
            def find(self, text, k=12):
                return [section('Сколько это стоит')]
        first = formatted('НЕТ', 'НЕТ ДАННЫХ', 'Стоимости у меня нет, подключу специалиста.')
        second = formatted('Ориентир стоимости 25 000 руб в год за прибор учёта.', 'ПОДТВЕРЖДЕНО', 'Ориентир — 25 000 руб в год за прибор.')
        settings = {'PROMPT': 'grounded', 'GROUNDED_CONTEXT': 1, 'GROUNDED_VALIDATE': 1, 'GROUNDED_RECHECK': 1}
        sources = [section('Что это'), section('Как посмотреть до покупки')]
        question = 'Сколько стоит платформа учёта?'
        answer, seen = self.run_flow([first, second], question, settings, sources, Lexical())
        self.assertEqual((answer, len(seen)), ('Ориентир — 25 000 руб в год за прибор.', 2))
        self.assertIn('Ориентир стоимости 25 000 руб', seen[1]['messages'][1]['content'])
        # A second answer that is not backed by a quote from the new sections does not replace the honest refusal.
        invented = formatted('Платформа стоит 10 руб.', 'ПОДТВЕРЖДЕНО', 'Платформа стоит 10 руб.')
        answer, seen = self.run_flow([first, invented], question, settings, sources, Lexical())
        self.assertEqual((answer, len(seen)), ('Стоимости у меня нет, подключу специалиста.', 2))
        # Without the switch a refusal is final after one call.
        answer, seen = self.run_flow([first], question, None, sources, Lexical())
        self.assertEqual(len(seen), 1)

    def test_validation_of_plain_answers_works_without_the_format(self):
        settings = {'GROUNDED_VALIDATE': 1}
        answer, seen = self.run_flow(['Офис на улице Мира, 35.', 'Адреса офиса у меня нет, оформим заявку.'], 'Где ваш офис?', settings)
        self.assertEqual((answer, len(seen)), ('Адреса офиса у меня нет, оформим заявку.', 2))
        self.assertEqual(seen[0]['messages'][0]['content'], solution.SYSTEM_FULL)
        self.assertNotIn('ОСНОВАНИЕ', seen[1]['messages'][3]['content'])

    def test_a_failure_inside_grounding_never_costs_the_answer(self):
        with patch.object(grounding, 'validate', side_effect=RuntimeError('boom')):
            answer, seen = self.run_flow([formatted('НЕТ', 'НЕТ ДАННЫХ', 'Срока у нас нет.')])
        self.assertEqual((answer, len(seen)), ('Срока у нас нет.', 1))

    def test_defaults_send_exactly_the_request_of_the_previous_version(self):
        answer, seen = self.run_flow(['Офис на улице Мира, 35.'], 'Где ваш офис?', {}, [section('Обратный звонок')])
        self.assertEqual((answer, len(seen)), ('Офис на улице Мира, 35.', 1))
        entry = section('Обратный звонок')
        expected = 'ДОКУМЕНТЫ:\n' + entry['source'] + ' / ' + entry['title'] + '\n' + entry['text'] + '\n\nДИАЛОГ ИЛИ ВОПРОС:\nГде ваш офис?'
        self.assertEqual(seen[0]['messages'], [{'role': 'system', 'content': solution.SYSTEM_FULL}, {'role': 'user', 'content': expected}])

    def test_no_time_left_means_no_extra_call(self):
        wrong = formatted('НЕТ', 'НЕТ ДАННЫХ', 'Адрес офиса: улица Мира, 35.')
        seen = []

        def transport(request, timeout):
            seen.append(1)
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': wrong}, 'finish_reason': 'stop'}]}).encode())
        env = dict(ENV, ET_PROMPT='grounded', ET_GROUNDED_VALIDATE='1')
        with patch.dict(os.environ, env, clear=True), patch.object(solution, 'urlopen', side_effect=transport):
            api = solution.API(deadline=solution.time.monotonic() + 8)
            answer = api.answer('Продиктуйте адрес офиса.', items())
        self.assertEqual((answer, len(seen)), ('Адрес офиса: улица Мира, 35.', 1))


class ServiceLineTests(unittest.TestCase):
    """Deviations of the format seen in other models must neither leak the service lines nor cut the answer."""
    ANSWER = 'Да, подключаем: минимального количества приборов учёта нет.'
    QUOTE = '«Минимального количества приборов учёта нет.»'

    def parsed(self, tail, answer=None):
        # The completion arrives cleaned: on one line, without double asterisks (solution.clean_answer).
        return grounding.parse(solution.clean_answer((answer or self.ANSWER) + '\n' + tail))

    def test_deviant_markers_never_reach_the_client(self):
        q = self.QUOTE
        for tail in ('ОСНОВАНИЕ — %s\nРЕЖИМ — ПОДТВЕРЖДЕНО' % q, 'ОСНОВАНИЕ - %s\nРЕЖИМ - ПОДТВЕРЖДЕНО' % q,
                     '[ОСНОВАНИЕ]: %s\n[РЕЖИМ]: ПОДТВЕРЖДЕНО' % q, 'ОСНОВАНИЕ： %s\nРЕЖИМ： ПОДТВЕРЖДЕНО' % q,
                     '*ОСНОВАНИЕ:* %s\n*РЕЖИМ:* ПОДТВЕРЖДЕНО' % q, '**ОСНОВАНИЕ:** %s\n**РЕЖИМ:** ПОДТВЕРЖДЕНО' % q,
                     'EVIDENCE: %s\nMODE: CONFIRMED' % q, '---\nОСНОВАНИЕ: %s\nРЕЖИМ: ПОДТВЕРЖДЕНО' % q,
                     'Служебные строки:\nОСНОВАНИЕ: %s\nРЕЖИМ: ПОДТВЕРЖДЕНО' % q, 'ОСНОВАНИЕ: %s\nРежим: подтверждено' % q):
            with self.subTest(tail=tail):
                parsed = self.parsed(tail)
                self.assertEqual(parsed['answer'], self.ANSWER)
                self.assertEqual(parsed['mode'], 'confirmed')
                self.assertEqual(parsed['evidence'], 'Минимального количества приборов учёта нет.')

    def test_the_word_inside_the_answer_is_not_a_marker(self):
        tail = 'ОСНОВАНИЕ: %s\nРЕЖИМ: ПОДТВЕРЖДЕНО' % self.QUOTE
        for answer in ('Да. Основание: договор и акт разграничения. Подать можно онлайн.',
                       'Режим: круглосуточный сбор показаний доступен.'):
            with self.subTest(answer=answer):
                parsed = self.parsed(tail, answer)
                self.assertEqual((parsed['answer'], parsed['mode']), (answer, 'confirmed'))

    def test_lower_case_markers_without_capitals_still_end_the_answer(self):
        self.assertEqual(grounding.parse('Да, подключаем. Основание: что-то Режим: подтверждено')['answer'], 'Да, подключаем.')


class ComputedNoteTests(unittest.TestCase):
    """Figures the numeric engine computed and put into the request are not inventions of the model."""
    QUESTION = 'Пользователь: Потребляем 300 кВт·ч в сутки. Проходим порог?'
    NOTE = ('ПРОВЕРКА ЧИСЕЛ (служебная пометка, в ответе её не упоминай).\n'
            '- 300 кВт·ч в сутки (за месяц от 8400 кВт·ч до 9300 кВт·ч) из вопроса — это МЕНЬШЕ, чем 10 тыс. кВт·ч в месяц.')
    ANSWER = 'Нет, 300 кВт·ч в сутки — это от 8 400 до 9 300 кВт·ч в месяц, меньше порога.'

    def test_note_figures_pass_and_other_figures_are_still_caught(self):
        self.assertIn('figure', [c for c, _ in grounding.ungrounded(self.ANSWER, self.QUESTION, items())])
        self.assertEqual(grounding.ungrounded(self.ANSWER, self.QUESTION, items(), self.NOTE), [])
        invented = self.ANSWER + ' Подключение займёт 17 дней.'
        self.assertEqual([c for c, _ in grounding.ungrounded(invented, self.QUESTION, items(), self.NOTE)], ['figure'])

    def test_engine_figures_cost_no_correction_and_no_hand_over(self):
        calls = []

        def transport(request, timeout):
            calls.append(json.loads(request.data))
            content = self.ANSWER + '\nОСНОВАНИЕ: НЕТ\nРЕЖИМ: НЕОДНОЗНАЧНО'
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}).encode())
        env = dict(ENV, ET_PROMPT='grounded', ET_GROUNDED_CONTEXT='1', ET_GROUNDED_VALIDATE='2', ET_NUMERIC_ENGINE='1')
        try:
            with patch.dict(os.environ, env, clear=True), patch.object(solution, 'urlopen', side_effect=transport), \
                    patch.object(solution.API, 'numeric_note', return_value=(self.NOTE, {'status': 'FAIL'})):
                grounding.REGISTRY = registry()
                answer = solution.API().answer(self.QUESTION, items())
        finally:
            config.reload()
            grounding.REGISTRY = None
        self.assertEqual((answer, len(calls)), (self.ANSWER, 1))
        self.assertIn(self.NOTE, calls[0]['messages'][1]['content'])


class PackagingTests(unittest.TestCase):
    def test_module_is_packaged_and_holds_no_knowledge_base_figures(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        import build_submission
        self.assertIn('grounding.py', build_submission.FILES)
        source = (ROOT / 'submission/grounding.py').read_text(encoding='utf-8')
        for fact in ('750', '25 000', '70 руб', '150 кВт', 'Лайт', 'Grade'):
            self.assertNotIn(fact, source)


if __name__ == '__main__':
    unittest.main()
