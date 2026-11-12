"""Deterministic checks placed next to the question: arithmetic and dialogue hygiene, never facts.

A small model (and sometimes a large one) reads "8 кВт" next to "от 15 до 500 кВт" and writes
"попадает в диапазон", or "1,8 млн" next to "от 1 000 000" and writes "ниже". The comparison is
plain arithmetic, so it is done here: the client's quantities are parsed (digits or words,
кВт/МВт/кВт·ч, тыс./млн), brought to one unit and compared with the quantities of the same
dimension found in the sections that are actually sent to the model. Which threshold applies and
what follows from it stays the model's job and comes from the documents.

No product facts live here: units, numerals and comparison words only.
"""
import re

import config
from retrieval import turns

UNITS = [
    # (pattern, dimension, factor to the base unit: кВт or кВт·ч)
    (r'(?:мвт|мегаватт[а-я]*)[\s·⋅∙•*\-‑–]*(?:ч|час|часа|часов|часах|часы)(?![а-я])', 'energy', 1000.0),
    (r'(?:квт|киловатт[а-я]*)[\s·⋅∙•*\-‑–]*(?:ч|час|часа|часов|часах|часы)(?![а-я])', 'energy', 1.0),
    (r'мвт(?![а-я])|мегаватт[а-я]*', 'power', 1000.0),
    (r'квт(?![а-я])|киловатт[а-я]*', 'power', 1.0),
]
MULTIPLIERS = [(r'тыс(?:яч[а-я]*)?\.?', 1e3), (r'млн\.?|миллион[а-я]*', 1e6), (r'млрд\.?|миллиард[а-я]*', 1e9)]
# Word numerals: (pattern, value, rank). Ranks let "сто двадцать пять" add up and stop "пять шесть".
NUMERALS = [
    (r'полтор[аы]|полутора', 1.5, 1),
    (r'один|одного|одному|одним|одном', 1, 1), (r'два|две|двух|двум|двумя', 2, 1), (r'три|тр[её]х|тремя', 3, 1),
    (r'четыре|четыр[её]х', 4, 1), (r'пять|пяти', 5, 1), (r'шесть|шести', 6, 1), (r'семь|семи', 7, 1),
    (r'восемь|восьми', 8, 1), (r'девять|девяти', 9, 1),
    (r'десять|десяти', 10, 2), (r'одиннадцат[ьи]', 11, 2), (r'двенадцат[ьи]', 12, 2), (r'тринадцат[ьи]', 13, 2),
    (r'четырнадцат[ьи]', 14, 2), (r'пятнадцат[ьи]', 15, 2), (r'шестнадцат[ьи]', 16, 2), (r'семнадцат[ьи]', 17, 2),
    (r'восемнадцат[ьи]', 18, 2), (r'девятнадцат[ьи]', 19, 2),
    (r'двадцат[ьи]', 20, 3), (r'тридцат[ьи]', 30, 3), (r'сорок|сорока', 40, 3), (r'пятьдесят|пятидесяти', 50, 3),
    (r'шестьдесят|шестидесяти', 60, 3), (r'семьдесят|семидесяти', 70, 3), (r'восемьдесят|восьмидесяти', 80, 3),
    (r'девяност[оа]', 90, 3),
    (r'сто|ста', 100, 4), (r'двести|двухсот', 200, 4), (r'триста|тр[её]хсот', 300, 4), (r'четыреста|четыр[её]хсот', 400, 4),
    (r'пятьсот|пятисот', 500, 4), (r'шестьсот|шестисот', 600, 4), (r'семьсот|семисот', 700, 4),
    (r'восемьсот|восьмисот', 800, 4), (r'девятьсот|девятисот', 900, 4),
]
DIGITS = r'\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?'
TOKEN = re.compile('|'.join(
    ['(?P<u%d>%s)' % (i, p) for i, (p, _, _) in enumerate(UNITS)] +
    ['(?P<m%d>(?<![а-я])(?:%s))' % (i, p) for i, (p, _) in enumerate(MULTIPLIERS)] +
    ['(?P<d>%s)' % DIGITS, r'(?P<r>--|[–—\-]|(?<![а-я])до(?![а-я]))', r'(?P<w>[а-яa-z]+)', r'(?P<p>[^\s\w])']))
NUMERAL = [(re.compile('(?:' + p + ')$'), v, rank) for p, v, rank in NUMERALS]
APPROX = {'на', 'по', 'около', 'примерно', 'где', 'то', 'этак', 'так', 'в', 'порядка'}
PERIODS = [('month', r'месяц|ежемесячн|месячн|/\s?мес|в мес'), ('year', r'(?<![а-я])год|ежегодн|годов'),
           ('day', r'сутк|суточн|в день|за день')]


def _tokens(text):
    """(kind, value, start, end): num, mult, unit=(dimension, factor), range, word, punct."""
    out = []
    text = text.lower().replace('ё', 'е')
    for m in TOKEN.finditer(text):
        kind = m.lastgroup
        if kind[0] == 'u' and kind[1:].isdigit():
            _, dimension, factor = UNITS[int(kind[1:])]
            out.append(('unit', (dimension, factor), m.start(), m.end()))
        elif kind[0] == 'm' and kind[1:].isdigit():
            out.append(('mult', MULTIPLIERS[int(kind[1:])][1], m.start(), m.end()))
        elif kind == 'd':
            out.append(('num', (float(re.sub(r'[   ]', '', m.group()).replace(',', '.')), 0), m.start(), m.end()))
        elif kind == 'r':
            out.append(('range', None, m.start(), m.end()))
        elif kind == 'w':
            numeral = next(((v, rank) for pattern, v, rank in NUMERAL if pattern.match(m.group())), None)
            out.append(('num', (float(numeral[0]), numeral[1]), m.start(), m.end()) if numeral
                       else ('word', m.group(), m.start(), m.end()))
        else:
            out.append(('punct', m.group(), m.start(), m.end()))
    return text, out


def _number(tokens, i):
    """Number starting at token i: digits, or word numerals of falling rank, plus an optional multiplier.
    Returns (value, next index) or None."""
    if i >= len(tokens) or tokens[i][0] != 'num':
        return None
    value, rank = tokens[i][1]
    j = i + 1
    if rank:  # "сто двадцать пять"; teens close the sequence
        while j < len(tokens) and tokens[j][0] == 'num' and 0 < tokens[j][1][1] < rank and rank != 2:
            value += tokens[j][1][0]
            rank = tokens[j][1][1]
            j += 1
    if j < len(tokens) and tokens[j][0] == 'mult':
        value *= tokens[j][1]
        j += 1
    return value, j


def _period(text, start, end):
    """Period named right after the unit, otherwise shortly before the number."""
    after, before = text[end:end + 24], text[max(0, start - 45):start]
    for window in (after, before):
        for name, pattern in PERIODS:
            if re.search(pattern, window):
                return name
    return None


def quantities(text, inverted=False):
    """Quantities with a unit as dicts: dimension, value (кВт or кВт·ч), period, position.

    inverted=True also reads the colloquial order of an approximate amount ("киловатт на восемь",
    "киловатт-часов тысяч двенадцать"); documents are parsed without it.
    """
    text, tokens = _tokens(text)
    found = []

    def add(value, unit, start, end):
        dimension, factor = unit
        found.append({'dimension': dimension, 'value': value * factor, 'position': start,
                      'period': _period(text, start, end) if dimension == 'energy' else None})

    used = set()
    for i, token in enumerate(tokens):
        if token[0] != 'unit':
            continue
        # Forward: NUMBER [MULT] UNIT, optionally preceded by NUMBER [MULT] (до|–) for a range.
        start = i - 1
        if start >= 0 and tokens[start][0] == 'mult':
            start -= 1
        while start > 0 and tokens[start][0] == 'num' and tokens[start - 1][0] == 'num' and tokens[start - 1][1][1] > tokens[start][1][1] > 0:
            start -= 1
        parsed = _number(tokens, start) if start >= 0 else None
        if parsed and parsed[1] == i:
            # "тысяч двенадцать киловатт-часов": a multiplier standing before the number.
            value = parsed[0]
            if start > 0 and tokens[start - 1][0] == 'mult' and tokens[i - 1][0] != 'mult' and not (start > 1 and tokens[start - 2][0] == 'num'):
                value *= tokens[start - 1][1]
            add(value, token[1], tokens[start][2], token[3])
            used.update(range(start, i + 1))
            # Earlier members of a range or list share the unit: "от 15 до 500 кВт", "15, 60 и 150 кВт".
            # They must grow towards the unit, which rejects years and other stray numbers.
            right, edge = value, start
            while edge >= 2 and (tokens[edge - 1][0] == 'range' or tokens[edge - 1][1] in (',', 'и', 'или')):
                left = edge - 2
                if tokens[left][0] == 'mult':
                    left -= 1
                while left > 0 and tokens[left][0] == 'num' and tokens[left - 1][0] == 'num' and tokens[left - 1][1][1] > tokens[left][1][1] > 0:
                    left -= 1
                other = _number(tokens, left) if left >= 0 else None
                if not other or other[1] != edge - 1 or other[0] >= right or any(k in used for k in range(left, edge - 1)):
                    break
                add(other[0], token[1], tokens[left][2], token[3])
                used.update(range(left, edge - 1))
                right, edge = other[0], left
            continue
        if i >= 1 and tokens[i - 1][0] == 'mult' and (i < 2 or tokens[i - 2][0] != 'num'):
            add(tokens[i - 1][1], token[1], tokens[i - 1][2], token[3])  # "миллион киловатт-часов"
            used.update((i - 1, i))
            continue
        if not inverted:
            continue
        # Inverted: UNIT [на|около|…] [MULT] NUMBER, when that number has no unit of its own.
        j = i + 1
        while j < len(tokens) and tokens[j][0] == 'word' and tokens[j][1] in APPROX and j - i <= 2:
            j += 1
        multiplier = 1.0
        if j < len(tokens) and tokens[j][0] == 'mult':
            multiplier = tokens[j][1]
            j += 1
        parsed = _number(tokens, j)
        if parsed and not (parsed[1] < len(tokens) and tokens[parsed[1]][0] == 'unit'):
            add(parsed[0] * multiplier, token[1], token[2], tokens[parsed[1] - 1][3])
            used.update(range(i, parsed[1]))
    return sorted(found, key=lambda q: q['position'])


def _plain(value):
    text = ('%.3f' % value).rstrip('0').rstrip('.')
    whole, _, fraction = text.partition('.')
    if len(whole) > 4:
        whole = re.sub(r'(?<=\d)(?=(?:\d{3})+$)', ' ', whole)
    return whole + (',' + fraction if fraction else '')


def show(dimension, value, period=None):
    """Human-readable quantity in the unit a reader expects, with the base unit when it differs."""
    suffix = {'month': ' в месяц', 'year': ' в год', 'day': ' в сутки'}.get(period, '')
    if dimension == 'power':
        if value >= 1000:
            return '%s МВт (= %s кВт)' % (_plain(value / 1000), _plain(value))
        return _plain(value) + ' кВт'
    if value >= 1e6:
        return '%s млн кВт·ч%s' % (_plain(value / 1e6), suffix)
    if value >= 1e4:
        return '%s тыс. кВт·ч%s' % (_plain(value / 1e3), suffix)
    return _plain(value) + ' кВт·ч' + suffix


def number_check(question, sources):
    """Lines comparing each quantity of the client with document quantities of the same dimension."""
    # Per dimension, the latest client turn that names it: a correction replaces the earlier value,
    # while a power given three turns ago still stands next to the consumption given now.
    # NUMBER_CHECK_SCOPE='last' reads the last client turn only: a quantity named earlier, when the
    # last question is about something else, pulled the answer towards thresholds (docs/checks-v1.md).
    client_turns = [text for is_user, text in turns(question) if is_user]
    if config.get('NUMBER_CHECK_SCOPE') == 'last':
        client_turns = client_turns[-1:]
    latest = {}
    for text in client_turns:
        found = quantities(text, inverted=True)
        for dimension in {q['dimension'] for q in found}:
            latest[dimension] = [q for q in found if q['dimension'] == dimension]
    client = [q for dimension in ('power', 'energy') for q in latest.get(dimension, [])]
    if not client:
        return []
    documents = []
    for source in sources:
        # A figure inside «…» is a quoted wording, usually one the documents forbid ("нельзя говорить «от N кВт»").
        documents.extend(quantities(re.sub(r'«[^»]*»', ' ', source.get('text', ''))))
    lines = []
    seen = set()
    limit = int(config.get('NUMBER_CHECK_VALUES'))
    for q in client[:4]:
        key = (q['dimension'], q['value'], q['period'])
        if key in seen:
            continue
        seen.add(key)
        value, note = q['value'], ''
        pool = [d for d in documents if d['dimension'] == q['dimension']]
        if q['dimension'] == 'energy':
            periods = {d['period'] for d in pool if d['period']}
            if q['period'] == 'year' and periods == {'month'}:
                value, note = q['value'] / 12, ' (около %s)' % show('energy', q['value'] / 12, 'month')
            elif q['period'] == 'month' and periods == {'year'}:
                value, note = q['value'] * 12, ' (около %s)' % show('energy', q['value'] * 12, 'year')
            elif q['period'] and periods and q['period'] not in periods:
                continue  # Different periods that cannot be converted safely: say nothing.
            elif not q['period'] and periods:
                note = ' (период не назван)'
        values = sorted({round(d['value'], 6) for d in pool})
        if not values:
            continue
        period = None
        if q['dimension'] == 'energy':
            named = {d['period'] for d in pool if d['period']}
            period = named.pop() if len(named) == 1 else None
        below = [v for v in values if v < value][-limit:]
        equal = [v for v in values if v == value]
        above = [v for v in values if v > value][:limit]
        parts = []
        if below:
            parts.append('БОЛЬШЕ, чем ' + ', '.join(show(q['dimension'], v, period) for v in below))
        if equal:
            parts.append('РАВНО ' + show(q['dimension'], equal[0], period))
        if above:
            parts.append('МЕНЬШЕ, чем ' + ', '.join(show(q['dimension'], v, period) for v in above))
        lines.append('- %s из вопроса%s — это %s (числа из документов).' % (show(q['dimension'], q['value'], q['period']), note, '; '.join(parts)))
    return lines


def block(question, sources):
    """Text appended after the dialogue, or '' when no check applies."""
    parts = []
    if config.get('NUMBER_CHECK'):
        lines = number_check(question, sources)
        if lines:
            # The notes never say "клиент": with that word the model started to answer about the client
            # in the third person (docs/checks-v1.md).
            parts.append('СВЕРКА ЧИСЕЛ (служебная пометка, в ответе её не упоминай; только арифметика — какой порог относится '
                         'к вопросу и что из него следует, определи по документам):\n' + '\n'.join(lines))
    if config.get('ASSISTANT_CHECK') and any(not is_user and text for is_user, text in turns(question)):
        # No ready-made opening phrase here: a literal template next to the question was copied into
        # every dialogue answer, including those where the assistant had been right (docs/checks-v1.md).
        parts.append('ПРОВЕРКА ДИАЛОГА (служебная пометка, в ответе её не упоминай): реплики «Ассистент» написаны до тебя '
                     'и не проверены. Сверь их числа, пороги и обещания с документами. Расходятся — поправь прямо и не подтверждай '
                     'вывод, построенный на ошибке. Совпадают — подтверди и ничего не исправляй.')
    return '\n\n'.join(parts)
