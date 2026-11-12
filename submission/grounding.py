"""Grounded answers: product labels, an evidence-first answer format and deterministic validation.

One mechanism against three failures seen on real answers (docs/grounded-v1.md):
  * a price, term or property of one product carried over to another;
  * numbers, contacts and conditions that are in no document;
  * a refusal or an invented caveat where the documents answer plainly.

Every section is tagged with the product it describes (read from the headings of the knowledge
base at run time). The model first quotes the sentence its answer rests on and names one of three
modes: CONFIRMED, INSUFFICIENT or AMBIGUOUS. Code then checks what can be checked without a model:
the quote is really in the documents, it belongs to the product in question, and every number and
contact of the answer occurs in the documents or in the dialogue. A failed check costs one
corrective regeneration; nothing else calls the model.

No product facts live here: units, numerals and wording patterns only.
"""
import re
from itertools import combinations

from retrieval import dialogue, topics, turns
import checks

# ----------------------------------------------------------------------------- entities

def section_products(source):
    """Products a section describes: topics of its file title and of its second-level heading.

    Deeper headings are not read: they name client types and details, not products."""
    parts = source.get('title', source.get('heading', '')).split(' / ')
    return topics(' / '.join(parts[:2]))


REGISTRY = None  # set once per run from the whole knowledge base (solution.main)


class Registry:
    """Product names as the knowledge base spells them (the title of the product's own file)."""

    def __init__(self, items=()):
        self.names = {}
        for item in items:
            top = item.get('title', item.get('heading', '')).split(' / ')[0]
            found = topics(top)
            if len(found) == 1:
                self.names.setdefault(next(iter(found)), top)

    def name(self, products):
        return ', '.join(self.names.get(p, p) for p in sorted(products))


def focus(question, sources=()):
    """Products the last client turn is about: (all named, the one named last).

    Falls back to earlier client turns, then to the product that dominates the strongest evidence."""
    latest, history = dialogue(question)
    for text in [latest] + list(reversed(history)):
        named = _mentions(text)
        if named:
            return set(named), named[-1]
    votes = {}
    for source in list(sources)[:3]:
        for product in section_products(source):
            votes[product] = votes.get(product, 0) + 1
    winners = [p for p, n in votes.items() if n >= 2]
    return (set(winners), winners[0]) if len(winners) == 1 else (set(), None)


def _mentions(text):
    """Products in order of their last mention."""
    from retrieval import TOPICS
    text = text.lower().replace('ё', 'е')
    last = {}
    for name, pattern in TOPICS.items():
        # At a word start only: the search vocabulary is loose on purpose ("запуСКАТься" must not name a roof).
        for match in re.finditer(r'(?<![а-яa-z])(?:%s)' % pattern, text):
            last[name] = match.start()
    return sorted(last, key=last.get)


def off_product(source, active):
    """A section of another product that does not even mention the product in question."""
    own = section_products(source)
    return bool(active) and bool(own) and not (own & active) and not (topics(source.get('text', '')) & active)


def label(source, registry):
    own = section_products(source)
    return '[Относится к: %s]' % registry.name(own) if own else '[Общие сведения]'


def arrange(selected, pool, active, limit):
    """Entity-aware selection: at most `limit` sections of other products; freed places go to the
    next sections of the product in question (or general ones) from the ranked pool."""
    if not active or limit < 0:
        return selected
    kept, dropped = [], 0
    for source in selected:
        if source.get('forced') or not off_product(source, active):
            kept.append(source)
        elif sum(off_product(s, active) for s in kept) < limit:
            kept.append(source)
        else:
            dropped += 1
    if dropped:
        present = {_key(s) for s in selected}
        for source in pool:
            if dropped <= 0:
                break
            if _key(source) not in present and not source.get('forced') and not off_product(source, active):
                kept.append(source)
                present.add(_key(source))
                dropped -= 1
    return kept


def _key(source):
    return source['source'], source.get('title', source.get('heading')), source['text']


NOTE = ('Перед каждым разделом указано, к какому продукту или теме он относится. Цена, срок, порог, гарантия или свойство '
        'действуют только для того продукта и варианта, для которых они указаны.')


def render(sources, registry):
    """Documents block with a product tag in front of every section."""
    return [label(s, registry) + ' ' + s['source'] + ' / ' + s.get('title', s['heading']) + '\n' + s['text'] for s in sources]


# ----------------------------------------------------------------------------- answer format

MODES = {'confirmed': 'CONFIRMED', 'insufficient': 'INSUFFICIENT', 'ambiguous': 'AMBIGUOUS'}

_MODES = '''КАК ВЫБРАТЬ РЕЖИМ:
- ПОДТВЕРЖДЕНО — основание относится именно к тому продукту и варианту, о котором идёт речь, и прямо отвечает на вопрос. Отвечай уверенно и прямо: да, нет или конкретный факт. Приводи только оговорки из самих документов; не добавляй условий, сомнений и отсылок к специалисту, которых в документах нет.
- НЕТ ДАННЫХ — в документах нет ответа: нужных сведений нет вовсе либо они указаны только для другого продукта, другого варианта или другого случая. Скажи об этом прямо в первом предложении, поясни, к чему относится известное, и предложи подключить специалиста. Не называй цифр, сроков, цен, адресов и каналов связи, которых нет в документах, и не обещай, что ответ появится после уточнений.
- НЕОДНОЗНАЧНО — документы дают разный ответ в зависимости от обстоятельства, которое не названо, или расходятся между собой. Назови, от чего зависит ответ, и задай один уточняющий вопрос либо предложи специалиста.
Перед каждым разделом документов указано, к какому продукту он относится. Цена, срок, порог, гарантия или свойство одного продукта не действуют для другого.'''

# The answer comes first and is written exactly as without the format; the two service lines follow
# it. With the evidence in front (FORMAT_FIRST) a small model shortened its answers to one sentence
# and dropped the caveats (docs/grounded-v1.md), so that order is kept as a variant only.
FORMAT = '''ПОСЛЕ ОТВЕТА добавь с новой строки две служебные строки (их увидит только проверка, не клиент):
ОСНОВАНИЕ: дословно одно-два предложения из документов, на которых держится ответ; если таких нет — слово НЕТ.
РЕЖИМ: одно из трёх — ПОДТВЕРЖДЕНО, НЕТ ДАННЫХ или НЕОДНОЗНАЧНО.
Сам ответ пиши как обычно, по правилам выше, без слов «основание» и «режим».

''' + _MODES

FORMAT_FIRST = '''ФОРМАТ ОТВЕТА — ровно три строки, без другого текста:
ОСНОВАНИЕ: дословно одно-два предложения из документов, которые отвечают на последнюю реплику; если таких нет — слово НЕТ.
РЕЖИМ: одно из трёх — ПОДТВЕРЖДЕНО, НЕТ ДАННЫХ или НЕОДНОЗНАЧНО.
ОТВЕТ: текст для клиента по правилам выше.

''' + _MODES

_FIELD = re.compile(r'(ОСНОВАНИЕ|РЕЖИМ|ОТВЕТ)\s*:\s*', re.I)


def parse(text):
    """Split a formatted completion. Always returns a dict; `answer` is '' only for an empty text.

    A completion without the markers is taken as a plain answer (mode unknown)."""
    text = (text or '').strip()
    fields = {}
    matches = list(_FIELD.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields.setdefault(match.group(1).upper(), text[match.end():end].strip())
    formatted = bool(fields)
    if fields.get('ОТВЕТ'):
        answer = fields['ОТВЕТ']
    else:
        # No answer marker: the answer is what stands before the service lines.
        answer = text if not matches else text[:matches[0].start()].strip()
    raw_mode = fields.get('РЕЖИМ', '').upper().replace('Ё', 'Е')
    mode = None
    if re.match(r'\W*(ПОДТВ|CONFIRM)', raw_mode):
        mode = 'confirmed'
    elif re.match(r'\W*(НЕТ|INSUFF|НЕДОСТАТ)', raw_mode):
        mode = 'insufficient'
    elif re.match(r'\W*(НЕОДН|AMBIG)', raw_mode):
        mode = 'ambiguous'
    evidence = fields.get('ОСНОВАНИЕ', '').strip(' «»"\'')
    if re.fullmatch(r'(?i)нет[.!]?|—|-|', evidence):
        evidence = ''
    return {'answer': answer.strip(), 'mode': mode, 'evidence': evidence, 'formatted': formatted}


# ----------------------------------------------------------------------------- fact-level checks

def _words(text):
    return re.findall(r'[а-яa-z0-9]+', text.lower().replace('ё', 'е'))


def _shingles(words, size=4):
    return {tuple(words[i:i + size]) for i in range(max(0, len(words) - size + 1))}


def locate(quote, sources):
    """Section the quote comes from and how much of it is found there: (source or None, share 0..1).

    Word 4-grams tolerate line wraps, markdown and small omissions; a table row, whose cells are
    interleaved in the file, is accepted when nearly all of its words stand in one section."""
    # A model may name the section instead of quoting it: "22-askue.md / … / Кому подходит".
    for source in sources:
        name = source.get('source', '').replace('\\', '/').split('/')[-1]
        if name and name in quote and source.get('heading', '') and source['heading'] in quote:
            return source, 1.0
    words = _words(quote)
    if len(words) < 3:
        return None, 0.0
    best, share = None, 0.0
    grams = _shingles(words) if len(words) >= 4 else None
    for source in sources:
        body = _words(source.get('title', '') + ' ' + source.get('text', ''))
        if grams:
            found = len(grams & _shingles(body)) / len(grams)
        else:
            found = 1.0 if ' '.join(words) in ' '.join(body) else 0.0
        if found < 0.5 and len(words) >= 5:
            bag = set(body)
            loose = sum(w in bag for w in words) / len(words)
            found = max(found, 0.5 if loose >= 0.9 else 0.0)
        if found > share:
            best, share = source, found
    return best, share


_NUMBER = r'\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?'
_SCALE = [(r'тыс(?:яч[а-я]*)?\.?', 1e3), (r'млн\.?|миллион[а-я]*', 1e6), (r'млрд\.?|миллиард[а-я]*', 1e9)]
_FIGURE = re.compile(r'(?<![\w.,:])(%s)(?![\d:])\s*((?:%s)(?![а-я]))?' % (_NUMBER, '|'.join(p for p, _ in _SCALE)), re.I)
# Word forms that stand for a number in the documents ("около полугода", "вдвое").
_SPOKEN = [(r'полугод|полгода', 6.0), (r'полтор|полутор', 1.5), (r'вдвое|двукратн', 2.0), (r'втрое', 3.0), (r'вчетверо', 4.0)]
_HARD = [
    ('адрес', r'(?<![а-я])(?:ул\.|улиц[аеыу]|проспект[а-я]*|пр-т|переул[а-я]*|шоссе|д\.\s?\d+)(?:\s+[а-яё-]+){0,2}(?:,?\s*(?:д\.\s*)?\d+[а-я]?)?'),
    ('время работы', r'(?<!\d)\d{1,2}[:.]\d{2}(?!\d)\s*(?:[-–—]|до)\s*\d{1,2}[:.]\d{2}|(?<!\d)\d{1,2}:\d{2}(?!\d)'),
    ('телефон', r'(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}'),
    ('почта', r'[\w.+-]+@[\w-]+\.[\w.]+'),
    ('сайт', r'(?:https?://|www\.)\S+|(?<![\w@])[\w-]+\.(?:ru|рф|com|org|net|su)(?![\w])'),
]
# Contact channels and departments; a word of this kind must occur in the documents or the dialogue.
_CHANNELS = [
    ('отдел продаж', r'отдел[а-я]* продаж'), ('техническая поддержка', r'тех(?:ническ[а-я]*\s+)?поддержк|службу? поддержки|саппорт'),
    ('горячая линия', r'горяч[а-я]* лини'), ('колл-центр', r'колл-?центр|контакт-?центр'),
    ('сайт', r'на (?:нашем |официальном )?сайте|через сайт'), ('мессенджер', r'whatsapp|ватсап|вайбер|viber|мессенджер'),
    ('телеграм', r'telegram|телеграм'), ('электронная почта', r'электронн[а-я]* почт|e-?mail|имейл'),
    ('офис', r'(?:в|наш[а-я]*|ближайш[а-я]*) офис'), ('личный менеджер', r'персональн[а-я]* менеджер|личн[а-я]* менеджер'),
]


def figures(text):
    """Numbers written in digits, with a thousand/million word applied: {value}."""
    out = set()
    for match in _FIGURE.finditer(text.replace('−', '-')):
        value = float(re.sub(r'[   ]', '', match.group(1)).replace(',', '.'))
        if match.group(2):
            value *= next(f for p, f in _SCALE if re.fullmatch(p, match.group(2).lower()))
        out.add(value)
    return out


def spoken(text):
    """Numbers written in words: numerals with multipliers and a few fixed forms."""
    text, tokens = checks._tokens(text)
    out = set()
    index = 0
    while index < len(tokens):
        parsed = checks._number(tokens, index) if tokens[index][0] == 'num' else None
        if parsed:
            out.add(parsed[0])
            index = parsed[1]
        else:
            index += 1
    for pattern, value in _SPOKEN:
        if re.search(pattern, text):
            out.add(value)
    return out


def _same(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _near(a, b):
    """Rounded results of arithmetic: within one percent."""
    return b != 0 and abs(a - b) <= 0.01 * abs(b)


def _scaled(value, pool, test):
    """value equals a pool number up to a unit change (Вт/кВт/МВт, руб/тыс./млн, share/percent)."""
    return any(test(value * factor, other) for other in pool for factor in (1, 1e3, 1e-3, 1e6, 1e-6, 100, 0.01))


def derivable(value, client, documents):
    """Arithmetic a consultant does with the client's own figures: a sum of several of them,
    a product, ratio or difference of a client figure with another client or document figure,
    or a yearly/monthly conversion."""
    client = sorted(client)
    for size in (2, 3, 4):
        if len(client) >= size and any(_scaled(value, [sum(group)], _near) for group in combinations(client[:12], size)):
            return True
    others = set(client) | set(documents) | {12.0}
    for a in client:
        for b in others:
            if a == 0 or b == 0:
                continue
            if _scaled(value, [a * b, a / b, b / a, abs(a - b)], _near):
                return True
    return False


def ungrounded(answer, question, sources):
    """Figures and contacts of the answer that occur neither in the documents sent to the model nor in
    the dialogue (and are not arithmetic on the client's figures). Returns a list of text fragments."""
    documents_text = '\n'.join(s.get('title', '') + '\n' + s.get('text', '') for s in sources)
    known_text = documents_text + '\n' + question
    low = known_text.lower().replace('ё', 'е')
    answer_low = answer.lower().replace('ё', 'е')
    issues = []
    for name, pattern in _HARD:
        for match in re.finditer(pattern, answer_low):
            fragment = answer[match.start():match.end()].strip()
            if re.sub(r'\W', '', fragment.lower().replace('ё', 'е')) not in re.sub(r'\W', '', low):
                issues.append(('contact', fragment))
    for name, pattern in _CHANNELS:
        match = re.search(pattern, answer_low)
        if match and not re.search(pattern, low):
            issues.append(('channel', answer[match.start():match.end()].strip()))
    client = figures(question) | spoken(question)
    documents = figures(documents_text) | spoken(documents_text)
    known = client | documents
    covered = [m.span() for _, p in _HARD for m in re.finditer(p, answer_low)]
    for match in _FIGURE.finditer(answer.replace('−', '-')):
        if any(start <= match.start() < end for start, end in covered):
            continue
        if re.match(r'[.)]\s+[А-ЯЁA-Z]', answer[match.end():]) and re.search(r'(?:^|[:;.?!]\s*)$', answer[:match.start()]):
            continue  # "1. … 2. …": the number of a list item
        value = next(iter(figures(match.group(0))), None)
        if value is None or _scaled(value, known, _same):
            continue
        if derivable(value, client, documents):
            continue
        issues.append(('figure', _fragment(answer, match)))
    seen, unique = set(), []
    for issue in issues:
        if issue not in seen:
            seen.add(issue)
            unique.append(issue)
    return unique


_UNITS = [('energy', r'(?:квт|мвт|киловатт|мегаватт)[а-я]*[\s·⋅∙•*\-‑–]*ч'), ('power', r'квт|мвт|киловатт|мегаватт|вт(?![а-я])'),
          ('voltage', r'кв(?![а-я])'), ('money', r'₽|руб'), ('percent', r'%|процент'), ('years', r'лет(?![а-я])|год'),
          ('months', r'месяц|мес(?![а-я])'), ('days', r'дн[еяю]|день|сут'), ('hours', r'час')]
_MEASURE = re.compile(r'(?<![\w.,:])(%s)(?![\d:])\s*((?:%s)(?![а-я]))?\s*(?:за\s+)?(%s)' % (
    _NUMBER, '|'.join(p for p, _ in _SCALE), '|'.join('(?P<%s>%s)' % (name, pattern) for name, pattern in _UNITS)), re.I)


def measures(text):
    """Figures with a unit: {(value, unit kind)} — "2 месяцев" and "2 кВт" are different facts."""
    out = set()
    for match in _MEASURE.finditer(text.lower().replace('ё', 'е').replace('−', '-')):
        value = float(re.sub(r'[ \u00a0\u202f]', '', match.group(1)).replace(',', '.'))
        if match.group(2):
            value *= next(f for p, f in _SCALE if re.fullmatch(p, match.group(2)))
        kind = next(name for name, _ in _UNITS if match.group(name))
        if kind == 'years' and 1900 <= value <= 2100:
            continue  # a calendar year, not a duration
        out.add((value, kind))
    return out


def foreign(answer, question, sources):
    """Figures of the answer that the documents give only for another product than the one in question.

    The client may have named such a figure himself ("у вас написано — два месяца"); that does not make
    it a fact about his product, so the dialogue is not consulted. Returns (figure text, product set)."""
    active, _ = focus(question, sources)
    if not active:
        return []
    own, alien = set(), {}
    for source in sources:
        found = measures(source.get('title', '') + ' ' + source.get('text', ''))
        if off_product(source, active):
            for item in found:
                alien.setdefault(item, set()).update(section_products(source))
        else:
            own |= found
    out = []
    low = answer.lower().replace('ё', 'е').replace('−', '-')
    for match in _MEASURE.finditer(low):
        item = next(iter(measures(match.group(0))), None)
        if item in alien and item not in own:
            out.append((answer[match.start():match.end()].strip(), alien[item]))
    return out


def _fragment(text, match):
    """The number with the two words that follow it, to show which fact is meant."""
    tail = re.match(r'[^.;!?\n]{0,28}', text[match.end():]).group(0)
    tail = ' '.join(tail.split()[:2])
    return (match.group(0).strip() + (' ' + tail if tail else '')).strip(' ,:—-')


_REFUSAL = re.compile(r'^[^.!?]{0,90}(?:нет (?:таких |точных |этих )?(?:сведени|данн|информаци)|не могу (?:подтвердить|назвать|сказать)|'
                      r'не располага|не указан|не описан|к сожалению)', re.I)


def validate(question, parsed, sources, registry):
    """Checks of one parsed completion. Returns a list of (code, detail) issues; empty when clean."""
    issues = []
    answer, mode, evidence = parsed['answer'], parsed['mode'], parsed['evidence']
    active, primary = focus(question, sources)
    if parsed['formatted']:
        if mode is None:
            issues.append(('format', 'режим не указан'))
        if mode == 'confirmed' and not evidence:
            issues.append(('no_evidence', ''))
        if evidence:
            source, share = locate(evidence, sources)
            if source is None or share < 0.5:
                issues.append(('quote_missing', evidence[:160]))
            elif mode == 'confirmed' and off_product(source, active):
                # A product the client did not name at all; with two products named, either may carry the answer.
                issues.append(('other_product', (registry.name(section_products(source)), registry.name(active))))
        if mode == 'confirmed' and _REFUSAL.search(answer.replace('ё', 'е')):
            issues.append(('refusal_despite_evidence', ''))
    if not any(code == 'other_product' for code, _ in issues) and mode != 'insufficient':
        # No quote needed: a figure that the documents give only for another product.
        alien = foreign(answer, question, sources)
        if alien:
            products = set().union(*(found for _, found in alien))
            issues.append(('other_product', (registry.name(products), registry.name(active))))
    issues.extend(ungrounded(answer, question, sources))
    return issues


# Findings code can be sure of. The rest (an inexact quote, a missing mode) are recorded for
# debugging and cost a corrective call only when GROUNDED_SOFT_RETRY is on: a small model quotes
# loosely, and its "corrections" of such findings were worse than the first answer as often as better.
HARD = ('other_product', 'contact', 'channel', 'figure', 'refusal_despite_evidence')


def hard(issues):
    return [issue for issue in issues if issue[0] in HARD]


def feedback(issues, formatted):
    """Corrective message after a failed check: findings only, no ready-made opening phrase."""
    lines = []
    loose = [detail for code, detail in issues if code in ('figure', 'contact', 'channel')]
    if loose:
        lines.append('в документах и в диалоге нет таких сведений из ответа: ' + '; '.join('«%s»' % d for d in loose[:6]) +
                     '. Убери их. Если без них ответить нельзя — прямо скажи, что таких сведений нет, и предложи специалиста.')
    for code, detail in issues:
        if code == 'quote_missing':
            lines.append('строки «ОСНОВАНИЕ» нет в документах дословно. Выпиши предложение, которое там действительно есть; '
                         'если подходящего нет — ОСНОВАНИЕ: НЕТ, режим НЕТ ДАННЫХ.')
        elif code == 'no_evidence':
            lines.append('режим ПОДТВЕРЖДЕНО указан без основания. Приведи дословное основание из документов или смени режим.')
        elif code == 'other_product':
            lines.append('основание взято из раздела о продукте «%s», а последняя реплика — о продукте «%s». Сведения одного '
                         'продукта на другой не переносятся. Найди основание в разделах о нужном продукте; если его там нет — '
                         'режим НЕТ ДАННЫХ: скажи, что для этого продукта таких сведений нет и к чему относится известное.' % detail)
        elif code == 'refusal_despite_evidence':
            lines.append('режим ПОДТВЕРЖДЕНО, а ответ начинается с отказа. Если основание отвечает на вопрос — ответь прямо и уверенно; '
                         'если не отвечает — смени режим.')
        elif code == 'format':
            lines.append('не указан режим.')
    head = 'Проверка ответа нашла расхождения:\n- ' + '\n- '.join(lines)
    tail = ('\nНапиши заново, исправив только это: ответ и строки ОСНОВАНИЕ и РЕЖИМ.' if formatted
            else '\nНапиши ответ заново, исправив только это. Только текст ответа, без пояснений.')
    return head + tail


def handoff(issues):
    """Honest hand-over when a corrected answer still fails a hard check; '' when no hard check failed.

    Hard checks are the ones code can be sure of: the evidence belongs to another product, or the
    answer names a figure, contact or channel found neither in the documents nor in the dialogue."""
    other = next((detail for code, detail in issues if code == 'other_product'), None)
    if other:
        # Only the product the known data belong to is named: that one is read from the section itself,
        # while the product of the question is an inference.
        return ('Подтверждённых сведений по этому вопросу у меня нет: известные данные относятся к направлению «%s», '
                'и переносить их на ваш случай нельзя. Могу передать вопрос специалисту — он ответит точно по вашему объекту.' % other[0])
    if any(code in ('contact', 'channel', 'figure') for code, _ in issues):
        return ('Подтверждённых сведений по этому вопросу у меня нет, поэтому называть цифры, условия и контакты не буду. '
                'Могу передать вопрос специалисту — он ответит точно по вашему объекту.')
    return ''


def second_look(sources, included, search, question, limit):
    """Sections for a second look when the model found no answer: those found by an alternative
    wording of the request (the last client turn alone) and the next ranked ones."""
    present = {_key(s) for s in included}
    fresh = []

    def take(candidates, count):
        for source in candidates:
            if count <= 0:
                break
            if _key(source) not in present and not source.get('forced'):
                fresh.append(source)
                present.add(_key(source))
                count -= 1

    if search is not None:
        latest, _ = dialogue(question)
        try:
            # The last turn alone: earlier turns may have pulled the search towards another subject.
            lexical = getattr(search, 'lexical', search)
            take(lexical.find(latest, k=limit), limit // 2)
        except Exception:
            pass
    take(sources, limit - len(fresh))
    return fresh
