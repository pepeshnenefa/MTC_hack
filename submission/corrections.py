"""Conservative, stateless correction of user history for retrieval only.

No product facts, model calls, or changes to the generation dialogue.
Ambiguous and unsupported corrections retain the original search input.
"""
import re

MARKER = re.compile(r'\b(?:ошибся|ошиблась|ошиблись|исправляю|поправка|исправление)\b', re.I)
UNITS = {
    'objects': r'объект(?:а|ов)?',
    'meters': r'сч[её]тчик(?:а|ов)?',
    'sites': r'площад(?:ка|ки|ок)',
    'buildings': r'здани(?:е|я|й)',
    'shops': r'магазин(?:а|ов)?',
    'power': r'кВт',
    'energy': r'кВт\s*[⋅*·]\s*ч',
}
# Energy must precede power; a bare kW match must not consume kWh.
QUANTITY = re.compile(
    r'(?<![\w.,+\-])(?P<number>\d+(?:[.,]\d+)?)\s*'
    r'(?P<unit>' + '|'.join('(?P<' + key + '>' + UNITS[key] + ')'
                            for key in ('energy', 'objects', 'meters', 'sites',
                                        'buildings', 'shops', 'power')) + r')(?!\w)', re.I)
PAIR = re.compile(r'\bне\s+(?P<old>[^,;:.!?\n]{1,70}),\s*а\s+'
                  r'(?P<new>[^,;:.!?\n]{1,70})(?=$|[.;!?\n])', re.I)


def literal(value):
    return re.compile(r'(?<!\w)' + re.escape(value.strip()) + r'(?!\w)', re.I)


def unit_key(match):
    return next(key for key in UNITS if match.group(key) is not None)


def safe_statement(text):
    # Do not interpret alternatives, questions, ranges or planned changes as facts.
    return not re.search(r'\?|\b(?:если|допустим|например|возможно|планируем|будет|будут|'
                         r'станет|станут|вместо|только|либо|или|от|до)\b|\d\s*[-–—+]', text, re.I)


def clean_turns(turns):
    """Return a copy; only explicit, unambiguous replacements remove old spans."""
    cleaned = []
    changed = False
    for original in turns:
        current = original
        # Sentence boundaries retain decimal points. Correction and value must
        # occur in the same statement; later questions remain untouched.
        sentences = re.split(r'(?<=[.!?;])\s+(?!\d)', original)
        for sentence in sentences:
            marker = MARKER.search(sentence)
            if not marker or not cleaned or not safe_statement(sentence):
                continue
            body = sentence[marker.end():].strip(' :,-')
            pair = PAIR.search(body)
            if pair:
                old, new = pair.group('old').strip(), pair.group('new').strip()
                if (old.lower() == new.lower() or len(old.split()) > 5
                        or len(new.split()) > 5 or re.search(r'\b(?:не|нет|и|а)\b', old + ' ' + new, re.I)):
                    continue
                pattern = literal(old)
                hits = [(i, m) for i, turn in enumerate(cleaned) for m in pattern.finditer(turn)]
                # Multiple occurrences may denote different objects or contexts.
                if len(hits) != 1:
                    continue
                i, match = hits[0]
                if (not safe_statement(cleaned[i]) or
                        re.search(r'\b(?:не|без)\s*$', cleaned[i][:match.start()], re.I)):
                    continue
                cleaned[i] = pattern.sub('', cleaned[i])
                current = current.replace(pair.group(0), new, 1)
                changed = True
                continue
            values = list(QUANTITY.finditer(body))
            if len(values) != 1 or re.search(r'\b(?:не|нет|без)\b', body, re.I):
                continue
            replacement = values[0]
            # No inference across named branches, locations, owners or periods.
            # The bare-number shorthand is accepted only as an explicit correction.
            if not re.fullmatch(r'(?:(?:у нас|всего|есть|мощность|потребление)\s+)?',
                                body[:replacement.start()], re.I):
                continue
            if body[replacement.end():].strip(' .!'):
                continue
            key = unit_key(replacement)
            hits = [(i, m) for i, turn in enumerate(cleaned)
                    for m in QUANTITY.finditer(turn) if unit_key(m) == key]
            if len(hits) != 1:
                continue
            i, match = hits[0]
            prefix = re.split(r'[.;!\n]', cleaned[i][:match.start()])[-1].strip()
            previous_marker = MARKER.search(prefix)
            if previous_marker:
                prefix = prefix[previous_marker.end():].strip(' :,-')
            if not (previous_marker and not prefix) and not re.fullmatch(
                    r'(?:у нас|всего|есть|мощность|потребление)\s*', prefix, re.I):
                continue
            # Leave historical alternatives, negatives, ranges and questions alone.
            if not safe_statement(cleaned[i]) or re.search(r'\b(?:не|нет|без)\b', cleaned[i], re.I):
                continue
            if match.group('number').replace(',', '.') == replacement.group('number').replace(',', '.'):
                continue
            cleaned[i] = cleaned[i][:match.start()] + cleaned[i][match.end():]
            changed = True
        cleaned.append(current)
    return cleaned, changed


def search_text(text, parse_dialogue):
    latest, history = parse_dialogue(text)
    turns, changed = clean_turns([*history, latest])
    if not changed:
        return text
    return '\n'.join('Пользователь: ' + turn for turn in turns)
