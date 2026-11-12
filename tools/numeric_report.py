"""Decision report of the numeric-engine-v1 check on holdout-v4, after a blind review.

    python3 tools/numeric_report.py work/eval/blind-holdout-v4 --baseline v4-hybrid --candidate v4-numeric-r2 \
        --cases tests/evaluation/holdout-v4/cases.json --set holdout4

Reads the reviewer's grades (tools/blind_pack.py layout), the run answers, traces and summaries, and
checks the five acceptance conditions written in docs/numeric-engine-v1.md before the run. Separately,
without influence on the decision: the engine verdict against the `numeric.verdict` field of the cases
and the time per question. Critical errors are the reviewer's notes starting with КРИТИЧНО; nothing is
guessed from the answer text. Development tool, never packaged.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from local_eval import SETS  # noqa: E402

NUMERIC = ('number_vs_threshold', 'number_needs_clarification')
IRRELEVANT = ('number_irrelevant',)
FALLBACK_START = 'К сожалению, сейчас не могу дать точный ответ'


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--set', default='holdout4')
    args = parser.parse_args()
    base = Path(args.directory)
    key = json.loads((base / 'key.json').read_text(encoding='utf-8'))
    rows = json.loads((base / 'grades.json').read_text(encoding='utf-8'))
    cases = {c['id']: c for c in json.loads(Path(args.cases).read_text(encoding='utf-8'))}
    systems = [args.baseline, args.candidate]
    grade = {s: {} for s in systems}
    pair = {}  # candidate relative to baseline: >0.5 a win of the candidate
    for row in rows:
        mapping = key[row['id']]
        for letter, value in row['grades'].items():
            grade[mapping[letter]][row['id']] = value
        for name, value in row.get('pairs', {}).items():
            left, right = name.split('-')
            value = float(value)
            pair[row['id']] = value if mapping[left] == args.candidate else 1 - value
    ids = sorted(cases)
    missing = [i for i in ids if i not in grade[args.baseline] or i not in grade[args.candidate] or i not in pair]
    if missing:
        raise SystemExit('Not graded: ' + ' '.join(missing))

    def critical(system, i):
        return grade[system][i].get('note', '').strip().upper().startswith('КРИТИЧНО')

    def decision(system, i):
        return grade[system][i]['marks'][2] == '1'

    groups = {
        'числовые (18)': [i for i in ids if cases[i]['error_class'] in NUMERIC],
        'число есть, вопрос о другом (8)': [i for i in ids if cases[i]['error_class'] in IRRELEVANT],
        'остальные классы и контроль (14)': [i for i in ids if cases[i]['error_class'] not in NUMERIC + IRRELEVANT],
        'все (40)': ids,
    }

    # Run artefacts: answers, engine records from the trace (matched to ids by the question text).
    questions = {r['question']: r['id'] for r in read_csv(SETS[args.set])}
    answers, trace, summary = {}, {}, {}
    for system in systems:
        out = ROOT / 'work/eval' / system
        answers[system] = {r['id']: r['answer'] for r in read_csv(out / (args.set + '.csv'))}
        trace[system] = {}
        for line in (out / (args.set + '.trace.jsonl')).read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            trace[system][questions[record['question']]] = record
        summary[system] = json.loads((out / (args.set + '.summary.json')).read_text(encoding='utf-8'))
    engine = {i: (trace[args.candidate].get(i) or {}).get('numeric') or {} for i in ids}
    status = {i: engine[i].get('status', 'нет записи') for i in ids}
    noted = {i for i in ids if engine[i].get('note')}

    print('| Группа | Сценариев | ' + ' | '.join('%s: критич. / верный вывод' % s for s in systems) + ' | Попарно: побед / поражений / ничьих кандидата |')
    print('|---|---:|' + '---|' * len(systems) + '---|')
    for title, group in groups.items():
        cells = ['%d / %d' % (sum(critical(s, i) for i in group), sum(decision(s, i) for i in group)) for s in systems]
        values = [pair[i] for i in group]
        print('| %s | %d | %s | %d / %d / %d |' % (title, len(group), ' | '.join(cells), sum(v > .5 for v in values),
                                                  sum(v < .5 for v in values), sum(v == .5 for v in values)))
    print()
    for title, group in groups.items():
        if title.startswith('все'):
            continue
        print('%s:' % title)
        for s in systems:
            print('   %s критические: %s' % (s, ' '.join(i for i in group if critical(s, i)) or '—'))
        print('   кандидат лучше: %s; хуже: %s' % (' '.join(i for i in group if pair[i] > .5) or '—',
                                                  ' '.join(i for i in group if pair[i] < .5) or '—'))
    print()

    # The five conditions, exactly as written in docs/numeric-engine-v1.md.
    num, irr, rest = groups['числовые (18)'], groups['число есть, вопрос о другом (8)'], groups['остальные классы и контроль (14)']
    crit = {s: {g: sum(critical(s, i) for i in group) for g, group in (('num', num), ('irr', irr), ('rest', rest))} for s in systems}
    dec = {s: sum(decision(s, i) for i in num) for s in systems}
    wins, losses = sum(pair[i] > .5 for i in ids), sum(pair[i] < .5 for i in ids)
    new_critical_irr = [i for i in irr if critical(args.candidate, i) and not critical(args.baseline, i)]
    noted_irr = [i for i in irr if i in noted]
    stubs = {s: [i for i in ids if not answers[s].get(i, '').strip() or answers[s][i].startswith(FALLBACK_START)] for s in systems}
    skipped = [i for i in ids if 'run time' in engine[i].get('reason', '')]
    no_record = [i for i in ids if i not in trace[args.candidate]]
    c, b = args.candidate, args.baseline
    conditions = [
        ('1. Числовые 18: критических меньше не менее чем на 2, верных выводов не меньше',
         crit[c]['num'] <= crit[b]['num'] - 2 and dec[c] >= dec[b],
         'критических %d против %d, верный вывод %d против %d' % (crit[c]['num'], crit[b]['num'], dec[c], dec[b])),
        ('2. Попарно по 40: побед больше поражений не менее чем на 3', wins - losses >= 3,
         'побед %d, поражений %d, ничьих %d' % (wins, losses, len(ids) - wins - losses)),
        ('3. «Вопрос о другом» 8: новых критических нет, пометка не более чем в одном',
         not new_critical_irr and len(noted_irr) <= 1,
         'новые критические: %s; пометка: %s' % (' '.join(new_critical_irr) or 'нет', ' '.join(noted_irr) or 'нет')),
        ('4. Остальные 14: критических не больше', crit[c]['rest'] <= crit[b]['rest'],
         'критических %d против %d' % (crit[c]['rest'], crit[b]['rest'])),
        ('5. Технически: 40 ответов, заглушек нет, движок не пропущен из-за времени',
         len(answers[c]) == len(ids) and not stubs[c] and not skipped and not no_record and summary[c]['returncode'] == 0,
         'ответов %d, заглушек %d, пропусков по времени %d, без записи в трассе %d, код возврата %s'
         % (len(answers[c]), len(stubs[c]), len(skipped), len(no_record), summary[c]['returncode'])),
    ]
    print('| Условие | Выполнено | Факт |')
    print('|---|---|---|')
    for title, ok, fact in conditions:
        print('| %s | %s | %s |' % (title, 'да' if ok else 'нет', fact))
    print('Итог: %s' % ('все условия выполнены' if all(ok for _, ok, _ in conditions) else 'не выполнены: ' +
                         ', '.join(t.split('.')[0] for t, ok, _ in conditions if not ok)))
    print('Заглушки у основной версии: %s' % (' '.join(stubs[b]) or 'нет'))
    print()

    # Separately, without influence on the decision.
    expected = {i: cases[i]['numeric']['verdict'] for i in ids}
    labels = ['PASS', 'FAIL', 'NEEDS_CLARIFICATION', 'NO_CHECK']
    print('Вердикт движка против поля numeric.verdict (строки — эталон, столбцы — движок):')
    print('| Эталон \\ движок | ' + ' | '.join(labels) + ' | Всего |')
    print('|---|' + '---:|' * (len(labels) + 1))
    for want in labels:
        group = [i for i in ids if expected[i] == want]
        if group:
            print('| %s | %s | %d |' % (want, ' | '.join(str(sum(status[i] == got for i in group)) for got in labels), len(group)))
    exact = [i for i in ids if status[i] == expected[i]]
    checked = [i for i in ids if expected[i] != 'NO_CHECK']
    print('Совпало %d из %d; из %d, где проверка нужна, верный вердикт %d, неверный вердикт %d, молчал %d; '
          'ложных срабатываний (эталон NO_CHECK, движок дал вердикт) %d из %d.'
          % (len(exact), len(ids), len(checked), sum(status[i] == expected[i] for i in checked),
             sum(status[i] not in (expected[i], 'NO_CHECK') for i in checked), sum(status[i] == 'NO_CHECK' for i in checked),
             sum(status[i] != 'NO_CHECK' for i in ids if expected[i] == 'NO_CHECK'), sum(expected[i] == 'NO_CHECK' for i in ids)))
    print('Расхождения: ' + ' '.join('%s[%s→%s]' % (i, expected[i], status[i]) for i in ids if status[i] != expected[i]))
    print('Пометка появилась: %d из 40 (%s)' % (len(noted), ' '.join(sorted(noted))))
    print()

    print('| Время | ' + ' | '.join(systems) + ' |')
    print('|---|' + '---:|' * len(systems))
    print('| Весь прогон, с | ' + ' | '.join('%.0f (частей %s)' % (summary[s]['seconds'], summary[s].get('parts', 1)) for s in systems) + ' |')
    print('| На вопрос, с | ' + ' | '.join('%.1f' % (summary[s]['seconds'] / len(ids)) for s in systems) + ' |')
    gated = [i for i in ids if engine[i].get('calls')]  # the gate passed and the extraction call was made
    extra = [engine[i]['seconds'] for i in gated]
    print('| Вызов извлечения, с (вопросов с вызовом) | — | %s |'
          % ('%.1f в среднем, %.1f максимум (%d)' % (sum(extra) / len(extra), max(extra), len(extra)) if extra else '—'))
    calls = {s: sum(trace[s][i]['calls'] for i in ids if i in trace[s]) for s in systems}
    calls[c] += sum(engine[i].get('calls', 0) for i in ids)
    print('| Вызовов модели на 40 вопросов | ' + ' | '.join(str(calls[s]) for s in systems) + ' |')


if __name__ == '__main__':
    main()
