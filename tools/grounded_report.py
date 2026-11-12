"""Comparison of systems on the three failures of the grounded-v1 iteration, after a blind review.

    python3 tools/grounded_report.py work/eval/blind-grounded-v1 --baseline g1-hybrid \
        --cases tests/evaluation/grounded-holdout-v1/cases.json --set grounded1

Reads the reviewer's grades (tools/blind_pack.py layout) and the run traces. Per system it prints:
accuracy (answers without a factual error), hallucinations, cross-product transfers, unjustified
refusals, completeness, critical errors by scenario class, the pairwise result against the baseline,
model calls and time. Tags ([ВЫДУМКА], [ПЕРЕНОС], [ОТКАЗ], [НЕПОЛНО], [НЕВЕРНЫЙ ВЫВОД]) are written by
the reviewer in the note; nothing is guessed from the answer text. Development tool, never packaged.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAGS = ['ВЫДУМКА', 'ПЕРЕНОС', 'ОТКАЗ', 'НЕПОЛНО', 'НЕВЕРНЫЙ ВЫВОД']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--set', default='grounded1')
    args = parser.parse_args()
    base = Path(args.directory)
    key = json.loads((base / 'key.json').read_text(encoding='utf-8'))
    rows = json.loads((base / 'grades.json').read_text(encoding='utf-8'))
    cases = {c['id']: c for c in json.loads(Path(args.cases).read_text(encoding='utf-8'))}
    systems = sorted({s for mapping in key.values() for s in mapping.values()}, key=lambda s: (s != args.baseline, s))
    grade = {s: {} for s in systems}
    pairs = {}
    for row in rows:
        mapping = key[row['id']]
        for letter, value in row['grades'].items():
            grade[mapping[letter]][row['id']] = value
        for pair, value in row.get('pairs', {}).items():
            left, right = pair.split('-')
            pairs.setdefault((mapping[left], mapping[right]), {})[row['id']] = float(value)
            pairs.setdefault((mapping[right], mapping[left]), {})[row['id']] = 1 - float(value)

    def group(case):
        name = case.get('error_class', 'control')
        if name == 'control':
            return 'контрольные'
        return 'зеркальные' if case.get('mirror') else 'ловушки'

    def critical(value):
        return value.get('note', '').strip().upper().startswith('КРИТИЧНО')

    def tagged(value, tag):
        return '[' + tag + ']' in value.get('note', '').upper()

    def rate(system, index, ids):
        marks = [grade[system][i]['marks'][index] for i in ids if grade[system][i]['marks'][index] != '-']
        return '%d/%d' % (marks.count('1'), len(marks))

    ids = sorted(cases)
    print('| Показатель | ' + ' | '.join(systems) + ' |')
    print('|---|' + '---:|' * len(systems))
    lines = [
        ('Точность: ответов без фактической ошибки (критерий 1)', lambda s: rate(s, 1, ids)),
        ('Верный вывод и ключевые факты (критерий 2)', lambda s: rate(s, 2, ids)),
        ('Оговорки сохранены (критерий 3)', lambda s: rate(s, 3, ids)),
        ('Прямой ответ (критерий 4)', lambda s: rate(s, 4, ids)),
        ('Корректный отказ там, где сведений нет (критерий 7)', lambda s: rate(s, 7, ids)),
        ('Ответов без единого замечания', lambda s: '%d/%d' % (sum('0' not in grade[s][i]['marks'] for i in ids), len(ids))),
        ('Критических ошибок', lambda s: str(sum(critical(grade[s][i]) for i in ids))),
    ]
    lines += [('Метка [%s]' % tag, (lambda tag: lambda s: str(sum(tagged(grade[s][i], tag) for i in ids)))(tag)) for tag in TAGS]
    for title, cell in lines:
        print('| ' + title + ' | ' + ' | '.join(cell(s) for s in systems) + ' |')

    print()
    print('| Класс | Сценариев | ' + ' | '.join(s + ': критич. / верный вывод / выдумка / перенос / отказ' for s in systems) + ' |')
    print('|---|---:|' + '---|' * len(systems))
    classes = {}
    for i in ids:
        classes.setdefault((cases[i].get('error_class', 'control'), bool(cases[i].get('mirror'))), []).append(i)
    order = sorted(classes, key=lambda k: (k[0] == 'control', k[0], k[1]))
    blocks = [((name, mirror), classes[(name, mirror)]) for name, mirror in order]
    for label in ('ловушки', 'зеркальные', 'контрольные'):
        blocks.append((('ВСЕГО ' + label, None), [i for i in ids if group(cases[i]) == label]))
    for (name, mirror), members in blocks:
        title = name if mirror is None else name + (' (зеркальные)' if mirror else '' if name == 'control' else ' (ловушки)')
        cells = []
        for s in systems:
            cells.append('%d / %s / %d / %d / %d' % (
                sum(critical(grade[s][i]) for i in members), rate(s, 2, members),
                sum(tagged(grade[s][i], 'ВЫДУМКА') for i in members), sum(tagged(grade[s][i], 'ПЕРЕНОС') for i in members),
                sum(tagged(grade[s][i], 'ОТКАЗ') for i in members)))
        print('| ' + title + ' | ' + str(len(members)) + ' | ' + ' | '.join(cells) + ' |')

    print()
    for s in systems:
        if s == args.baseline or (s, args.baseline) not in pairs:
            continue
        values = pairs[(s, args.baseline)]
        wins = sorted(i for i, v in values.items() if v > 0.5)
        losses = sorted(i for i, v in values.items() if v < 0.5)
        print('%s против %s: побед %d, поражений %d, равных %d, среднее %.3f' % (
            s, args.baseline, len(wins), len(losses), len(values) - len(wins) - len(losses), sum(values.values()) / len(values)))
        print('   победы: ' + ' '.join(wins))
        print('   поражения: ' + ' '.join(losses))
        for label in ('ловушки', 'зеркальные', 'контрольные'):
            members = [i for i in values if group(cases[i]) == label]
            print('   %s: побед %d, поражений %d' % (label, sum(values[i] > 0.5 for i in members), sum(values[i] < 0.5 for i in members)))

    print()
    print('| Система | Вызовов модели | Вопросов с повторным вызовом | Среднее время, с | Исправлено проверкой | Передано специалисту кодом | Повторный взгляд помог | Служебных строк в ответах |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for s in systems:
        trace = ROOT / 'work/eval' / s / (args.set + '.trace.jsonl')
        items = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()] if trace.exists() else []
        info = [item.get('grounding') or {} for item in items]
        leaked = sum(any(word in item.get('answer', '') for word in ('ОСНОВАНИЕ:', 'РЕЖИМ:')) for item in items)
        print('| %s | %d | %d | %.1f | %d | %d | %d | %d |' % (
            s, sum(i.get('calls', 0) for i in items), sum(i.get('calls', 0) > 1 for i in items),
            sum(i.get('seconds', 0) for i in items) / max(1, len(items)),
            sum(bool(g.get('corrected')) for g in info), sum(bool(g.get('handoff')) for g in info),
            sum(bool(g.get('rechecked')) for g in info), leaked))


if __name__ == '__main__':
    main()
