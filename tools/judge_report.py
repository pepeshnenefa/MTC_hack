"""Open the blind key after an independent review and summarise per system.

    python3 tools/judge_report.py work/eval/blind-holdout --baseline sample-baseline

Prints, per system: scenarios with every applicable criterion satisfied, per-criterion counts,
and a platform-style comparison against the baseline (Match, Win, Score = 0.7*Match + 0.3*Win).
The local reviewer is not the platform judge; these numbers are an estimate of direction only.
"""
import argparse
import json
from pathlib import Path

NAMES = ['техн', 'факты', 'ключ', 'оговор', 'прямой', 'уточн', 'диалог', 'отказ', 'форма', 'шум']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--baseline', default='sample-baseline')
    parser.add_argument('--grades', default='grades.json')
    parser.add_argument('--cases', help='cases.json with error_class/mirror for a per-class breakdown')
    args = parser.parse_args()
    base = Path(args.directory)
    key = json.loads((base / 'key.json').read_text(encoding='utf-8'))
    rows = json.loads((base / args.grades).read_text(encoding='utf-8'))
    systems = sorted({s for mapping in key.values() for s in mapping.values()})
    marks = {s: {} for s in systems}
    notes = {s: {} for s in systems}
    pairs = {}
    for row in rows:
        mapping = key[row['id']]
        for letter, grade in row['grades'].items():
            marks[mapping[letter]][row['id']] = grade['marks']
            notes[mapping[letter]][row['id']] = grade.get('note', '')
        for pair, value in row.get('pairs', {}).items():
            left, right = pair.split('-')
            pairs.setdefault((mapping[left], mapping[right]), {})[row['id']] = float(value)
            pairs.setdefault((mapping[right], mapping[left]), {})[row['id']] = 1 - float(value)
    for system in systems:
        graded = marks[system]
        clean = [i for i, m in graded.items() if '0' not in m]
        per = []
        for index, label in enumerate(NAMES):
            applicable = [m[index] for m in graded.values() if m[index] != '-']
            per.append('%s %d/%d' % (label, applicable.count('1'), len(applicable)))
        print('%s: без единой ошибки %d/%d | %s' % (system, len(clean), len(graded), ' | '.join(per)))
        print('   не выполнено: ' + ' '.join('%s[%s]' % (i, ''.join(str(k) for k, c in enumerate(m) if c == '0'))
                                              for i, m in sorted(graded.items()) if '0' in m))
    # Critical errors are marked by the reviewer in the note; counted, never guessed from marks.
    critical = {s: sorted(i for i, note in notes[s].items() if note.strip().upper().startswith('КРИТИЧНО')) for s in systems}
    for system in systems:
        print('%s: критических %d | %s' % (system, len(critical[system]), ' '.join(critical[system])))
    if args.cases:
        cases = {c['id']: c for c in json.loads(Path(args.cases).read_text(encoding='utf-8'))}
        groups = {}
        for case in cases.values():
            if 'error_class' in case:
                groups.setdefault((case['error_class'], bool(case.get('mirror'))), []).append(case['id'])
        for (name, mirror), ids in sorted(groups.items()):
            cells = []
            for system in systems:
                graded = [i for i in ids if i in marks[system]]
                clean = sum('0' not in marks[system][i] for i in graded)
                decision = sum(marks[system][i][2] == '1' for i in graded)
                cells.append('%s: вывод верен %d/%d, без ошибок %d, критических %d'
                             % (system, decision, len(graded), clean, sum(i in critical[system] for i in graded)))
            print('  %-22s %s | %s' % (name, 'зеркальные' if mirror else 'ловушки  ' if name != 'control' else 'контроль  ', ' | '.join(cells)))
    for system in systems:
        if system == args.baseline or (system, args.baseline) not in pairs:
            continue
        values = list(pairs[(system, args.baseline)].values())
        match = sum(values) / len(values)
        wins, losses = sum(v > 0.5 for v in values), sum(v < 0.5 for v in values)
        win = wins / (wins + losses) if wins + losses else 0.5
        print('%s против %s: Match %.3f, Win %.3f (%d побед, %d поражений, %d ничьих), Score %.3f'
              % (system, args.baseline, match, win, wins, losses, len(values) - wins - losses, 0.7 * match + 0.3 * win))
    others = [s for s in systems if s != args.baseline]
    for i, left in enumerate(others):
        for right in others[i + 1:]:
            if (left, right) in pairs:
                values = list(pairs[(left, right)].values())
                print('%s против %s: среднее %.3f, побед %d, поражений %d' % (left, right, sum(values) / len(values),
                                                                             sum(v > 0.5 for v in values), sum(v < 0.5 for v in values)))


if __name__ == '__main__':
    main()
