"""Per-criterion grades of real answers and comparison between versions.

Grades are entered by a reviewer after reading every answer against the reference
(golden answer / required decision) and the knowledge base; nothing is scored by keywords.
File format (work/eval/<name>/grades.txt), one line per scenario:

    <id> <ten marks for criteria 0..9: 1, 0 or - (not applicable)> <note>

Criteria (fixed, see docs/unified-v1.md): 0 technical, 1 factual accuracy, 2 key facts,
3 caveats kept, 4 direct answer first, 5 clarifying question appropriate, 6 dialogue history,
7 correct refusal, 8 form and length, 9 no noise.

    python3 tools/grade.py summary base-prompt-v1
    python3 tools/grade.py compare base-prompt-v1 e1-topk5
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['техн', 'факты', 'ключ', 'оговор', 'прямой', 'уточн', 'диалог', 'отказ', 'форма', 'шум']


def load(name):
    grades = {}
    for line in (ROOT / 'work/eval' / name / 'grades.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split(None, 2)
        marks = parts[1]
        if len(marks) != 10 or set(marks) - set('10-'):
            raise ValueError('Bad marks: ' + line)
        grades[parts[0]] = (marks, parts[2] if len(parts) > 2 else '')
    return grades


def passed(marks):
    return '0' not in marks


def summary(name, grades=None):
    grades = grades or load(name)
    total = len(grades)
    ok = sum(passed(m) for m, _ in grades.values())
    per = []
    for i, label in enumerate(NAMES):
        applicable = [m[i] for m, _ in grades.values() if m[i] != '-']
        per.append('%s %d/%d' % (label, applicable.count('1'), len(applicable)))
    print('%s: сценариев без единой ошибки %d/%d | %s' % (name, ok, total, ' | '.join(per)))
    return ok, total


def compare(old, new):
    a, b = load(old), load(new)
    summary(old, a)
    summary(new, b)
    fixed = [i for i in b if i in a and not passed(a[i][0]) and passed(b[i][0])]
    broken = [i for i in b if i in a and passed(a[i][0]) and not passed(b[i][0])]
    print('исправлено:', ' '.join(fixed) or '-')
    print('регрессии:', ' '.join(broken) or '-')
    for i in b:
        if i in a and a[i][0] != b[i][0]:
            print('  %-6s %s -> %s  %s' % (i, a[i][0], b[i][0], b[i][1]))


if __name__ == '__main__':
    if sys.argv[1] == 'summary':
        for name in sys.argv[2:]:
            summary(name)
    elif sys.argv[1] == 'compare':
        compare(sys.argv[2], sys.argv[3])
