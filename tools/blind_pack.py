"""Prepare a blind comparison file for an independent reviewer.

    python3 tools/blind_pack.py --set holdout --systems sample-baseline base-prompt-v1 final --out work/eval/blind-holdout

Writes <out>/answers.json (per question: three answers under random letters) and <out>/key.json
(letter -> system). The reviewer gets answers.json only; key.json is opened after grading.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from local_eval import SETS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', required=True)
    parser.add_argument('--systems', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=20261111)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    questions = list(csv.DictReader(SETS[args.set].open(encoding='utf-8-sig')))
    answers = {}
    for system in args.systems:
        path = ROOT / 'work/eval' / system / (args.set + '.csv')
        answers[system] = {r['id']: r['answer'] for r in csv.DictReader(path.open(encoding='utf-8-sig'))}
    blind, key = [], {}
    letters = [chr(ord('A') + i) for i in range(len(args.systems))]
    for row in questions:
        order = args.systems[:]
        rng.shuffle(order)
        key[row['id']] = dict(zip(letters, order))
        blind.append({'id': row['id'], 'question': row['question'],
                      'answers': {letter: answers[system][row['id']] for letter, system in zip(letters, order)}})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'answers.json').write_text(json.dumps(blind, ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'key.json').write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding='utf-8')
    print('questions', len(blind), 'systems', len(args.systems))


if __name__ == '__main__':
    main()
