"""Replay the numeric engine's rules on a stored run, without the model.

    python3 tools/numeric_replay.py                                   # holdout-v4, run v4-numeric-r2
    python3 tools/numeric_replay.py --show X02 X35                    # values, rejections and note of single cases

The extractor's reply is taken from the run's trace; search and packing are recomputed (hybrid search,
embeddings from the proxy cache: start tools/local_proxy.py, no model server is needed), so the
document blocks are what the run had. Prints the stored and the new verdict, the expected one from
cases.json, and whether the note changed. Valid only while blocks() and the extraction prompt are
unchanged: otherwise the stored extraction no longer answers the request the new code would send.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import numeric_regression as nr  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', default=str(ROOT / 'work/eval/v4-numeric-r2/holdout4.trace.jsonl'))
    parser.add_argument('--questions', default=str(ROOT / 'tests/evaluation/holdout-v4/questions.csv'))
    parser.add_argument('--cases', default=str(ROOT / 'tests/evaluation/holdout-v4/cases.json'))
    parser.add_argument('--base-url', default='http://127.0.0.1:8090/v1')
    parser.add_argument('--show', nargs='*', default=[])
    args = parser.parse_args()
    args.model, args.scope, args.lexical, args.env = 'qwen3-4b-local', 'last', False, None
    nr.setup(args)
    import config
    import numeric_engine
    from context import pack_context
    cases = {c['id']: c for c in json.loads(Path(args.cases).read_text(encoding='utf-8'))}
    rows = list(csv.DictReader(open(args.questions, encoding='utf-8')))
    trace = {}
    for line in open(args.trace, encoding='utf-8'):
        record = json.loads(line)
        trace[record['question']] = record
    search = nr.build_search([r['question'] for r in rows])
    top_k = max(1, min(100, int(config.get('TOP_K'))))
    same = right = 0
    for row in rows:
        question = row['question']
        old = trace[question].get('numeric') or {}
        _, report = pack_context(question, search.find(question, k=min(100, max(36, top_k * 3))), limit=top_k)
        data = old.get('extraction')
        new = numeric_engine.run(question, report['included'], lambda messages: json.dumps(data, ensure_ascii=False) if data is not None else '')
        expect = (cases.get(row['id'], {}).get('numeric') or {}).get('verdict') or 'NO_CHECK'
        same += new['status'] == old.get('status')
        right += new['status'] == expect
        mark = ('' if new['status'] == old.get('status') else '  <- verdict changed') + ('' if new['note'] == old.get('note', '') else '  [note changed]')
        print('%s expect %-20s stored %-20s new %-20s%s' % (row['id'], expect, old.get('status'), new['status'], mark))
        if row['id'] in args.show:
            print('   values:', [(v['text'], v['value'], v['object']) for v in new['values']])
            print('   rejected:', new['rejected'])
            print('   note:', new['note'] or '—')
    print('verdict unchanged %d/%d, matches expected %d/%d' % (same, len(rows), right, len(rows)))


if __name__ == '__main__':
    main()
