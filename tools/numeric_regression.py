"""Numeric regression: the numeric validation engine against the existing number check on real cases.

    python3 tools/numeric_regression.py --name dev1
    python3 tools/numeric_regression.py --name dev1 --show H01 W11        # extraction, verdict and note of single cases

Every case goes through the real pipeline up to the answering call: the search and context packing of
the best archive (hybrid search, the same sections the model would receive), then
  * the engine: one real extraction call to the local model, validation and comparison by code;
  * the existing check (checks.number_check, last-turn scope): deterministic, no model.
Both are scored against the expectation stored with the case (tests/evaluation/numeric-regression-v1):
PASS / FAIL / NEEDS_CLARIFICATION / NO_CHECK, the bound and the side of it the client's value falls on.
The existing check has no verdict, so it is scored leniently: it is right when it stays silent where no
check is needed, or when its line holds the client's value and the expected bound on the correct side.

No answer is generated here: this measures the comparison, not the final wording (that is local_eval).
Extraction replies are cached in work/numeric/ so that validation code can be re-run without the model.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))


def setup(args):
    os.environ.update({'LLM_BASE_URL': args.base_url, 'LLM_API_KEY': 'local-test-key', 'LLM_MODEL': args.model,
                       'TEMPERATURE': '0', 'MAX_TOKENS': '700', 'ET_NUMERIC_ENGINE': '1', 'ET_NUMERIC_ENGINE_SCOPE': args.scope,
                       'ET_NUMBER_CHECK_SCOPE': 'last'})
    if not args.lexical:
        os.environ.update({'ET_USE_HYBRID': '1', 'ET_W_COLLOQUIAL': '0.5'})
    for pair in args.env or []:
        key, _, value = pair.partition('=')
        os.environ[key] = value
    import config
    config.reload()


def build_search(questions):
    import config
    import solution
    from retrieval import Search, chunks
    search = Search(chunks(solution.documents(ROOT / 'Knowledge-base')))
    if config.get('USE_HYBRID'):
        api = solution.API(deadline=time.monotonic() + 600)
        search = solution.prepare_hybrid(search, questions, api, 'baai/bge-m3', time.monotonic() + 600, anchor_k=int(config.get('TOP_K')))
    return search


def close(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


def score_engine(result, expect):
    status_ok = result['status'] in expect['status']
    if not status_ok or result['status'] not in ('PASS', 'FAIL') or expect.get('limit') is None:
        return status_ok, None
    primary = False
    for check in result['checks']:
        limit = check['limit']
        if not (close(limit['low'], expect['limit']) or close(limit['high'], expect['limit'])):
            continue
        relation_ok = expect.get('relation') is None or check['relation'] == expect['relation'] or \
            (expect['relation'] == 'inside' and check['relation'] in ('inside', 'equal'))
        client_ok = expect.get('client') is None or close(check.get('client'), expect['client'])
        primary = primary or (relation_ok and client_ok)
    return status_ok, primary


def score_old(lines, expect):
    """The existing check prints comparisons, not a verdict: lenient scoring, see the module docstring."""
    import checks
    if not lines:
        return 'NO_CHECK' in expect['status']
    if expect['status'] == ['NO_CHECK'] or expect.get('limit') is None or expect.get('relation') is None:
        return False
    dimension = expect.get('dimension', 'power')
    if dimension not in ('power', 'energy'):
        return False
    side = {'below': 'МЕНЬШЕ, чем', 'above': 'БОЛЬШЕ, чем', 'inside': 'БОЛЬШЕ, чем'}[expect['relation']]
    for text in lines:
        head, _, tail = text.partition(' — это ')
        for period in (None, 'month', 'year'):
            client = checks.show(dimension, expect['client'], period)
            bound = checks.show(dimension, expect['limit'], period)
            segment = next((part for part in tail.split('; ') if part.startswith(side)), '')
            if client in head and bound in segment:
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--cases', default=str(ROOT / 'tests/evaluation/numeric-regression-v1/cases.json'))
    parser.add_argument('--scope', default='last', choices=['last', 'dialog'])
    parser.add_argument('--base-url', default='http://127.0.0.1:8090/v1')
    parser.add_argument('--model', default='qwen3-4b-local')
    parser.add_argument('--lexical', action='store_true', help='lexical search instead of the hybrid one')
    parser.add_argument('--env', action='append')
    parser.add_argument('--only', nargs='*', help='case ids')
    parser.add_argument('--show', nargs='*', help='print extraction, verdict and note of these ids from a stored run')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'work/numeric'
    out.mkdir(parents=True, exist_ok=True)
    target = out / (args.name + '.json')
    if args.show is not None:
        stored = {r['id']: r for r in json.loads(target.read_text(encoding='utf-8'))['rows']}
        for i in args.show or stored:
            row = stored[i]
            print('### %s expect %s -> engine %s%s | old check %s' % (i, '/'.join(row['expect']['status']), row['engine']['status'],
                                                                     '' if row['engine_correct'] else '  ✗', 'ok' if row['old_correct'] else '✗'))
            print('Q:', row['question'].replace('\n', ' // '))
            print('extraction:', json.dumps(row['engine'].get('extraction'), ensure_ascii=False))
            print('rejected:', json.dumps(row['engine'].get('rejected'), ensure_ascii=False), '| reason:', row['engine'].get('reason'))
            print('note:', row['engine'].get('note') or '—')
            print('old lines:', row['old_lines'] or '—')
            print()
        return
    setup(args)
    import checks
    import config
    import numeric_engine
    import solution
    from context import pack_context
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    if args.only:
        cases = [c for c in cases if c['id'] in args.only]
    search = build_search([c['question'] for c in cases])
    api = solution.API()
    cache_path = out / 'extraction-cache.json'
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() and not args.no_cache else {}
    top_k = max(1, min(100, int(config.get('TOP_K'))))
    rows = []
    for case in cases:
        question = case['question']
        sources = search.find(question, k=min(100, max(36, top_k * 3)))
        _, report = pack_context(question, sources, limit=top_k)
        included = report['included']
        timing = {'seconds': 0.0, 'cached': True}

        def llm(messages):
            key = hashlib.sha256((args.model + json.dumps(messages, ensure_ascii=False)).encode('utf-8')).hexdigest()
            if key not in cache:
                started = time.monotonic()
                body = {'model': api.model, 'temperature': 0, 'max_tokens': int(config.get('NUMERIC_ENGINE_TOKENS')), 'messages': messages}
                cache[key] = api.post('/chat/completions', body, attempts=2)['choices'][0]['message'].get('content')
                timing.update(seconds=round(time.monotonic() - started, 1), cached=False)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
            return cache[key]
        engine = numeric_engine.run(question, included, llm)
        lines = checks.number_check(question, included)
        status_ok, primary = score_engine(engine, case['numeric'])
        row = {'id': case['id'], 'kind': case['kind'], 'origin': case['origin'], 'question': question, 'expect': case['numeric'],
               'engine': engine, 'engine_status_ok': status_ok, 'engine_primary_ok': primary,
               'engine_correct': bool(status_ok and primary is not False), 'old_lines': lines,
               'old_correct': score_old(lines, case['numeric']), 'blocks': len(numeric_engine.blocks(included, numeric_engine.gate(question, args.scope))),
               'seconds': timing['seconds'], 'cached': timing['cached']}
        rows.append(row)
        print('%-7s %-8s expect %-28s engine %-20s %s %s | old check %s | %ss' % (
            row['id'], row['kind'], '/'.join(case['numeric']['status']), engine['status'],
            'ok ' if row['engine_correct'] else '✗  ', '' if primary is not False else '(bound or side differs)',
            'ok' if row['old_correct'] else '✗ ', row['seconds']), flush=True)

    def share(items, key):
        return '%d/%d' % (sum(bool(r[key]) for r in items), len(items))
    groups = [('проверка нужна', [r for r in rows if r['kind'] == 'check']), ('проверка не нужна', [r for r in rows if r['kind'] == 'no_check']), ('всего', rows)]
    summary = {name: {'engine': share(items, 'engine_correct'), 'old_check': share(items, 'old_correct')} for name, items in groups}
    summary['engine_calls'] = sum(r['engine'].get('calls', 0) for r in rows)
    fresh = [r['seconds'] for r in rows if not r['cached']]
    summary['seconds_per_fresh_call'] = round(sum(fresh) / len(fresh), 1) if fresh else None
    summary['false_alarms'] = {'engine': [r['id'] for r in rows if r['kind'] == 'no_check' and r['engine']['status'] != 'NO_CHECK'],
                               'old_check': [r['id'] for r in rows if r['kind'] == 'no_check' and r['old_lines']]}
    summary['wrong_verdicts'] = [r['id'] for r in rows if r['kind'] == 'check' and r['engine']['status'] in ('PASS', 'FAIL', 'NEEDS_CLARIFICATION') and not r['engine_correct']]
    summary['missed'] = [r['id'] for r in rows if r['kind'] == 'check' and r['engine']['status'] == 'NO_CHECK' and not r['engine_correct']]
    target.write_text(json.dumps({'name': args.name, 'scope': args.scope, 'model': args.model, 'summary': summary, 'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
