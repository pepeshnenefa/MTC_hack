"""Offline retrieval metrics and weight search on the TUNING scenarios only.

Labels: tests/retrieval_cases.json (demo 1-18, p1-p8) and decision-v1 (source + heading).
The holdout set is evaluated only with --holdout and must never be used to pick weights.
This measures whether the needed section reaches the model, not answer quality.

    python3 tools/tune_retrieval.py report                 # ranks and context sizes at current weights
    python3 tools/tune_retrieval.py ablate                 # switch every component off, one at a time
    python3 tools/tune_retrieval.py search --rounds 2      # coordinate search over the weights
    python3 tools/tune_retrieval.py report --set NAME=VALUE ...
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
import config  # noqa: E402
from context import pack_context  # noqa: E402
from retrieval import Search, chunks  # noqa: E402

GRID = {
    'BM25_K1': [0.8, 1.2, 1.6, 2.0], 'BM25_B': [0.2, 0.35, 0.5, 0.75],
    'W_TITLE': [0, 0.2, 0.35, 0.6, 1.0], 'W_HISTORY': [0, 0.2, 0.4, 0.6, 0.8],
    'W_ASSISTANT': [0, 0.1, 0.2, 0.3, 0.4], 'W_ALIAS': [0, 0.2, 0.35, 0.6],
    'W_PREFIX': [0, 0.4, 0.65, 0.9], 'W_INPUTS': [0, 0.3, 0.6, 0.9],
    'B_INPUTS_TITLE': [0, 4, 8, 12], 'M_TOPIC_MATCH': [1.0, 1.2, 1.35, 1.6],
    'B_TOPIC_MATCH': [0, 2, 4, 6], 'M_TOPIC_MISMATCH': [0.2, 0.4, 0.6, 0.8, 1.0],
    'B_BODY_TOPIC': [0, 1, 2, 3], 'B_MULTI_TOPIC': [0, 2.5, 5, 8],
    'B_STAGED': [0, 4, 8, 12], 'B_SUITABILITY': [0, 2, 4, 6], 'B_SCOPE': [0, 0.9, 2, 4],
    'W_DENSE': [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    'W_COLLOQUIAL': [0, 0.2, 0.35, 0.5, 0.8],
    'M_DENSE_MISMATCH': [0.25, 0.5, 0.75, 1.0],
}
OFF = {'M_TOPIC_MATCH': 1.0, 'M_TOPIC_MISMATCH': 1.0}  # neutral value where 0 is not "off"


def documents():
    base = ROOT / 'Knowledge-base'
    return [(str(p.relative_to(base)), p.read_text(encoding='utf-8-sig')) for p in sorted(base.rglob('*.md'))]


def load_cases(holdout=False):
    cases = []
    if holdout:
        base = ROOT / 'tests/evaluation/holdout-v1'
        questions = {r['id']: r['question'] for r in csv.DictReader((base / 'questions.csv').open(encoding='utf-8-sig'))}
        for case in json.loads((base / 'cases.json').read_text(encoding='utf-8')):
            groups = [{'source': f['source'].split('/')[-1], 'headings': [f['heading']]} for f in case.get('key_facts', [])
                      if f.get('source') and f.get('heading')]
            if groups:
                cases.append({'id': case['id'], 'question': questions[case['id']], 'groups': groups})
        return cases
    demo = {r['id']: r['question'] for r in csv.DictReader((ROOT / 'demo_questions.csv').open(encoding='utf-8-sig'))}
    for case in json.loads((ROOT / 'tests/retrieval_cases.json').read_text(encoding='utf-8')):
        cases.append({'id': case['id'], 'question': case.get('question', demo.get(case['id'])), 'groups': case['groups']})
    for case in json.loads((ROOT / 'tests/evaluation/decision-v1/cases.json').read_text(encoding='utf-8'))['cases']:
        cases.append({'id': case['id'], 'question': case['question'],
                      'groups': [{'source': case['source'].split('/')[-1], 'headings': [case['heading']]}]})
    # Colloquial paraphrases written for tuning the vocabulary (direct evidence group only).
    for case in json.loads((ROOT / 'tests/retrieval_paraphrases.json').read_text(encoding='utf-8')):
        cases.append({'id': case['id'], 'question': case['question'], 'groups': case['groups']})
    return cases


_VECTORS = {}


def similarity(search, question):
    """Cached bge-m3 cosine similarities (tools/embed_cache.py); None when the dense channel is off."""
    if not config.get('USE_HYBRID') or config.get('W_DENSE') <= 0:
        return None
    import hashlib
    from retrieval import query
    if not _VECTORS:
        _VECTORS.update(json.loads((ROOT / 'work/emb/cache.json').read_text(encoding='utf-8')))
    digest = lambda text: hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]
    if 'documents' not in _VECTORS:
        _VECTORS['documents'] = [_VECTORS[digest(c.get('title', c['heading']) + '\n' + c['text'])] for c in search.items]
    key = 'sim:' + digest(query(question))
    if key not in _VECTORS:  # similarities do not depend on the weights being tuned
        q = _VECTORS[digest(query(question))]
        _VECTORS[key] = [sum(a * b for a, b in zip(q, v)) for v in _VECTORS['documents']]
    return _VECTORS[key]


def matches(source, group):
    return source['source'].replace('\\', '/').endswith(group['source']) and source['heading'] in group['headings']


def measure(search, cases, ks=(3, 4, 6, 8, 12)):
    """Rank of each needed group in the context actually packed for the model."""
    result = {'groups': sum(len(c['groups']) for c in cases), 'hit': {}, 'chars': {}, 'rr': 0.0, 'misses': {}}
    candidates = [search.find(case['question'], k=36, dense=similarity(search, case['question'])) for case in cases]
    for k in ks:
        hit = chars = 0
        misses = []
        for case, found in zip(cases, candidates):
            _, report = pack_context(case['question'], found, limit=k)
            included = report['included']
            chars += report['characters']
            for group in case['groups']:
                rank = next((i + 1 for i, s in enumerate(included) if matches(s, group)), None)
                hit += rank is not None
                if rank is None:
                    misses.append(case['id'])
                if k == max(ks):
                    result['rr'] += 1 / rank if rank else 0
        result['hit'][k] = hit
        result['chars'][k] = round(chars / len(cases))
        result['misses'][k] = misses
    result['rr'] = round(result['rr'] / result['groups'], 4)
    return result


def objective(metrics):
    # Evidence must reach a small context first; rank quality breaks ties.
    hit = metrics['hit']
    return (2 * hit[4] + 2 * hit[6] + hit[3] + hit[8] + hit[12]) / (7 * metrics['groups']) + 0.15 * metrics['rr']


def line(name, metrics):
    return (name.ljust(34) + ' '.join('@%d=%d' % (k, v) for k, v in metrics['hit'].items()) + '/%d' % metrics['groups'] +
            '  MRR=%.3f  obj=%.4f  chars@4/6/12=%d/%d/%d' % (metrics['rr'], objective(metrics), metrics['chars'][4],
                                                           metrics['chars'][6], metrics['chars'][12]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['report', 'ablate', 'search'])
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--holdout', action='store_true', help='final check only; never for choosing weights')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.holdout and args.command != 'report':
        raise SystemExit('The holdout set is for the final report only')
    overrides = dict(pair.split('=', 1) for pair in args.set)
    search = Search(chunks(documents()))
    cases = load_cases(args.holdout)
    with config.override(**overrides):
        base = measure(search, cases)
        print(line('current' + (' (holdout)' if args.holdout else ''), base))
        if args.verbose or args.command == 'report':
            for k, misses in base['misses'].items():
                print('  misses @%d: %s' % (k, ' '.join(misses) or '-'))
        if args.command == 'ablate':
            for name in GRID:
                with config.override(**{name: OFF.get(name, 0)}):
                    metrics = measure(search, cases)
                print(line('off ' + name, metrics) + '  Δobj=%+.4f' % (objective(metrics) - objective(base)))
        if args.command == 'search':
            best = dict(config.snapshot())
            best_score = objective(base)
            for round_number in range(args.rounds):
                improved = False
                for name, values in GRID.items():
                    for value in values:
                        if value == best[name]:
                            continue
                        with config.override(**{**{n: best[n] for n in GRID}, name: value}):
                            score = objective(measure(search, cases))
                        # Require a real gain; ties keep the earlier (confirmed) value.
                        if score > best_score + 0.003:
                            print('round %d: %s %s -> %s  obj %.4f -> %.4f' % (round_number + 1, name, best[name], value, best_score, score))
                            best[name] = value
                            best_score = score
                            improved = True
                if not improved:
                    break
            changed = {n: best[n] for n in GRID if best[n] != config.get(n)}
            with config.override(**{n: best[n] for n in GRID}):
                print(line('best', measure(search, cases)))
            print('changed weights:', json.dumps(changed, ensure_ascii=False))


if __name__ == '__main__':
    main()
