"""Run a solution directory end-to-end against a local OpenAI-compatible model server.

Real generation, not a mock: the unmodified entry point (solution.py) is started as on the
platform, only the proxy address points to a local model. The local model is a stand-in for
the competition model; its answers are evidence about the pipeline, not a platform score.

    python3 tools/local_eval.py run --name prompt-v1 --solution submission --set demo
    python3 tools/local_eval.py show --name prompt-v1 --set demo

Results are written to work/eval/<name>/ (ignored by Git, never packaged).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = {
    'demo': ROOT / 'demo_questions.csv',
    'decision': ROOT / 'tests/evaluation/decision-v1/questions.csv',
    'holdout': ROOT / 'tests/evaluation/holdout-v1/questions.csv',
    'holdout2': ROOT / 'tests/evaluation/holdout-v2/questions.csv',
    'holdout3': ROOT / 'tests/evaluation/holdout-v3/questions.csv',
    'holdout4': ROOT / 'tests/evaluation/holdout-v4/questions.csv',
    'grounded1': ROOT / 'tests/evaluation/grounded-holdout-v1/questions.csv',
    'numreg': ROOT / 'tests/evaluation/numeric-regression-v1/questions.csv',
}
PASS_THROUGH = ('PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def run(args):
    out = ROOT / 'work/eval' / args.name
    out.mkdir(parents=True, exist_ok=True)
    solution = Path(args.solution)
    if solution.suffix == '.zip':  # Evaluate a stored archive exactly as built.
        target = out / 'code'
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(solution) as archive:
            archive.extractall(target)
        solution = target
    questions = SETS.get(args.set, Path(args.set)).resolve()
    label = args.set if args.set in SETS else Path(args.set).stem
    if args.parts > 1:
        return run_parts(args, out, solution, questions, label)
    answers = out / (label + '.csv')
    trace = out / (label + '.trace.jsonl')
    if trace.exists():
        trace.unlink()
    env = {k: os.environ[k] for k in PASS_THROUGH if k in os.environ}
    env.update({'LLM_BASE_URL': args.base_url, 'LLM_API_KEY': 'local-test-key',
                'LLM_MODEL': args.model, 'MAX_WORKERS': str(args.workers), 'TEMPERATURE': '0',
                'MAX_TOKENS': str(args.max_tokens), 'TRACE_FILE': str(trace), 'PYTHONDONTWRITEBYTECODE': '1'})
    for pair in args.env or []:
        key, _, value = pair.partition('=')
        env[key] = value
    started = time.monotonic()
    process = subprocess.run([sys.executable, str(Path(solution).resolve() / 'solution.py'), '--questions', str(questions),
                              '--kb', str(ROOT / 'Knowledge-base'), '--output', str(answers)],
                             env=env, capture_output=True, text=True, cwd=out)
    seconds = round(time.monotonic() - started, 1)
    (out / (label + '.stderr.log')).write_text(process.stderr, encoding='utf-8')
    events = [line for line in process.stderr.splitlines() if line.startswith('{')]
    summary = {'name': args.name, 'set': label, 'returncode': process.returncode, 'seconds': seconds,
               'model': args.model, 'overrides': args.env or [], 'stdout': process.stdout.strip()[-400:],
               'regenerations': sum('"regeneration"' in e for e in events)}
    (out / (label + '.summary.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


def run_parts(args, out, solution, questions, label):
    """The local model is slow: a long set is run as several independent launches and merged.

    Every launch is the unmodified entry point with its own run budget; rows are independent
    test cases, so the merged file equals one launch on a faster server."""
    rows = read_csv(questions)
    size = -(-len(rows) // args.parts)
    merged, traces, summaries = [], [], []
    for index in range(args.parts):
        part = rows[index * size:(index + 1) * size]
        if not part:
            continue
        name = '%s.part%d' % (label, index + 1)
        source = out / 'parts' / (name + '.csv')  # its stem names the answers file in the run directory
        source.parent.mkdir(exist_ok=True)
        with source.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, ['id', 'question'])
            writer.writeheader()
            writer.writerows(part)
        single = argparse.Namespace(**dict(vars(args), parts=1, set=str(source), solution=str(solution)))
        run(single)
        stem = source.stem
        merged.extend(read_csv(out / (stem + '.csv')))
        trace = out / (stem + '.trace.jsonl')
        if trace.exists():
            traces.append(trace.read_text(encoding='utf-8'))
        summaries.append(json.loads((out / (stem + '.summary.json')).read_text(encoding='utf-8')))
    if [r['id'] for r in merged] != [r['id'] for r in rows]:
        raise SystemExit('Merged answers do not match the question ids')
    with (out / (label + '.csv')).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, ['id', 'answer'])
        writer.writeheader()
        writer.writerows(merged)
    (out / (label + '.trace.jsonl')).write_text(''.join(traces), encoding='utf-8')
    summary = {'name': args.name, 'set': label, 'parts': len(summaries),
               'returncode': max(s['returncode'] for s in summaries), 'seconds': round(sum(s['seconds'] for s in summaries), 1),
               'model': args.model, 'overrides': args.env or [], 'regenerations': sum(s['regenerations'] for s in summaries)}
    (out / (label + '.summary.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


def sample(args):
    """Organizers' minimal example (sample_submission/solution.py), unmodified, as the comparison baseline."""
    out = ROOT / 'work/eval' / args.name
    label = args.set if args.set in SETS else Path(args.set).stem
    data = out / ('data-' + label)
    data.mkdir(parents=True, exist_ok=True)
    (data / 'questions.csv').write_bytes(SETS.get(args.set, Path(args.set)).resolve().read_bytes())
    with zipfile.ZipFile(data / 'knowledge-base.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        base = ROOT / 'Knowledge-base'
        for path in sorted(base.rglob('*.md')):
            archive.write(path, str(path.relative_to(base)))
    env = {k: os.environ[k] for k in PASS_THROUGH if k in os.environ}
    env.update({'LLM_BASE_URL': args.base_url, 'LLM_API_KEY': 'local-test-key', 'LLM_MODEL': args.model,
                'MAX_WORKERS': '1', 'TEMPERATURE': '0', 'MAX_TOKENS': str(args.max_tokens), 'DATA_DIR': str(data)})
    started = time.monotonic()
    process = subprocess.run([str(ROOT / 'work/.venv/bin/python'), str(ROOT / 'sample_submission/solution.py')],
                             env=env, capture_output=True, text=True, cwd=data)
    (out / (label + '.stderr.log')).write_text(process.stderr, encoding='utf-8')
    produced = data / 'submission.csv'
    if produced.exists():
        (out / (label + '.csv')).write_bytes(produced.read_bytes())
    print(json.dumps({'name': args.name, 'set': label, 'returncode': process.returncode,
                      'seconds': round(time.monotonic() - started, 1), 'stdout': process.stdout.strip()[-200:]}, ensure_ascii=False))


def show(args):
    out = ROOT / 'work/eval' / args.name
    label = args.set if args.set in SETS else Path(args.set).stem
    questions = {r['id']: r['question'] for r in read_csv(SETS.get(args.set, Path(args.set)))}
    traces = {}
    path = out / (label + '.trace.jsonl')
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            traces[item['question']] = item
    for row in read_csv(out / (label + '.csv')):
        question = questions[row['id']]
        trace = traces.get(question, {})
        print('### ' + row['id'] + ' [' + str(len(row['answer'].split())) + ' сл.' +
              (', ' + str(trace.get('context_chars')) + ' симв. контекста, ' + str(trace.get('calls')) + ' выз., ' +
               str(trace.get('seconds')) + ' с' if trace else '') + ']')
        if args.questions:
            print('Q: ' + question.replace('\n', ' // '))
        print('A: ' + row['answer'])
        if args.sources and trace:
            for source in trace.get('sources', []):
                print('   - ' + source)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--name', required=True)
    p.add_argument('--solution', default=str(ROOT / 'submission'))
    p.add_argument('--set', default='demo')
    p.add_argument('--base-url', default='http://127.0.0.1:8090/v1')
    p.add_argument('--model', default='qwen3-4b-local')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--max-tokens', type=int, default=700)
    p.add_argument('--env', action='append')
    p.add_argument('--parts', type=int, default=1, help='split a long set into several launches and merge the answers')
    p.set_defaults(func=run)
    p = sub.add_parser('sample')
    p.add_argument('--name', default='sample-baseline')
    p.add_argument('--set', default='demo')
    p.add_argument('--base-url', default='http://127.0.0.1:8090/v1')
    p.add_argument('--model', default='qwen3-4b-local')
    p.add_argument('--max-tokens', type=int, default=700)
    p.set_defaults(func=sample)
    p = sub.add_parser('show')
    p.add_argument('--name', required=True)
    p.add_argument('--set', default='demo')
    p.add_argument('--questions', action='store_true')
    p.add_argument('--sources', action='store_true')
    p.set_defaults(func=show)
    args = parser.parse_args()
    args.func(args)
