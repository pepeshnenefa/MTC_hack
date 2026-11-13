"""Compare archive retrieval and run CSV integration with synthetic embeddings."""

import json
import math
from pathlib import Path
import subprocess
import sys
import time

from build_chunking_submissions import BASE, MODES, ROOT
import mock_platform_run as mock


def measure(archive):
    sys.path.insert(0, str(archive))
    import solution
    from context import pack_context
    from retrieval import query

    from chunking import semantic_groups, split_sections, windows
    from hybrid import HybridSearch, vector

    assert semantic_groups(['a', 'b', 'c'], [[1, 0], [1, 0], [0, 1]]) == ['a\n\nb', 'c']
    assert windows('a b c d e', size=3, overlap=1) == ['a b c', 'c d e']
    start = time.perf_counter()
    sections = solution.chunks(solution.documents(ROOT / 'Knowledge-base'))
    mode = solution.config.DEFAULTS.get('CHUNK_MODE')

    class Embeddings:
        calls = 0

        def post(self, path, payload):
            assert path == '/embeddings'
            self.calls += 1
            return {'data': [
                {'index': index, 'embedding': mock.vector(text)}
                for index, text in enumerate(payload['input'])
            ]}

    api = Embeddings()
    items = split_sections(sections, mode, api, 'mock') if mode else sections
    search = solution.Search(items)
    indexing_ms = round((time.perf_counter() - start) * 1000, 1)
    assert len(items) > len(sections) if mode else len(items) == len(sections)
    assert all((item['text'] in item.get('parent_text', item['text'])) for item in items
               if mode != 'semantic')
    questions = {row['id']: row['question'] for row in mock.read_questions(ROOT / 'demo_questions.csv')}
    cases = json.loads((ROOT / 'tests/retrieval_cases.json').read_text(encoding='utf-8'))
    questions.update({case['id']: case['question'] for case in cases if 'question' in case})
    dense = HybridSearch(
        search, [vector(mock.vector((item.get('title') or '') + '\n' + item['text'])) for item in items],
        {q: vector(mock.vector(query(q))) for q in questions.values()}, policy='weighted',
    )
    metrics = {}
    for label, index in [('lexical', search), ('synthetic_hybrid', dense)]:
        hits = total = characters = 0
        misses = []
        for case in cases:
            question = questions[case['id']]
            _, report = pack_context(question, index.find(question, k=36), limit=12)
            characters += report['characters']
            sources = report['included']
            assert len(sources) == len({(s['source'], s['title'], s['text']) for s in sources})
            for group in case['groups']:
                total += 1
                if any(s['source'].replace('\\', '/').endswith(group['source'])
                       and s['heading'] in group['headings'] for s in sources):
                    hits += 1
                else:
                    misses.append(case['id'])
        metrics[label] = {'hits': hits, 'total': total, 'misses': misses,
                          'mean_context_chars': round(characters / len(cases))}
    return {'archive': archive.name, 'items': len(items), 'indexing_ms': indexing_ms,
            'boundary_batches': api.calls, 'document_batches': math.ceil(len(items) / 16),
            'metrics': metrics}


if __name__ == '__main__':
    if len(sys.argv) == 2:
        print(json.dumps(measure(Path(sys.argv[1]))))
    else:
        output = ROOT / 'work' / 'chunking-check'
        output.mkdir(parents=True, exist_ok=True)
        # This experiment exercises retrieval. Use plain mock answers for all four archives.
        mock.REPLIES[:] = ['Тестовый ответ для проверки формата CSV.'] * len(mock.REPLIES)
        reports = []
        for archive in [BASE, *(ROOT / 'dist' / f'energytool-s07-chunk-{mode}.zip' for mode in MODES)]:
            run = subprocess.run([sys.executable, __file__, str(archive)], check=True,
                                 capture_output=True, text=True)
            report = json.loads(run.stdout)
            report['csv'] = mock.run(archive, ROOT / 'demo_questions.csv', ROOT / 'Knowledge-base',
                                     output / (archive.stem + '.csv'))
            assert report['csv']['passed'] and 'hybrid_ready' in report['csv']['stderr_events']
            assert report['csv']['calls']['embeddings'] == (
                report['boundary_batches'] + report['document_batches'] + 2)
            reports.append(report)
        result = json.dumps(reports, indent=2, ensure_ascii=False)
        (output / 'report.json').write_text(result, encoding='utf-8')
        print(result)
