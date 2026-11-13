"""Local answer review ledger; never judges semantics by keyword matching."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT/'tests/evaluation/decision-v1'
RUNS = ROOT/'work/decision-evaluation'
REPORT = ROOT/'docs/test-scores.md'
CRITERIA = ['decision', 'conditions', 'grounded', 'useful_action']

def read_suite():
    return json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))['cases']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def register(answers, variant, artifact, run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('Use letters, digits, hyphen or underscore for run ID')
    cases=read_suite()
    with Path(answers).open(encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['id','answer']:
            raise ValueError('Expected exactly id,answer')
        rows=list(reader)
    if len(rows)!=len(cases) or {r['id'] for r in rows}!={c['id'] for c in cases}:
        raise ValueError('Missing, extra or duplicate IDs')
    if any(not r['answer'] or not r['answer'].strip() or len(r['answer'])>10000 for r in rows):
        raise ValueError('Empty or oversized answer')
    by_id={r['id']:r['answer'] for r in rows}
    run=dict(run_id=run_id, variant=variant,
        registered_at=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        artifact_sha256=digest(artifact), answers_sha256=digest(answers),
        suite_sha256=digest(SUITE/'cases.json'),
        reviewer='', provenance='real_model', latency_seconds=None, api_calls=None,
        cases=[dict(id=c['id'], pair=c['pair'], answer=by_id[c['id']],
                    scores={k:None for k in CRITERIA}, critical_error=None,
                    evidence_in_context=None, notes='') for c in cases])
    RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS/(run_id+'.json')).open('x',encoding='utf-8') as f:
        json.dump(run,f,ensure_ascii=False,indent=2)
    report()

def report():
    cases=read_suite(); runs=[]
    for path in sorted(RUNS.glob('*.json')):
        run=json.loads(path.read_text(encoding='utf-8'))
        if run['suite_sha256'] != digest(SUITE/'cases.json'):
            raise ValueError('Suite changed since run: '+path.name)
        if len(run['cases'])!=len(cases) or {r['id'] for r in run['cases']}!={c['id'] for c in cases}:
            raise ValueError('Invalid run cases: '+path.name)
        for row in run['cases']:
            if set(row['scores']) != set(CRITERIA) or any(type(v) not in (int,type(None)) or v not in (None,0,1) for v in row['scores'].values()):
                raise ValueError('Scores must be null, 0 or 1')
            if row['critical_error'] is not None and type(row['critical_error']) is not bool:
                raise ValueError('critical_error must be null, true or false')
        runs.append(run)
    lines=['# Журнал оценок тестов', '',
      'Обновляется командой `python tools/decision_evaluation.py report`. Не редактировать эту сводку вручную.',
      'Локальные оценки не равны баллам платформы. Пустая оценка — не ноль.', '',
      '## Конкурсные проверки (сообщены пользователем)', '',
      '| Версия | Статус | Балл платформы | Длительность |',
      '|---|---|---:|---:|',
      '| Исходная, 2026-11-09 | Использована | 46.694 | 285.8 с |',
      '| context-v1 | Использована | 53.028 | 272.2 с |',
      '| hybrid-v1 | Использована | 49.348 | 455.2 с |',
      '| anchored-v1 | Использована | 51.19 | 320.1 с |',
      '| hierarchy-v1 | Использована, успешно; 2026-11-10 11:47 | 47.083 | 275.1 с |',
      '| context-decision-v1 | Отправлена 2026-11-10 12:03; ошибка скоринга, попытка израсходована | — | Не сообщена |',
      '| context-decision-v2 | Использована, успешно; 2026-11-10 12:38 | 50.119 | 798.4 с |',
      '| hierarchy-decision-v1 | Подготовлена; не отправлена | — | — |',
      '| context-intent-v1 | Использована, успешно; 2026-11-10 22:40 | 49.623 | 593.0 с |',
      '| prompt-v1 | Использована, успешно; 23:34 | 52.95 | 578.4 с |',
      '| context-corrections-v1 | Использована, успешно; 2026-11-11 12:09 | 52.111 | 182.6 с |',
      '| context-rerank-review-v1 | Использована, успешно; 2026-11-11 21:27 | 49.829 | 653.9 с |',
      '| context-structured-query-v1 | Использована, успешно; 2026-11-11 21:39 | 50.879 | 608.6 с |',
      '| unified-v1 | Использована, успешно; 2026-11-11 22:56 | 54.393 | 451.4 с |',
      '| unified-v1-hybrid | Использована, успешно; 23:52 | 55.88 | 294.9 с |',
      '| unified-v1-hybrid-semantic07 | Лучшее решение; успешно; 2026-11-12 12:02 | 61.019 | 294.6 с |',
      '| unified-v1-hybrid-semantic05 | Использована, успешно; 2026-11-12 12:12 | 55.82 | 295.9 с |',
      '| unified-v1-hybrid-grouped-v1 | Использована, успешно; 2026-11-12 22:12 | 50.571 | 302.3 с |',
      '| unified-v1-hybrid-numbers | Подготовлена; балл не сообщён | — | — |',
      '| s07-grounded | Подготовлена; не отправлена | — | — |',
      '| s07-numeric | Подготовлена; не отправлена | — | — |',
      '| s07-grounded-numeric | Использована, успешно; 2026-11-12 20:21 | 44.625 | 694.6 с |',
      '| s07-chunk-paragraph | Использована, успешно; 22:29 | 54.944 | 345.3 с |',
      '| s07-chunk-semantic | Использована, успешно; 22:40 | 50.903 | 306.3 с |', '',
      'Источник истории: docs/results-analysis.md. Точные настройки и ответы этих запусков неизвестны.', '',
      '## Локальные наборы', '',
      '| Набор | Использование | Что известно |', '|---|---|---|',
      '| demo 1–18 | Уже использован для разработки | Реальные ответы текущей модели не оценены в этом журнале |',
      '| retrieval 1–18, p1–p8 | Уже использован для настройки поиска | 33/33 групп у context; это покрытие источников, не качество ответов |',
      '| decision-v1, P01–P10 A/B | Доступен разработчику; не скрытый контрольный | Статусы каждого вопроса ниже |', '',
      '## Прогоны decision-v1', '',
      '| Прогон / вариант | Тип | Оценено | Баллы / максимум оценённых | Полностью правильные пары |',
      '|---|---|---:|---:|---:|']
    reviewed={c['id']:[] for c in cases}; received={c['id']:0 for c in cases}
    for run in runs:
        completed=[]; passed=[]
        for row in run['cases']:
            if run['provenance']!='real_model':
                continue
            received[row['id']]+=1
            done=bool(run['reviewer']) and all(v is not None for v in row['scores'].values()) and row['critical_error'] is not None
            if done:
                score=sum(row['scores'].values()); completed.append(score)
                reviewed[row['id']].append(f"{run['run_id']}: {score}/4"+(' (критическая ошибка)' if row['critical_error'] else ''))
                if score==4 and not row['critical_error']: passed.append(row['id'])
        pairs=sum(all(p+'-'+s in passed for s in ('A','B')) for p in sorted({c['pair'] for c in cases}))
        value=f'{sum(completed)}/{4*len(completed)}' if completed else '—'
        lines.append(f"| {run['run_id']} / {run['variant']} | {run['provenance']} | {len(completed)}/20 | {value} | {pairs}/10 |")
    if not runs: lines.append('| — | Реальная генерация не запускалась | 0/20 | — | — |')
    lines+=['', '## Каждый сценарий', '', '| ID | Статус реальной генерации | Оценки |', '|---|---|---|']
    for c in cases:
        status=f"Получены ответы: {received[c['id']]}" if received[c['id']] else 'Не запускался'
        lines.append(f"| {c['id']} | {status} | {'; '.join(reviewed[c['id']]) or '—'} |")
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('report'); add=sub.add_parser('register')
    for name in ['answers','variant','artifact','run-id']: add.add_argument('--'+name,required=True)
    a=p.parse_args()
    if a.command=='report': report()
    else: register(a.answers,a.variant,a.artifact,a.run_id)
