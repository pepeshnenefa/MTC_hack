"""Qwen3 Reranker stage for semantic07: the hybrid pool (BM25 + bge-m3, usually 36 sections)
is reordered so that the RERANK_KEEP (12) sections the reranker finds most relevant come first.

Nothing else changes: the pool itself, its remaining sections (still available to the
limits/conditions supplement of context.py), forced guardrail sections and the context packing
stay as in semantic07. Any failure keeps the hybrid order of semantic07 for that question;
repeated failures or slow calls switch the stage off for the rest of the run.
Standard library only; the reranker is reached through the same platform proxy and key.
"""
import json
import math
import sys
import threading
import time
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import re

import config
from context import select_sources
from retrieval import query

# Default from the organizers' sample (sample_submission/solution.py); RERANKER_MODEL wins.
DEFAULT_MODEL = 'qwen/qwen3-reranker-8b'
LIMITS = re.compile(r'ограничен|не подходит|чего утверждать нельзя|что мы не делаем|условия')
RETRY_STATUS = (408, 429, 500, 502, 503, 504)
# The endpoint or the key is unusable: no later question can succeed either.
FATAL_STATUS = (401, 403, 404, 405)


def document(source):
    """Same text as the embedding index (title + body), cut for the reranker input only."""
    text = source.get('title', source['heading']) + '\n' + source['text']
    return text[:max(1, int(config.get('RERANK_DOC_CHARS')))]


def ranking(data, size):
    """Indices from a /rerank response, most relevant first. Strict: any invalid entry rejects it."""
    entries = data.get('results', data.get('data')) if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError('No rerank results')
    indices, scores = [], []
    for entry in entries:
        index = entry.get('index') if isinstance(entry, dict) else None
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < size or index in indices:
            raise ValueError('Invalid rerank index')
        score = entry.get('relevance_score', entry.get('score'))
        indices.append(index)
        scores.append(float(score) if isinstance(score, (int, float)) and not isinstance(score, bool)
                      and math.isfinite(score) else None)
    if all(s is not None for s in scores):
        # Proxies differ in the order they return; the score decides (ties keep the response order).
        order = sorted(range(len(indices)), key=lambda i: -scores[i])
        return [indices[i] for i in order], [scores[i] for i in order]
    return indices, scores


def limits_section(source):
    # The same title test as context.select_sources: limits, "what we do not do", conditions.
    return bool(LIMITS.search(source.get('title', source['heading']).lower()))


def protect(question, sources, ranked, order, others=()):
    """Keep every limits/conditions section of the semantic07 context in the new context.

    Relevance ranking pushes "what must not be claimed" and "what we do not do" sections down,
    yet they are what keeps answers inside the knowledge base. Each one semantic07 would show is
    first placed right after the TOP_K chosen sections, where the unchanged supplement of
    context.py (up to SUPPORT_EXTRAS limits sections beyond TOP_K) takes it as in semantic07.
    Only a section the supplement still leaves out replaces the reranker's weakest pick among the
    first TOP_K (all of which reach the model). Returns (order, number of picks replaced)."""
    limit = max(1, min(100, int(config.get('TOP_K'))))  # the context size used by solution.API.answer
    position = {id(s): i for i, s in enumerate(ranked)}
    protected = [position[id(s)] for s in select_sources(question, sources, limit)
                 if id(s) in position and limits_section(s)]
    if not protected:
        return order, 0
    base = order[:limit]
    order = base + [i for i in protected if i not in base] + [i for i in order[limit:] if i not in protected]
    replaced = 0
    for _ in protected:
        chosen = {id(s) for s in select_sources(question, [ranked[i] for i in order] + list(others), limit)}
        need = [i for i in protected if id(ranked[i]) not in chosen]
        free = [i for i in order[:limit] if i not in protected]
        if not need or not free:
            break
        # The weakest free pick moves just behind the chosen sections; the missing limits section takes its place.
        weakest, missing = free[-1], need[0]
        front = [missing if i == weakest else i for i in order[:limit]]
        order = front + [weakest] + [i for i in order[limit:] if i != missing]
        replaced += 1
    return order, replaced


class Stage:
    def __init__(self, api):
        self.api = api
        self.lock = threading.Lock()
        self.bad = 0
        self.disabled = None
        self.announced = False

    def log(self, event, **fields):
        print(json.dumps(dict(event=event, **fields)), file=sys.stderr)

    def post(self, body):
        """One reranker request with its own short timeout; one retry only for a quick refusal."""
        g = config.get
        attempts = max(1, int(g('RERANK_ATTEMPTS')))
        for attempt in range(attempts):
            left = (self.api.deadline - time.monotonic() if self.api.deadline is not None else 1e9) - g('RERANK_RESERVE')
            if left <= 1:
                raise TimeoutError('Rerank deadline')
            request = Request(self.api.base + '/rerank', data=json.dumps(body).encode(),
                              headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api.key})
            try:
                with urlopen(request, timeout=min(g('RERANK_TIMEOUT'), left)) as response:
                    return json.load(response)
            except HTTPError as exc:
                if exc.code not in RETRY_STATUS or attempt + 1 >= attempts:
                    raise
            time.sleep(1)

    def failed(self, reason, fatal=False):
        with self.lock:
            self.bad += 1
            if self.disabled is None and (fatal or self.bad >= int(config.get('RERANK_MAX_BAD'))):
                self.disabled = reason
                self.log('rerank_disabled', reason=reason, bad=self.bad)

    def order(self, question, sources):
        """(sources, info): the pool with the reranker's choice first, or the unchanged hybrid order."""
        g = config.get
        if not g('USE_RERANKER'):
            return sources, {'status': 'off'}
        if self.disabled is not None:
            return sources, {'status': 'disabled', 'reason': self.disabled}
        # Guardrail and supplement sections are placed by context.py itself; only search results compete.
        ranked = [s for s in sources if not s.get('forced') and not s.get('semantic_supplement')]
        others = [s for s in sources if s.get('forced') or s.get('semantic_supplement')]
        keep = max(1, int(g('RERANK_KEEP')))
        if len(ranked) < 2:
            return sources, {'status': 'skipped', 'reason': 'pool'}
        if self.api.deadline is not None and self.api.deadline - time.monotonic() < g('RERANK_RESERVE'):
            return sources, {'status': 'skipped', 'reason': 'time'}
        started = time.monotonic()
        try:
            data = self.post({'model': self.api.reranker, 'query': query(question),
                              'documents': [document(s) for s in ranked], 'top_n': min(keep, len(ranked))})
            indices, scores = ranking(data, len(ranked))
        except Exception as exc:
            seconds = round(time.monotonic() - started, 2)
            code = exc.code if isinstance(exc, HTTPError) else None
            reason = type(exc).__name__ + (':' + str(code) if code else '')
            self.log('rerank_fallback', reason=reason, seconds=seconds)
            self.failed(reason, fatal=code in FATAL_STATUS)
            return sources, {'status': 'fallback', 'reason': reason, 'seconds': seconds}
        seconds = round(time.monotonic() - started, 2)
        picked = indices[:keep]
        # A proxy that returns fewer results than asked: the hybrid order fills the free places.
        filled = [i for i in range(len(ranked)) if i not in picked][:keep - len(picked)]
        head = picked + filled
        order, restored = protect(question, sources, ranked, head + [i for i in range(len(ranked)) if i not in head], others)
        slow = seconds > g('RERANK_SLOW')
        with self.lock:
            if not slow:
                self.bad = 0
            first = not self.announced
            self.announced = True
        if first:
            self.log('rerank_ready', documents=len(ranked), returned=len(indices), seconds=seconds)
        if slow:  # the answer is still used; only the next questions are protected
            self.failed('slow')
        info = {'status': 'ok', 'seconds': seconds, 'pool': len(ranked), 'picked': order[:keep],
                'filled': len(filled), 'restored_limits': restored,
                'scores': [round(s, 4) if s is not None else None for s in scores[:keep]]}
        return [ranked[i] for i in order] + others, info
