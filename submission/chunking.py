"""Split sections for retrieval while retaining full sections as evidence."""

import re
import time


MODES = ('paragraph', 'semantic', 'window')
MIN_SECTION_CHARS = 700


def prepare(search, questions, api, model, deadline, anchor_k=12, workers=1):
    """Prepare chunk boundaries and the dense index within the same budget."""
    import json
    import sys

    import config
    from hybrid import prepare as prepare_hybrid
    from retrieval import Search

    mode = config.get('CHUNK_MODE')
    try:
        items = split_sections(search.items, mode, api, model, deadline, workers)
    except Exception as exc:
        print(json.dumps({'event': 'semantic_chunking_fallback',
                          'reason': type(exc).__name__}), file=sys.stderr)
        items = split_sections(search.items, 'paragraph')
    return prepare_hybrid(Search(items), questions, api, model, deadline,
                          anchor_k=anchor_k, workers=workers)


def paragraphs(text):
    # Markdown tables and contiguous lists stay in the same block.
    return re.split(r'\n[ \t]*\n+', text)


def windows(text, size=120, overlap=30):
    words = list(re.finditer(r'\S+', text))
    if len(words) <= size:
        return [text]
    pieces = []
    for start in range(0, len(words), size - overlap):
        end = min(start + size, len(words))
        pieces.append(text[words[start].start():words[end - 1].end()])
        if end == len(words):
            break
    return pieces


def semantic_groups(blocks, vectors, threshold=0.65, max_chars=1800):
    groups = [blocks[0]]
    for previous, current, block in zip(vectors, vectors[1:], blocks[1:]):
        similarity = sum(a * b for a, b in zip(previous, current))
        if similarity >= threshold and len(groups[-1]) + len(block) + 2 <= max_chars:
            groups[-1] += '\n\n' + block
        else:
            groups.append(block)
    return groups


def split_sections(sections, mode, api=None, model=None, deadline=None, workers=1):
    layouts = []
    for section in sections:
        body = section['text']
        if len(body) <= MIN_SECTION_CHARS:
            blocks = [body]
        elif mode == 'window':
            blocks = windows(body)
        else:
            blocks = [block.strip() for block in paragraphs(body) if block.strip()]
        layouts.append(blocks)

    if mode == 'semantic' and api is not None:
        from hybrid import embed

        # Use paragraph text alone so the shared heading cannot inflate similarity.
        inputs = [block for blocks in layouts if len(blocks) > 1 for block in blocks]
        if inputs:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('No semantic chunking budget')
            vectors = iter(embed(api, model, inputs, workers))
            layouts = [
                semantic_groups(blocks, [next(vectors) for _ in blocks])
                if len(blocks) > 1 else blocks for blocks in layouts
            ]

    items = []
    for section, blocks in zip(sections, layouts):
        if len(blocks) == 1:
            items.append(section)
        else:
            items.extend(dict(section, text=block, parent_text=section['text']) for block in blocks)
    return items


def restore_sections(candidates):
    """Keep the highest scoring chunk for each section and restore its evidence."""
    seen = set()
    for candidate in candidates:
        text = candidate.get('parent_text', candidate['text'])
        key = candidate['source'], candidate.get('title'), candidate['heading'], text
        if key not in seen:
            seen.add(key)
            restored = dict(candidate, text=text)
            restored.pop('parent_text', None)
            yield restored
