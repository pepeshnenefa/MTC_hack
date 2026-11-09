"""Conservative duplicate removal and grounded context packing."""
import re
from retrieval import dialogue, topics, tokens


def normalized(text):
    return ' '.join(text.lower().replace('ё','е').split())


def signature(text):
    # Preserve differences in numeric bounds and negation. Similar is not equivalent.
    return (tuple(re.findall(r'\d+(?:[.,]\d+)?',text)),
            tuple(re.findall(r'\b(?:не|нет|нельзя|без|только|кроме)\b',normalized(text))))


def duplicate(left,right):
    a,b=normalized(left['text']),normalized(right['text'])
    same_product=(left['source']==right['source'] or
                  left.get('title','').split(' / ')[0]==right.get('title','!').split(' / ')[0])
    if not same_product:return False
    if a==b:return True
    # Only punctuation/layout variants of the same words are duplicates.
    # A high similarity ratio alone can hide a changed price or condition.
    if signature(a)!=signature(b):return False
    pattern=r'\d+(?:[.,]\d+)?|[^\W\d_]+'
    return re.findall(pattern,a)==re.findall(pattern,b)


def deduplicate(sources):
    selected=[]
    for source in sources:
        if not any(duplicate(source,other) for other in selected):selected.append(source)
    return selected


def select_sources(question,sources,limit=12):
    sources=deduplicate(sources)
    semantic=[s for s in sources if s.get("semantic_supplement")][:2]
    sources=[s for s in sources if not s.get("semantic_supplement")]
    if not sources:return semantic
    latest,history=dialogue(question)
    active=topics(latest)
    if not active:
        for previous in reversed(history):
            active=topics(previous)
            if active:break
    # Use the strongest evidence as a secondary relevance gate for qualifications.
    primary=sources[:min(3,limit)]
    primary_files={s['source'] for s in primary}
    support=[]
    for s in sources[len(primary):]:
        title=s.get('title',s['heading'])
        scope=bool(re.search(r'ограничен|не подходит|чего утверждать нельзя|что мы не делаем|условия',title.lower()))
        source_topics=topics(title)
        related=(bool(active & source_topics) if active and source_topics
                 else s['source'] in primary_files)
        if scope and related:support.append(s)
    # Qualifications supplement, never evict, the top-k direct evidence.
    base=sources[:limit]
    extras=[s for s in support if s not in base][:2]
    important=[s for s in support if s in base]+extras
    return primary+important+[s for s in base if s not in primary and s not in important]+semantic


def pack_context(question,sources,limit=12,max_chars=40000):
    selected=select_sources(question,sources,limit)
    entries=[];included=[];dropped=[];used=0
    for source in selected:
        entry=source['source']+' / '+source.get('title',source['heading'])+'\n'+source['text']
        cost=len(entry)+(2 if entries else 0)
        if used+cost>max_chars:
            dropped.append(source);continue
        entries.append(entry);included.append(source);used+=cost
    return '\n\n'.join(entries),{'selected':selected,'included':included,'dropped':dropped,'characters':used}
