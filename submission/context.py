"""Conservative duplicate removal and grounded context packing."""
import re
from functools import lru_cache
from retrieval import dialogue, topics, tokens
import config


@lru_cache(maxsize=8192)
def normalized(text):
    return ' '.join(text.lower().replace('ё','е').split())


@lru_cache(maxsize=8192)
def signature(text):
    # Preserve differences in numeric bounds and negation. Similar is not equivalent.
    return (tuple(re.findall(r'\d+(?:[.,]\d+)?',text)),
            tuple(re.findall(r'\b(?:не|нет|нельзя|без|только|кроме)\b',normalized(text))))


@lru_cache(maxsize=8192)
def word_sequence(text):
    return tuple(re.findall(r'\d+(?:[.,]\d+)?|[^\W\d_]+',text))


def duplicate(left,right):
    a,b=normalized(left['text']),normalized(right['text'])
    same_product=(left['source']==right['source'] or
                  left.get('title','').split(' / ')[0]==right.get('title','!').split(' / ')[0])
    if not same_product:return False
    if a==b:return True
    # Only punctuation/layout variants of the same words are duplicates.
    # A high similarity ratio alone can hide a changed price or condition.
    if signature(a)!=signature(b):return False
    return word_sequence(a)==word_sequence(b)


def deduplicate(sources):
    selected=[]
    for source in sources:
        if not any(duplicate(source,other) for other in selected):selected.append(source)
    return selected


def key(source):
    return source['source'],source.get('title',source['heading']),source['text']


def select_sources(question,sources,limit=12):
    g=config.get
    # Deterministic guardrail sections travel with the candidates but are placed separately.
    forced=[s for s in sources if s.get('forced')]
    sources=deduplicate([s for s in sources if not s.get('forced')])
    semantic=[s for s in sources if s.get("semantic_supplement")][:int(g('HYBRID_EXTRAS'))]
    sources=[s for s in sources if not s.get("semantic_supplement")]
    if not sources:return semantic+forced
    latest,history=dialogue(question)
    active=topics(latest)
    if not active:
        for previous in reversed(history):
            active=topics(previous)
            if active:break
    # Use the strongest evidence as a secondary relevance gate for qualifications.
    primary=sources[:min(int(g('PRIMARY')),limit)]
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
    extras=[s for s in support if s not in base][:int(g('SUPPORT_EXTRAS'))]
    important=[s for s in support if s in base]+extras
    rest=[s for s in base if s not in primary and s not in important]
    # Optional precision gate: weak tail sections dilute attention more than they inform.
    ratio=g('MIN_SCORE_RATIO')
    if ratio>0 and sources[0].get('score'):
        floor=ratio*sources[0]['score']
        rest=[s for s in rest if s.get('score',floor)>=floor]
    selected=primary+important+rest+semantic
    present={key(s) for s in selected}
    return selected+[s for s in forced if key(s) not in present]


def pack_context(question,sources,limit=12,max_chars=None):
    if max_chars is None:max_chars=int(config.get('CONTEXT_CHARS'))
    selected=select_sources(question,sources,limit)
    grounded=bool(config.get('GROUNDED_CONTEXT'))
    if grounded:
        # Entity-aware context: sections of other products are capped, every section carries its product tag.
        import grounding
        registry=grounding.REGISTRY or grounding.Registry(sources)
        active,_=grounding.focus(question,selected)
        selected=grounding.arrange(selected,sources,active,int(config.get('OFF_PRODUCT_MAX')))
    # Guardrails are short and must not be the part that falls out of the budget.
    selected=sorted(selected,key=lambda s:0 if s.get('forced') else 1) if any(s.get('forced') for s in selected) else selected
    entries=[];included=[];dropped=[];used=0
    if grounded:used=len(grounding.NOTE)+2
    for source in selected:
        entry=grounding.render([source],registry)[0] if grounded else source['source']+' / '+source.get('title',source['heading'])+'\n'+source['text']
        cost=len(entry)+(2 if entries else 0)
        if used+cost>max_chars:
            dropped.append(source);continue
        entries.append((source,entry));included.append(source);used+=cost
    # Reading order for the model: direct evidence first, guardrails last (next to the question).
    order=[e for e in entries if not e[0].get('forced')]+[e for e in entries if e[0].get('forced')]
    included=[e[0] for e in order]
    text='\n\n'.join(e[1] for e in order)
    if grounded and text:text=grounding.NOTE+'\n\n'+text
    return text,{'selected':selected,'included':included,'dropped':dropped,'characters':used}
