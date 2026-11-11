"""Runtime-only dense index with lexical fallback; no bundled data."""
import math,time,json,sys
import config
from retrieval import Search,query
from context import deduplicate


def vector(raw):
    values=[float(x) for x in raw]
    if not values or not all(math.isfinite(v) for v in values):raise ValueError('Invalid embedding')
    norm=math.sqrt(sum(v*v for v in values))
    if not math.isfinite(norm) or norm==0:raise ValueError('Invalid embedding norm')
    return [v/norm for v in values]


def embed_batch(api,model,batch):
    data=api.post('/embeddings',{'model':model,'input':batch})['data']
    vectors=[None]*len(batch)
    for item in data:
        index=item['index']
        if not isinstance(index,int) or not 0<=index<len(batch) or vectors[index] is not None:raise ValueError('Invalid embedding index')
        vectors[index]=vector(item['embedding'])
    if any(v is None for v in vectors):raise ValueError('Missing embedding')
    return vectors


def embed(api,model,texts,workers=1):
    """Batches of 16; several batches in flight keep the preparation short on the platform."""
    batches=[texts[offset:offset+16] for offset in range(0,len(texts),16)]
    if workers>1 and len(batches)>1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(workers,len(batches))) as pool:
            results=list(pool.map(lambda batch:embed_batch(api,model,batch),batches))
    else:results=[embed_batch(api,model,batch) for batch in batches]
    output=[v for vectors in results for v in vectors]
    if output and any(len(v)!=len(output[0]) for v in output):raise ValueError('Embedding dimensions differ')
    return output


class HybridSearch:
    def __init__(self,lexical,document_vectors,query_vectors,policy="rrf",anchor_k=12):
        self.policy=policy;self.anchor_k=anchor_k
        self.lexical=lexical;self.items=lexical.items
        self.vectors=document_vectors;self.queries=query_vectors
        if len(self.items)!=len(self.vectors):raise ValueError('Document count mismatch')
        dims={len(v) for v in document_vectors+list(query_vectors.values())}
        if len(dims)!=1:raise ValueError('Embedding dimensions differ')
    def find(self,text,k=12):
        if text not in self.queries:return self.lexical.find(text,k)
        q=self.queries[text]
        similarity=[sum(a*b for a,b in zip(q,v)) for v in self.vectors]
        if self.policy=='weighted':
            # Tunable score-level fusion (config W_DENSE); lexical heuristics stay in force.
            return self.lexical.find(text,k,dense=similarity)
        dense=sorted(range(len(self.items)),key=lambda i:-similarity[i])
        if self.policy=='anchored':
            lexical=self.lexical.find(text,k)
            protected=lexical[:self.anchor_k]
            key=lambda c:(c['source'],c['heading'],c['text'])
            protected_keys={key(c) for c in protected}
            extras=[]
            for i in dense:
                candidate=self.items[i]
                if key(candidate) in protected_keys:continue
                extras.append(dict(candidate,semantic_supplement=True))
                if len(extras)>=int(config.get('HYBRID_EXTRAS')):break
            extra_keys={key(c) for c in extras}
            return [c for c in lexical if key(c) not in extra_keys]+extras
        # Rank fusion avoids comparing arbitrary BM25 and cosine scales.
        pool=min(len(self.items),max(36,k*2))
        lexical=self.lexical.find(text,pool)
        lookup={(c['source'],c['heading'],c['text']):i for i,c in enumerate(self.items)}
        scores={}
        for rank,i in enumerate(dense[:pool],1):scores[i]=1/(60+rank)
        for rank,c in enumerate(lexical,1):
            i=lookup[(c['source'],c['heading'],c['text'])]
            scores[i]=scores.get(i,0)+1/(60+rank)
        ordered=sorted(scores,key=lambda i:(-scores[i],i))
        return deduplicate([dict(self.items[i],score=round(scores[i],6)) for i in ordered])[:k]


def prepare(lexical,questions,api,model,deadline,anchor_k=12,workers=1):
    """All vectors are prepared once, before parallel generation. Fail closed to lexical."""
    try:
        if deadline-time.monotonic()<2:raise TimeoutError('No embedding budget')
        unique=list(dict.fromkeys(questions))
        doc_text=[c.get('title',c['heading'])+'\n'+c['text'] for c in lexical.items]
        docs=embed(api,model,doc_text,workers)
        queries=embed(api,model,[query(q) for q in unique],workers)
        if deadline-time.monotonic()<0:raise TimeoutError('Embedding budget exhausted')
        search=HybridSearch(lexical,docs,dict(zip(unique,queries)),policy=config.get('HYBRID_POLICY'),anchor_k=anchor_k)
        print(json.dumps({'event':'hybrid_ready','documents':len(docs),'queries':len(queries)}),file=sys.stderr)
        return search
    except Exception as exc:
        print(json.dumps({'event':'hybrid_fallback','reason':type(exc).__name__}),file=sys.stderr)
        return lexical
