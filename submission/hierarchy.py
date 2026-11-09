"""Runtime document routing; complete selected documents, lexical fallback."""
import re
from collections import defaultdict
from retrieval import Search,chunks,dialogue,topics,tokens

class HierarchicalSearch:
    def __init__(self,documents):
        self.documents={name.replace('\\','/'):text for name,text in documents}
        self.lexical=Search(chunks(documents));self.items=self.lexical.items
        self.titles={s:next((line.lstrip('# ').strip() for line in text.splitlines() if line.startswith('# ')),s) for s,text in self.documents.items()}
        self.document_topics={s:topics(title) for s,title in self.titles.items()}
        # Recognize corpus structure at runtime, never embed document contents.
        self.products={s for s in self.documents if '/20-products/' in '/'+s and re.search(r'/2[1-5]-','/'+s)}
    def route(self,question):
        latest,history=dialogue(question);active=topics(latest)
        if not active:
            for earlier in reversed(history):
                active=topics(earlier)
                if active:break
        if not active:return [],self.lexical.find(question,k=100)
        if active=={'photo'} and history:active |= topics(' '.join(history[-2:])) & {'metering'}
        ranked=self.lexical.find(question,k=100)
        grouped=defaultdict(list)
        for c in ranked:grouped[c['source']].append(c['score'])
        scores={s:sum(values[:3])/len(values[:3]) for s,values in grouped.items() if s in self.products}
        if active:scores={s:v for s,v in scores.items() if active & self.document_topics.get(s,set())}
        ordered=sorted(scores,key=lambda s:(-scores[s],s))
        if not ordered:return [],ranked
        # For an ambiguous question, avoid committing to a product on a weak tie.
        if not active and (len(ordered)<2 or scores[ordered[0]]<1.2*scores[ordered[1]]):return [],ranked
        selected=[ordered[0]]
        if len(ordered)>1 and len(active)>1:
            remaining=active-self.document_topics.get(selected[0],set())
            for s in ordered[1:]:
                if remaining & self.document_topics.get(s,set()):selected.append(s);break
        return selected,ranked
    def find(self,question,k=12):
        selected,ranked=self.route(question)
        if not selected:return self.lexical.find(question,k)
        result=[];whole=set();chars=0
        for s in selected:
            text=self.documents[s]
            # Reserve context room for facts from reference and energy documents.
            if chars+len(text)>29000:continue
            result.append({'source':s,'heading':self.titles[s],'title':self.titles[s],
                           'text':text,'whole_document':True})
            whole.add(s);chars+=len(text)
        if not result:return self.lexical.find(question,k)
        # Complement complete product material with targeted non-product evidence.
        remainder=[c for c in ranked if c['source'] not in whole]
        same_product=[c for c in remainder if c['source'] in selected]
        references=[c for c in remainder if c['source'] not in self.products]
        # Keep the highest-scoring independent evidence before broad related FAQs.
        result+=same_product[:3]+references[:4]
        return result[:max(k,4)]
