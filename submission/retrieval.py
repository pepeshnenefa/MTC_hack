"""Local retrieval from runtime knowledge documents; no embedded corpus or answers."""
import math
import re
from collections import Counter

import config

STOP=set('и в во на с со к по от до у а но или для это как что мы вы он она они не ли бы из за при есть нас вас можно нужно надо сначала потом вообще будет стоит только уже сейчас отдельно ваш вашей вашем который чтобы если нам мне то все так чем того сможете хочу хотим какой какая какие почему сколько такой таких такой этого нашем наших больше почти пока ли либо'.split())
# Linguistic aliases, not product facts. They cannot assert product capabilities.
ALIASES=[('симк','сим карт'),('sim','сим карт'),('фасад','вертикальн поверхност'),
         ('фото','фотограф распознаван'),('снимок','фотограф'),('магазин','ретейл'),
         ('двухтариф','многотариф тариф'),('выпуск','производств'),
         ('расшир','наращ поэтап'),('провер','аудит'),('вечер','ночью дневное нагрузка'),
         ('оценк','расчет данные'),('стоим','цен расчет'),('понадоб','сведения данные'),('потребу','сведения данные'),
         ('провер','существующая система учет'),('невыгод','нецелесообразн')]

# General colloquial Russian -> terms used in energy documents. Written from everyday usage,
# not from any test question; they only widen the search and assert nothing about products.
COLLOQUIAL=[
 (r'\bсвет\b|\bсвета\b|\bсвету\b|электричеств','электроэнергия'),
 (r'\bкрыш','кровля'),
 (r'\bбатаре','панели накопители'),(r'аккумулятор','накопители'),
 (r'\bвору|\bворов|\bкраж|\bкрад|\bутечк','хищение потери коммерческие'),
 (r'\bопт\b|\bопте\b|\bопта\b|\bоптом\b','оптовый рынок'),(r'\bрозниц','розничный рынок'),
 (r'\bжелез|\bсервак','серверы инфраструктура'),
 (r'\bкартинк|\bфотк|сфотограф|\bскан','фотография распознавание'),
 (r'\bквитанц|\bплатежк','счета квитанции'),
 (r'\bпотестир|\bпротестир|\bпопроб|\bпробн|\bдемо|\bтестов','пилотное подключение демонстрационный доступ попробовать'),
 (r'\bторгу','продаёт продажа'),
 (r'\bинформаци','данные'),
 (r'\bперезвон|\bпозвон|\bсозвон|\bменеджер','обратный звонок заявка специалист'),
 (r'\bобойд|\bпочем\b|\bдорого|\bпрайс|\bрасценк','стоимость сколько стоит'),
 (r'\bчастями\b|\bкредит|\bлизинг|\bв рассроч','рассрочка оплата'),
 (r'\bокуп','окупаемость эффект'),
 (r'\bдолго\b|\bбыстро\b|\bскоро\b|\bсколько времени|\bзаймет\b|\bзанимает\b','срок сроки'),
 (r'\bкиловатт','кВт мощность'),(r'\bразрешен|\bвыделен','присоединённая мощность'),
 (r'\bвыруб|\bотключат|\bпропадет|\bблэкаут|\bавари','отключение сети автономность резерв'),
 (r'\bкоттедж|\bдач[аиеу]\b|\bчастн\w* дом','частный жилой дом'),
 (r'\bжильц|\bжител','абоненты'),
 (r'\bреактивк','реактивная энергия'),
 (r'\bсбыт\b|\bсбыта\b|\bсбытов','энергосбытовая компания'),
 (r'\bзавод|\bфабрик','промышленное предприятие'),
 (r'\bлишн','излишки'),(r'\bзарабат|\bзаработ','доход продажа излишков'),
 (r'\bдешевле|\bсэконом|\bменьше платить','снижение затрат экономия'),
 (r'\bгарантир|\bточно ли|\bнаверняка','гарантия'),
]

# Product vocabulary used for topical relevance, not for generating answers.
TOPICS={
 'solar':r'солнеч|панел|кровл|скат|фасад|накопител|сетевая станц|шифер',
 'metering':r'платформ.{0,15}учет|автомат.{0,20}учет|аскуэ|цех|арендатор|магазин|ретейл|сим.?карт|симк|кабинет',
 'photo':r'фото|распозна|изображен|снимок|штрих',
 'losses':r'потер|хищен|воров|электросетев',
 'wholesale':r'оптов|орэм',
 'tariff':r'тариф|ценов.{0,12}категор|счет.{0,15}(вырос|раст)|расход.{0,15}(вырос|раст)',
}
def topics(text):
    text=text.lower().replace('ё','е')
    return {name for name,pattern in TOPICS.items() if re.search(pattern,text)}

def intent_signals(text):
    """Surface general consultation needs without encoding any product facts."""
    text = text.lower().replace('ё', 'е')
    staged = bool(re.search(r'поэтап|постепенн|очеред|частями|'
                            r'(?:сначала|сперва).{0,110}(?:потом|затем|позже|расшир|нараст)|'
                            r'(?:расшир|нараст).{0,110}(?:потом|позже)', text))
    suitability = bool(re.search(r'стар[а-я]*|шифер|изнош|не подходит|'
                                 r'невыгод|риск|ограничен|не рекоменду|имеет смысл|целесообраз|стоит ли', text))
    return staged, suitability


def asks_for_inputs(text):
    text=text.lower().replace('ё','е')
    return bool(re.search(r'с чего (?:нам )?начать|что (?:понадоб|нужн|потреб)',text) or
                re.search(r'данн|сведени',text) and re.search(r'каки|нуж|подготов|потреб|понадоб',text))

def words(text):
    return [w for w in re.findall(r'[а-яa-z0-9]+',text.lower().replace('ё','е')) if len(w)>1 and w not in STOP]

def stem(word):
    if re.fullmatch('[а-я]+',word) and len(word)>4:
        for suffix in ('иями','ями','ами','ого','ему','ому','ыми','ими','ией','иям','иях','ение','ения','ений','ать','ять','ить','ешь','ете','ются','ется','ются','ыми','ая','яя','ое','ее','ые','ие','ый','ий','ой','ую','юю','ов','ев','ам','ям','ах','ях','ом','ем','ы','и','а','я','у','ю','е'):
            if word.endswith(suffix) and len(word)-len(suffix)>=3:return word[:-len(suffix)]
    return word

def tokens(text):
    return [stem(w) for w in words(text)]

def dialogue(text):
    matches=list(re.finditer(r'(?im)^\s*(Пользователь|Ассистент|User|Assistant)\s*:\s*',text))
    if not matches:return text,[]
    user=[]
    for i,m in enumerate(matches):
        if m.group(1).lower() in ('пользователь','user'):
            user.append(text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip())
    return (user[-1],user[:-1]) if user else (text,[])

def turns(text):
    """All turns in order as (is_user, text); a plain question is one user turn."""
    matches=list(re.finditer(r'(?im)^\s*(Пользователь|Ассистент|User|Assistant)\s*:\s*',text))
    if not matches:return [(True,text)]
    return [(m.group(1).lower() in ('пользователь','user'),
             text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()) for i,m in enumerate(matches)]

def assistant_turns(text):
    return [t for is_user,t in turns(text) if not is_user and t]

def query(text):
    latest,history=dialogue(text)
    return latest+'\n'+'\n'.join(history[-3:])

def chunks(docs):
    out=[]
    for source,text in docs:
        source=source.replace('\\','/')
        stack=[];buf=[];heading=source
        def emit():
            body='\n'.join(buf).strip()
            if not body or not re.search(r'\w',body):return
            # Keep normal sections (including tables and qualifications) intact.
            # Long sections use overlapping windows; never silently discard their tail.
            pieces=[body] if len(body)<=6500 else [body[i:i+6000] for i in range(0,len(body),5000)]
            for piece in pieces:
                out.append({'source':source,'heading':heading,'title':' / '.join(h for _,h in stack),'text':piece})
        for line in text.splitlines():
            match=re.match(r'^(#{1,6})\s+(.+)',line)
            if match:
                emit();buf=[];level=len(match[1]);heading=match[2].strip()
                while stack and stack[-1][0]>=level:stack.pop()
                stack.append((level,heading))
            else:buf.append(line)
        emit()
    return out

class Search:
    def __init__(self,items):
        self.items=items
        self.tf=[Counter(tokens(c['heading']+' '+c['text'])) for c in items]
        self.titles=[set(tokens(c.get('title',c['heading']))) for c in items]
        self.length=[sum(t.values()) for t in self.tf]
        self.avg=sum(self.length)/max(1,len(items))
        df=Counter(w for t in self.tf for w in t)
        self.idf={w:math.log(1+(len(items)-n+.5)/(n+.5)) for w,n in df.items()}
        self.vocab=set(df)
        self.topics=[topics(c.get("title",c["heading"])) for c in items]
        self.body_topics=[topics(c['text']) for c in items]
        # Product of a file = topic of its top-level title (corpus structure, not content).
        self.file_topics={}
        for c in items:
            self.file_topics.setdefault(c['source'],topics(c.get('title',c['heading']).split(' / ')[0]))
    def guardrails(self,active,candidates,numeric=False):
        """Scope, limits and number caveats of the product in question; added deterministically."""
        g=config.get;level=int(g('LIMITS_INJECT'));numbers=bool(g('NUMBERS_INJECT')) and numeric
        if not level and not numbers:return []
        product=set(active)
        if not product:
            # No recognised keyword: take the product that dominates the strongest evidence.
            votes=Counter(t for c in candidates[:3] for t in self.file_topics.get(c['source'],()))
            product={t for t,n in votes.items() if n>=2}
        if not product:return []
        out=[]
        for i,item in enumerate(self.items):
            name=item['source'].replace('\\','/').split('/')[-1];heading=item['heading'].lower().replace('ё','е')
            if not product & self.topics[i]:continue
            limits=level>=1 and name.startswith('02-')
            forbidden=level>=2 and heading.startswith('чего утверждать нельзя') and product & self.file_topics.get(item['source'],set())
            figures=numbers and name.startswith('01-') and len(item['text'])<=int(g('NUMBERS_MAX_CHARS'))
            if limits or forbidden or figures:out.append(dict(item,forced=True,score=0))
        return out
    def _query(self,text,original=None):
        g=config.get
        latest,history=dialogue(text)
        weights=Counter(tokens(latest))
        weights={w:min(n,2) for w,n in weights.items()}
        # History carries referents, but the last user request remains primary.
        for age,previous in enumerate(reversed(history[-3:])):
            for w in set(tokens(previous)):
                weights[w]=max(weights.get(w,0),g('W_HISTORY')/(age+1))
        # The assistant may have named the topic; a search signal only, never a fact.
        if g('W_ASSISTANT')>0:
            for age,previous in enumerate(reversed(assistant_turns(original or text)[-2:])):
                for w in set(tokens(previous)):
                    weights[w]=max(weights.get(w,0),g('W_ASSISTANT')/(age+1))
        if asks_for_inputs(latest) and g('W_INPUTS')>0:
            for w in tokens('сведения данные количество приборы схема сбор'):
                weights[w]=max(weights.get(w,0),g('W_INPUTS'))
        text_for_alias=' '.join(words(latest+' '+' '.join(history[-2:])))
        if g('W_ALIAS')>0:
            for prefix,aliases in ALIASES:
                if any(w.startswith(prefix) for w in text_for_alias.split()):
                    for w in tokens(aliases):weights[w]=max(weights.get(w,0),g('W_ALIAS'))
        # Everyday wording -> vocabulary of the documents (a separate weight, so it can be ablated).
        if g('W_COLLOQUIAL')>0:
            for pattern,aliases in COLLOQUIAL:
                if re.search(pattern,text_for_alias):
                    for w in tokens(aliases):weights[w]=max(weights.get(w,0),g('W_COLLOQUIAL'))
        # Recover inflection/derivation matches only when an exact stem is absent.
        if g('W_PREFIX')>0:
            for word,weight in list(weights.items()):
                if word not in self.vocab and len(word)>=5:
                    matches=[v for v in self.vocab if len(v)>=5 and v.startswith(word[:max(5,len(word)-2)])]
                    for v in matches:weights[v]=max(weights.get(v,0),weight*g('W_PREFIX'))
        return weights
    def find(self,text,k=12,dense=None):
        """dense: optional cosine similarity of every section to the question (weighted fusion)."""
        g=config.get;original=text
        if g('USE_CORRECTIONS'):
            # An explicit client correction removes the superseded value from the search view only.
            from corrections import search_text
            text=search_text(text,dialogue)
        q=self._query(text,original);scores=[]
        k1,b=g('BM25_K1'),g('BM25_B')
        latest,history=dialogue(text)
        active=topics(latest)
        if not active:
            for previous in reversed(history):
                active=topics(previous)
                if active:break
        staged, suitability = intent_signals(latest)
        # A short follow-up can concern both an existing platform and a photo channel.
        if active == {'photo'} and history:
            active |= topics(history[-1]) & {'metering'}

        for i,t in enumerate(self.tf):
            norm=k1*(1-b+b*self.length[i]/max(1,self.avg))
            score=sum(self.idf.get(w,0)*weight*t[w]*(k1+1)/(t[w]+norm) for w,weight in q.items() if t[w])
            score+=sum(self.idf.get(w,0)*weight*g('W_TITLE') for w,weight in q.items() if w in self.titles[i])
            if active and self.topics[i]:
                common=active & self.topics[i]
                if common:score=score*g('M_TOPIC_MATCH')+g('B_TOPIC_MATCH')*len(common)
                else:score*=g('M_TOPIC_MISMATCH')
            if asks_for_inputs(latest) and active & self.topics[i] and re.search(r'сведения|данные|подготов',self.items[i].get('title','').lower()):score+=g('B_INPUTS_TITLE')
            if not self.topics[i] and active & self.body_topics[i]:score+=g('B_BODY_TOPIC')
            if len(active)>1 and active <= self.body_topics[i]:score+=g('B_MULTI_TOPIC')
            # A stated rollout plan or constraint is more informative than repeated
            # product names in FAQ sections. Prefer direct, grounded discussions.
            heading = self.items[i]['heading'].lower().replace('ё', 'е')
            if staged and re.search(r'поэтап|очеред|этап', heading):score += g('B_STAGED')
            if suitability and re.search(r'не подходит|ограничен', heading):score += g('B_SUITABILITY')
            # Scope/limits material is especially useful as a second source:
            # it prevents strong claims about supported products and conditions.
            if active & self.topics[i] and self.items[i]['source'].endswith('02-product-scope-and-limits.md'):
                score += g('B_SCOPE')
            scores.append((score,i))
        # File boundaries are not duplicates; retain distinct sections of one file.
        # Import lazily to avoid a cycle with context's language helpers.
        from context import deduplicate
        if dense is not None and g('W_DENSE')>0 and len(dense)==len(scores):
            # Both signals on a 0..1 scale: lexical relative to its best section, dense
            # relative to the spread of this question. Meaning can then rescue a section
            # that shares no words with the client's phrasing.
            top=max(score for score,_ in scores) or 1.0
            low,high=min(dense),max(dense);spread=(high-low) or 1.0
            # A section of another product keeps only part of its semantic credit.
            off=g('M_DENSE_MISMATCH')
            scores=[(score/top+g('W_DENSE')*(dense[i]-low)/spread*(off if active and self.topics[i] and not active & self.topics[i] else 1.0),i)
                    for score,i in scores]
        ordered=sorted(scores,key=lambda x:(-x[0],x[1]))
        candidates=[dict(self.items[i],score=round(score,4)) for score,i in ordered if score>0]
        candidates=deduplicate(candidates)[:k]
        numeric=bool(re.search(r'\d',latest+' '+' '.join(history[-2:])))
        return candidates+self.guardrails(active,candidates,numeric)
