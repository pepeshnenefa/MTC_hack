"""Local retrieval from runtime knowledge documents; no embedded corpus or answers."""
import math
import re
from collections import Counter

STOP=set('и в во на с со к по от до у а но или для это как что мы вы он она они не ли бы из за при есть нас вас можно нужно надо сначала потом вообще будет стоит только уже сейчас отдельно ваш вашей вашем который чтобы если нам мне то все так чем того сможете хочу хотим какой какая какие почему сколько такой таких такой этого нашем наших больше почти пока ли либо'.split())
# Linguistic aliases, not product facts. They cannot assert product capabilities.
ALIASES=[('симк','сим карт'),('sim','сим карт'),('фасад','вертикальн поверхност'),
         ('фото','фотограф распознаван'),('снимок','фотограф'),('магазин','ретейл'),
         ('двухтариф','многотариф тариф'),('выпуск','производств'),
         ('расшир','наращ поэтап'),('провер','аудит'),('вечер','ночью дневное нагрузка'),
         ('оценк','расчет данные'),('стоим','цен расчет'),('понадоб','сведения данные'),('потребу','сведения данные'),
         ('провер','существующая система учет'),('невыгод','нецелесообразн')]

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
    def _query(self,text):
        latest,history=dialogue(text)
        weights=Counter(tokens(latest))
        weights={w:min(n,2) for w,n in weights.items()}
        # History carries referents, but the last user request remains primary.
        for age,previous in enumerate(reversed(history[-3:])):
            for w in set(tokens(previous)):
                weights[w]=max(weights.get(w,0),.4/(age+1))
        if asks_for_inputs(latest):
            for w in tokens('сведения данные количество приборы схема сбор'):
                weights[w]=max(weights.get(w,0),.6)
        text_for_alias=' '.join(words(latest+' '+' '.join(history[-2:])))
        for prefix,aliases in ALIASES:
            if any(w.startswith(prefix) for w in text_for_alias.split()):
                for w in tokens(aliases):weights[w]=max(weights.get(w,0),.35)
        # Recover inflection/derivation matches only when an exact stem is absent.
        for word,weight in list(weights.items()):
            if word not in self.vocab and len(word)>=5:
                matches=[v for v in self.vocab if len(v)>=5 and v.startswith(word[:max(5,len(word)-2)])]
                for v in matches:weights[v]=max(weights.get(v,0),weight*.65)
        return weights
    def find(self,text,k=12):
        q=self._query(text);scores=[]
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
            score=sum(self.idf.get(w,0)*weight*t[w]*2.2/(t[w]+1.2*(.65+.35*self.length[i]/max(1,self.avg))) for w,weight in q.items() if t[w])
            score+=sum(self.idf.get(w,0)*weight*.35 for w,weight in q.items() if w in self.titles[i])
            if active and self.topics[i]:
                common=active & self.topics[i]
                if common:score=score*1.35+4*len(common)
                else:score*=.4
            if asks_for_inputs(latest) and active & self.topics[i] and re.search(r'сведения|данные|подготов',self.items[i].get('title','').lower()):score+=8
            if not self.topics[i] and active & self.body_topics[i]:score+=2
            if len(active)>1 and active <= self.body_topics[i]:score+=5
            # A stated rollout plan or constraint is more informative than repeated
            # product names in FAQ sections. Prefer direct, grounded discussions.
            heading = self.items[i]['heading'].lower().replace('ё', 'е')
            if staged and re.search(r'поэтап|очеред|этап', heading):score += 8
            if suitability and re.search(r'не подходит|ограничен', heading):score += 4
            # Scope/limits material is especially useful as a second source:
            # it prevents strong claims about supported products and conditions.
            if active & self.topics[i] and self.items[i]['source'].endswith('02-product-scope-and-limits.md'):
                score += 0.9
            scores.append((score,i))
        # Avoid flooding the prompt with near-duplicate sections from a single
        # product file when several documents provide complementary evidence.
        ordered = sorted(scores,key=lambda x:(-x[0],x[1]))
        selected, per_source = [], Counter()
        per_file_limit = max(4, (2*k+2)//3) if k >= 8 else k
        for score, index in ordered:
            if score<=0:break
            source=self.items[index]['source']
            if per_source[source]>=per_file_limit:continue
            selected.append(dict(self.items[index],score=round(score,4)))
            per_source[source]+=1
            if len(selected)==k:break
        return selected
