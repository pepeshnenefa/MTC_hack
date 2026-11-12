"""Controlled mechanism check; synthetic similarities are NOT bge-m3 evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dist/energytool-unified-v1-hybrid.zip'
SHA='8c8301c401f86b19295a095a853aa37ea0aa7d93e9402a8fc2c42af58ff836d5'
OUTPUT=ROOT/'dist/energytool-unified-v1-hybrid-semantic07.zip'


def main(build=False, baseline=1.0, candidate=0.7):
    started=time.monotonic()
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==SHA
    sys.path.insert(0,str(BASE))
    import config
    import retrieval
    import context
    assert str(BASE) in config.__file__ and str(BASE) in retrieval.__file__
    docs=[('solar.md','# Солнечные панели\n## Возможности\nОписание первого решения.'),
          ('wholesale.md','# Оптовый рынок\n## Возможности\nОписание второго решения.'),
          ('metering.md','# Автоматический учет\n## Возможности\nОписание третьего решения.'),
          ('general.md','# Общая информация\n## Возможности\nОбщие сведения.')]
    cases=[('explicit','Расскажите про солнечные панели.',{'solar'}),
           ('comparison','Сравните солнечные панели и оптовый рынок.',{'solar','wholesale'}),
           ('unspecified','Какое решение мне выбрать?',set()),
           ('followup','Пользователь: Солнечные панели.\nПользователь: Какие условия?',{'solar'}),
           ('topic_switch','Пользователь: Солнечные панели.\nПользователь: Теперь оптовый рынок.',{'wholesale'})]
    results=[]
    with config.override(**config.DEFAULTS):
        assert config.get('USE_HYBRID')==1 and config.get('W_COLLOQUIAL')==0.5
        index=retrieval.Search(retrieval.chunks(docs))
        # Explicitly controlled values, deliberately favour the competing product.
        dense=[0.4,0.6,0.2,0.1]
        assert len(index.items)==len(dense)
        for name,question,expected_active in cases:
            latest,history=retrieval.dialogue(question)
            active=retrieval.topics(latest)
            if not active:
                for previous in reversed(history):
                    active=retrieval.topics(previous)
                    if active:break
            assert active==expected_active,(name,active)
            ranked=[]
            lexical=[]
            for factor in (baseline,candidate):
                with config.override(M_DENSE_MISMATCH=factor):
                    ranked.append(index.find(question,k=100,dense=dense))
                    lexical.append(index.find(question,k=100))
            assert lexical[0]==lexical[1], 'Fallback lexical search changed'
            by_source=[{s['source']:s['score'] for s in found} for found in ranked]
            changed=[]
            for i,source in enumerate(index.items):
                key=source['source']
                penalized=bool(active and index.topics[i] and not active & index.topics[i])
                penalty=(baseline-candidate)*config.get('W_DENSE')*(dense[i]-min(dense))/(max(dense)-min(dense)) if penalized else 0
                if key in by_source[0] and key in by_source[1]:
                    delta=by_source[0][key]-by_source[1][key]
                    assert abs(delta-penalty)<0.0002,(name,key,delta,penalty)
                    if abs(delta)>0.0002:changed.append(key)
            if name=='unspecified':assert ranked[0]==ranked[1]
            if name in ('explicit','followup'):
                assert ranked[1][0]['source']=='solar.md'
                if baseline==1.0:assert ranked[0][0]['source']=='wholesale.md'
            if name=='comparison':
                for key in ('solar.md','wholesale.md'):
                    assert by_source[0][key]==by_source[1][key]
            results.append(dict(case=name,penalized=changed,
                baseline_order=[s['source'] for s in ranked[0]],
                candidate_order=[s['source'] for s in ranked[1]],passed=True))
        # A narrow spread is still amplified by min-max normalization: not fixed here.
        with config.override(M_DENSE_MISMATCH=candidate):
            question=cases[0][1]
            first=index.find(question,k=100,dense=dense)
            compressed=index.find(question,k=100,dense=[0.5+v*0.00001 for v in dense])
            assert first==compressed
            strong=index.find(question,k=100,dense=[0.4,0.9,0.6,0.2])
            assert any(s['source']=='wholesale.md' for s in strong)  # Not a product filter.
    report=dict(kind='synthetic-mechanism-only',base_sha256=SHA,
                only_change={'M_DENSE_MISMATCH':[baseline,candidate]},cases=results,
                lexical_fallback_unchanged=True,comparison_topics_preserved=True,
                small_spread_amplification_still_present=True,
                real_embedding_quality_measured=False,model_calls=0,
                seconds=round(time.monotonic()-started,3))
    if build:
        manifest=OUTPUT.with_suffix('.manifest.json')
        if OUTPUT.exists() or manifest.exists():raise ValueError('Refusing to overwrite artifact')
        with zipfile.ZipFile(BASE) as z:
            content={n:z.read(n) for n in z.namelist()}
        anchor=("'M_DENSE_MISMATCH': "+str(baseline)).encode()
        assert content['config.py'].count(anchor)==1
        content['config.py']=content['config.py'].replace(anchor,("'M_DENSE_MISMATCH': "+str(candidate)).encode())
        for n,data in content.items():
            assert '/' not in n and (n.endswith('.py') or n=='requirements.txt')
            if n.endswith('.py'):compile(data,n,'exec')
        with zipfile.ZipFile(OUTPUT,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for n,data in content.items():
                info=zipfile.ZipInfo(n,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(info,data)
        with zipfile.ZipFile(OUTPUT) as z,zipfile.ZipFile(BASE) as base:
            assert z.testzip() is None and z.namelist()==base.namelist()
            assert [n for n in z.namelist() if z.read(n)!=base.read(n)]==['config.py']
        metadata=dict(archive=OUTPUT.name,base_sha256=SHA,sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            default_overrides={'M_DENSE_MISMATCH':candidate},
            files={n:hashlib.sha256(b).hexdigest() for n,b in content.items()},
            status='prepared_not_submitted',score=None)
        manifest.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        report['artifact']=metadata
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true')
    main(parser.parse_args().build)
