"""Exact semantic07 versus candidate: evidence invariants and generation payload."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('group_builder',ROOT/'tools/build_grouped_context_submission.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


class GroupedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names=('config','batch_io','retrieval','context','grouped_context','corrections','hybrid','hierarchy','solution')
        cls.previous={n:sys.modules.pop(n,None) for n in cls.names}
        with zipfile.ZipFile(builder.BASE) as z:cls.baseline={n:z.read(n) for n in z.namelist()}
        cls.content=builder.files()
        for name in cls.names:
            module=types.ModuleType(name);module.__file__=str(builder.BASE)+'/'+name+'.py'
            sys.modules[name]=module
            exec(compile(cls.content[name+'.py'],name+'.py','exec'),module.__dict__)
        cls.config=sys.modules['config']
        cls.config_scope=cls.config.override(**cls.config.DEFAULTS);cls.config_scope.__enter__()
        cls.candidate=sys.modules['context'];cls.formatter=sys.modules['grouped_context']
        cls.old=types.ModuleType('baseline_context')
        exec(compile(cls.baseline['context.py'],'baseline_context.py','exec'),cls.old.__dict__)
        cls.solution=sys.modules['solution']

    @classmethod
    def tearDownClass(cls):
        cls.config_scope.__exit__(None,None,None)
        for name in cls.names:
            sys.modules.pop(name,None)
            if cls.previous[name] is not None:sys.modules[name]=cls.previous[name]

    def pool(self):
        return [dict(source=p,heading=h,title=t,text=b,score=10-i) for i,(p,h,t,b) in enumerate([
            ('20-products/a.md','Цена','Солнечные панели / Цена','Цена 100 только при условии A.'),
            ('20-products/b.md','Цена','Оптовый рынок / Цена','Цена 200 при условии B.'),
            ('20-products/a.md','Ограничения','Солнечные панели / Ограничения','Не применять к другому типу.'),
            ('00-reference/n.md','Солнечные панели','Числа / Солнечные панели','Ориентир 15–500, не гарантия.'),
            ('20-products/general.md','Общее','Общие вопросы / Общее','Общие сведения.')])]

    def test_only_formatter_and_context_changed(self):
        changed=[n for n in self.baseline if self.baseline[n]!=self.content[n]]
        self.assertEqual(changed,['context.py'])
        self.assertEqual(set(self.content)-set(self.baseline),{'grouped_context.py'})
        self.assertEqual(self.config.get('M_DENSE_MISMATCH'),0.7)

    def test_product_grouping_keeps_references_separate(self):
        original,report=self.old.pack_context('Сравните солнечные панели и оптовый рынок',self.pool())
        text,new=self.formatter.format_context(original,report,40000)
        self.assertTrue(new['grouping_applied'])
        self.assertLess(text.index('Не применять'),text.index('Цена 200'))
        self.assertGreater(text.index('Ориентир'),text.index('Цена 200'))
        self.assertEqual(new['characters'],len(text))
        self.assertEqual(new['selected'],report['selected'])
        self.assertEqual(new['dropped'],report['dropped'])
        for source in report['included']:
            entry=source['source']+' / '+source['title']+'\n'+source['text']
            self.assertEqual(text.count(entry),1)

    def test_tight_budget_falls_back_without_losing_evidence(self):
        original,report=self.old.pack_context('Вопрос',self.pool())
        text,new=self.formatter.format_context(original,report,len(original))
        self.assertEqual(text,original)
        self.assertFalse(new['grouping_applied'])
        self.assertEqual(new['included'],report['included'])

    def test_empty_and_unknown_metadata(self):
        text,report=self.candidate.pack_context('Вопрос',[])
        self.assertEqual(text,'');self.assertEqual(report['included'],[])
        source=dict(source='unknown.txt',heading='Раздел',text='Факт')
        text,report=self.candidate.pack_context('Вопрос',[source])
        self.assertIn('Справочный или иной документ',text)
        self.assertIn('unknown.txt / Раздел\nФакт',text)

    def test_fifty_existing_questions_keep_identical_evidence(self):
        paths=[ROOT/'demo_questions.csv',ROOT/'tests/evaluation/holdout-v1/questions.csv']
        questions=[]
        for path in paths:
            with path.open(encoding='utf-8-sig',newline='') as stream:
                questions.extend(r['question'] for r in csv.DictReader(stream))
        search=self.solution.Search(self.solution.chunks(self.solution.documents(ROOT/'Knowledge-base')))
        applied=0
        for question in questions:
            sources=search.find(question,k=36)
            before,old=self.old.pack_context(question,sources)
            after,new=self.candidate.pack_context(question,sources)
            with self.subTest(question_index=questions.index(question)):
                self.assertCountEqual([self.old.key(s) for s in old['included']],
                                      [self.old.key(s) for s in new['included']])
                self.assertEqual(old['dropped'],new['dropped'])
                self.assertLessEqual(len(after),40000)
                for source in old['included']:
                    self.assertIn(source['source']+' / '+source.get('title',source['heading'])+'\n'+source['text'],after)
            applied+=new['grouping_applied']
        self.assertEqual(len(questions),50)
        self.assertGreater(applied,0)
        print(json.dumps(dict(kind='format-not-retrieval-quality',questions=len(questions),grouped=applied)),flush=True)

    def test_one_generation_original_dialogue_and_no_leak_between_calls(self):
        with patch.dict(os.environ,{'LLM_BASE_URL':'http://test.invalid','LLM_API_KEY':'test','LLM_MODEL':'test-model'},clear=True):
            api=self.solution.API()
            result={'choices':[{'message':{'content':'Ответ.'},'finish_reason':'stop'}]}
            with patch.object(api,'post',return_value=result) as call:
                self.assertEqual(api.answer('Первый диалог',self.pool()),'Ответ.')
                first=call.call_args.args[1]
                self.assertIn('Первый диалог',first['messages'][1]['content'])
                self.assertIn('Продуктовый документ',first['messages'][1]['content'])
                api.answer('Второй диалог',[])
                self.assertNotIn('Первый диалог',call.call_args.args[1]['messages'][1]['content'])
                self.assertEqual(call.call_count,2)


if __name__=='__main__':unittest.main()
