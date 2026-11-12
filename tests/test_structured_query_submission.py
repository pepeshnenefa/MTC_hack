import ast
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sq_builder',ROOT/'tools/build_structured_query_submission.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


@contextlib.contextmanager
def loaded():
    content=builder.files()
    names=('batch_io','retrieval','context','structured_query','solution')
    previous={n:sys.modules.pop(n,None) for n in names}
    try:
        for name in names:
            module=types.ModuleType(name);module.__file__=str(ROOT/'submission'/(name+'.py'))
            sys.modules[name]=module
            exec(compile(content[name+'.py'],module.__file__,'exec'),module.__dict__)
        yield sys.modules['solution'],sys.modules['structured_query']
    finally:
        for name in names:
            sys.modules.pop(name,None)
            if previous[name] is not None:sys.modules[name]=previous[name]


def task():
    return dict(goal='Какие условия для склада?',facts=[dict(subject='склада',field='quantity',
        value='12 объектов',turn=1,quote='Для склада 12 объектов')],queries=['Ограничения'])


class StructuredTests(unittest.TestCase):
    def setUp(self):
        self.environment=patch.dict(os.environ,{'LLM_BASE_URL':'http://test.invalid',
            'LLM_API_KEY':'test','LLM_MODEL':'test-model','USE_RERANKER':'0'},clear=True)
        self.environment.start();self.addCleanup(self.environment.stop)
        self.modules=loaded();self.solution,self.sq=self.modules.__enter__()
        self.addCleanup(self.modules.__exit__,None,None,None)
        self.api=self.solution.API(deadline=time.monotonic()+120)

    def test_archive_baseline_modules_and_prompt(self):
        content=builder.files()
        self.assertEqual(len(content),6)
        with zipfile.ZipFile(ROOT/'dist/energytool-context-v1.zip') as z:
            for name in ('retrieval.py','context.py','batch_io.py','requirements.txt'):
                self.assertEqual(content[name],z.read(name))
            def system(data):
                return next(ast.literal_eval(n.value) for n in ast.parse(data).body
                    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets))
            self.assertEqual(system(content['solution.py']),system(z.read('solution.py')))

    def test_user_roles_and_numbering(self):
        self.assertEqual(self.sq.user_turns('User: Первый\nAssistant: Придумано\nUser: Второй'),['Первый','Второй'])
        self.assertEqual(self.sq.user_turns('Assistant: Только ассистент'),[])

    def test_verified_facts_affect_query(self):
        turns=['Для склада 20 объектов','Для склада 12 объектов']
        parsed=self.sq.parse(json.dumps(task()),turns)
        query=self.sq.queries(parsed)[0]
        self.assertIn('12 объектов',query);self.assertNotIn('20 объектов',query)
        self.assertEqual(len(self.sq.queries(parsed)),2)

    def test_unverified_fields_and_ambiguous_values_rejected(self):
        turns=['Для склада 20 объектов','Для склада 12 объектов']
        for field,value in [('turn',True),('turn',9),('value','200 объектов'),('subject','завода'),
                            ('quote','Выдумано'),('field','invented')]:
            data=task();data['facts'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.sq.parse(json.dumps(data),turns)
        data=task();data['facts'].append(dict(subject='склада',field='quantity',value='20 объектов',
                                            turn=0,quote=turns[0]))
        with self.assertRaises(ValueError):self.sq.parse(json.dumps(data),turns)

    def test_different_objects_not_overwritten(self):
        turns=['Для склада 12 объектов','Для завода 20 объектов']
        data=dict(goal='Сравнение',queries=[],facts=[
            dict(subject='склада',field='quantity',value='12 объектов',turn=0,quote=turns[0]),
            dict(subject='завода',field='quantity',value='20 объектов',turn=1,quote=turns[1])])
        self.assertEqual(len(self.sq.parse(json.dumps(data),turns)['facts']),2)

    def test_bad_json_and_null_rejected(self):
        for text in (None,'{}','[]','not json','<think>unfinished'):
            with self.subTest(text=text),self.assertRaises((ValueError,TypeError)):
                self.sq.parse(text,['Вопрос'])

    def test_plan_http_failure_and_budget(self):
        before=self.api.deadline
        with patch.object(self.sq,'urlopen',side_effect=TimeoutError()) as call:
            self.assertIsNone(self.sq.plan(self.api,'Вопрос'))
        self.assertEqual(call.call_count,1);self.assertEqual(self.api.deadline,before)
        self.api.deadline=time.monotonic()+10
        with patch.object(self.sq,'urlopen') as call:
            self.assertIsNone(self.sq.plan(self.api,'Вопрос'))
        call.assert_not_called()

    def test_failed_plan_retains_original_search(self):
        pool=[dict(source='a',heading='a',text='a')]
        search=types.SimpleNamespace(find=lambda question,k:pool)
        with patch.object(self.sq,'plan',return_value=None),patch.object(self.api,'answer',return_value='Ответ') as call:
            self.assertEqual(self.sq.answer(self.api,search,'Исходный вопрос',36),'Ответ')
        self.assertEqual(call.call_args.args,('Исходный вопрос',pool))

    def test_additions_preserve_context_and_do_not_leak_plan(self):
        pool=[dict(source=str(i),heading='Раздел',title='Раздел',text='Текст '+str(i),score=20-i) for i in range(15)]
        search=types.SimpleNamespace(find=lambda question,k:pool if question=='Исходный вопрос' else [pool[14]])
        response={'choices':[{'message':{'content':'Ответ.'},'finish_reason':'stop'}]}
        with patch.object(self.sq,'plan',return_value=dict(goal='Секретный план',facts=[],queries=[])), \
             patch.object(self.api,'post',return_value=response) as call:
            self.sq.answer(self.api,search,'Исходный вопрос',36)
        packed=sys.modules['context'].pack_context('Исходный вопрос',pool)[0]
        user=call.call_args.args[1]['messages'][1]['content']
        self.assertIn(packed,user);self.assertIn('Текст 14',user)
        self.assertNotIn('Секретный план',user);self.assertTrue(user.endswith('Исходный вопрос'))

    def test_context_budget_never_truncates_base(self):
        base='x'*39999
        extra=[dict(source='a',heading='a',text='new')]
        self.assertEqual(self.sq.append_context(base,extra),base)

    def test_no_plan_state_between_sessions(self):
        with patch.object(self.sq,'urlopen',side_effect=TimeoutError()) as call:
            self.sq.plan(self.api,'Первая сессия');self.sq.plan(self.api,'Вторая сессия')
        payload=json.loads(call.call_args.args[0].data)
        self.assertNotIn('Первая сессия',payload['messages'][1]['content'])


if __name__=='__main__':unittest.main()
