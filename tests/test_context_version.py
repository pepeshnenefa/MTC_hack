import json,os,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
from context import deduplicate,pack_context,select_sources
from retrieval import Search
import solution

BASE={'LLM_BASE_URL':'http://localhost.invalid','LLM_API_KEY':'synthetic','LLM_MODEL':'synthetic'}
def source(i,text,title='Солнечные станции / Описание',file=None):
 return {'source':file or str(i)+'.md','heading':title.split(' / ')[-1],'title':title,'text':text}

class ContextVersionTests(unittest.TestCase):
 def test_unique_sections_same_file_not_capped(self):
  items=[source(i,'спецификация '+str(i),file='one.md') for i in range(10)]
  self.assertEqual(len(Search(items).find('спецификация',12)),10)
 def test_duplicates_replaced_by_distinct_evidence(self):
  items=[source(1,'Одинаковое описание'),source(2,'Одинаковое описание'),source(3,'Другое условие')]
  self.assertEqual(len(deduplicate(items)),2)
 def test_numeric_and_negative_differences_preserved(self):
  body='Подключение системы по индивидуальному проекту. '*20
  items=[source(1,body+'Мощность от 100 кВт.'),source(2,body+'Мощность от 200 кВт.'),
         source(3,body+'Подключение не поддерживается.')]
  self.assertEqual(len(deduplicate(items)),3)
 def test_almost_identical_changed_condition_not_removed(self):
  body='Общее описание сервиса и подключения. '*30
  items=[source(1,body+'Мощность от 100 кВт.'),source(2,body+'Мощность до 100 кВт.'),
         source(3,body+'Гарантия 100 дней.'),source(4,body+'Стоимость 100 рублей.')]
  self.assertEqual(len(deduplicate(items)),4)
 def test_layout_duplicate_removed(self):
  self.assertEqual(len(deduplicate([source(1,'Текст: описание услуги.'),source(2,'Текст — описание услуги!')])),1)
 def test_different_products_not_merged(self):
  items=[source(1,'Подключение по договору.',title='Фото / Цена'),source(2,'Подключение по договору.',title='Учёт / Цена')]
  self.assertEqual(len(deduplicate(items)),2)
 def test_qualifications_reserved_and_unrelated_not_added(self):
  sources=[source(i,'Описание '+str(i)) for i in range(12)]
  sources += [source(20,'Важная оговорка',title='Солнечные станции / Не подходит, если'),
              source(21,'Чужая оговорка',title='Распознавание фото / Ограничения')]
  selected=select_sources('Подходят солнечные панели?',sources,12)
  self.assertIn(sources[12],selected);self.assertNotIn(sources[13],selected)
  self.assertEqual(selected[:3],sources[:3]);self.assertEqual(len(selected),13)
  self.assertTrue(all(s in selected for s in sources[:12]))
 def test_packing_reports_drop_and_counts_separator(self):
  items=[source(1,'a'*100),source(2,'b'*100)]
  context,report=pack_context('q',items,max_chars=150)
  self.assertEqual(len(report['included']),1);self.assertEqual(len(report['dropped']),1)
  self.assertEqual(report['characters'],len(context));self.assertLessEqual(len(context),150)
 def run_answers(self,replies,deadline=None):
  captured=[]
  with patch.dict(os.environ,BASE,clear=True):
   api=solution.API(deadline=deadline)
   def post(endpoint,body):
    captured.append(body)
    item=replies[len(captured)-1]
    if isinstance(item,Exception):raise item
    return item
   api.post=post
   answer=api.answer('Вопрос',[source(1,'Условия')])
  return answer,captured
 def reply(self,text,reason='stop'):
  return {'choices':[{'message':{'content':text},'finish_reason':reason}]}
 def test_complete_answer_needs_one_call(self):
  answer,calls=self.run_answers([self.reply('Да, при условии.')])
  self.assertEqual(len(calls),1);self.assertEqual(answer,'Да, при условии.')
 def test_truncated_recovered_once(self):
  answer,calls=self.run_answers([self.reply('Можно, но только если','length'),self.reply('Можно при выполнении условий.')])
  self.assertEqual(answer,'Можно при выполнении условий.');self.assertEqual(len(calls),2)
  self.assertEqual(calls[0]['messages'][1],calls[1]['messages'][1])
  self.assertNotIn('Можно, но только если',json.dumps(calls[1],ensure_ascii=False))
 def test_failed_recovery_retains_original(self):
  answer,calls=self.run_answers([self.reply('Полезный фрагмент','length'),TimeoutError('synthetic')])
  self.assertEqual(answer,'Полезный фрагмент');self.assertEqual(len(calls),2)
 def test_second_truncation_does_not_replace_original(self):
  answer,calls=self.run_answers([self.reply('Первый полезный ответ','length'),self.reply('Другой обрыв','length')])
  self.assertEqual(answer,'Первый полезный ответ');self.assertEqual(len(calls),2)
 def test_short_deadline_skips_recovery(self):
  answer,calls=self.run_answers([self.reply('Первый ответ','length')],time.monotonic()+5)
  self.assertEqual(len(calls),1)
 def test_reasoning_not_exposed_and_empty_can_recover(self):
  answer,calls=self.run_answers([self.reply('<think>secret reasoning','length'),self.reply('Законченный ответ')])
  self.assertEqual(answer,'Законченный ответ');self.assertNotIn('secret',answer)
 def test_no_usable_answer_is_error(self):
  with self.assertRaises(ValueError):self.run_answers([self.reply('','length'),self.reply('','length')])

if __name__=='__main__':unittest.main()
