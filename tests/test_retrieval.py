import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
from retrieval import Search,chunks,dialogue,topics

class RetrievalTests(unittest.TestCase):
 def test_heading_ancestors_and_portable_source(self):
  data=chunks([('folder\\solar.md','# Солнечные станции\n## Стоимость\n### Условия\nИндивидуальный расчёт.')])
  self.assertEqual(data[0]['title'],'Солнечные станции / Стоимость / Условия')
  self.assertEqual(data[0]['source'],'folder/solar.md')
 def test_long_section_tail_is_indexed(self):
  text='Вводный текст. '*1200+'\nПоследний редкий термин абракадабра.'
  found=Search(chunks([('a.md','# Документ\n'+text)])).find('абракадабра',3)
  self.assertTrue(any('Последний редкий термин' in c['text'] for c in found))
  self.assertTrue(all(len(c['text'])<=6500 for c in found))
 def test_table_and_qualification_stay_together(self):
  text='| Цена | Значение |\n| --- | --- |\n| Ориентир | 100 |\n\nЭто не гарантия.'
  self.assertEqual(chunks([('a.md','# Цена\n'+text)])[0]['text'],text)
 def test_explicit_topic_switch(self):
  docs=[('s.md','# Солнечные панели\n## Функции\nМонтаж панелей на кровле.'),('p.md','# Распознавание фотографий\n## Функции\nСерийный номер счётчика по фотографии.')]
  question='Пользователь: Интересуют солнечные панели на кровле.\nАссистент: Какой регион?\nПользователь: Теперь про фото: считываете серийный номер прибора?'
  self.assertEqual(Search(chunks(docs)).find(question,1)[0]['source'],'p.md')
 def test_context_and_corrections_preserved(self):
  latest,history=dialogue('Пользователь: Мощность 800 кВт.\nАссистент: Вы подходите.\nПользователь: Ошибся: 500 кВт.')
  self.assertEqual(latest,'Ошибся: 500 кВт.');self.assertNotIn('Вы подходите.',' '.join(history))
 def test_no_evidence_for_unknown_vocabulary(self):
  self.assertEqual(Search(chunks([('a.md','# Панели\nМонтаж на крыше.')])).find('динозавр тираннозавр'),[])

# Generalizable ranking regressions. These are source-retrieval tests, not
# attempts to reproduce the 18 official sample answers.
class ConsultationRetrievalTests(unittest.TestCase):
 def test_phased_rollout_with_more_specific_document_competition(self):
  docs=[('solar.md','# Станция\n## Целевой диапазон мощности\nСолнечная станция допускает расширение.'),
        ('general.md','# Общие вопросы\n## Можно ли внедрять поэтапно\nНачать с части оборудования и постепенно наращивать мощность.'),
        ('extra.md','# Мощность\n## Оборудование\nСтанция, мощность, дополнительные вопросы.')]
  results=Search(chunks(docs)).find('Можно ли сначала поставить станцию, а потом расширить?',k=3)
  self.assertIn('general.md',[s['source'] for s in results])
 def test_constraints_take_priority_for_suitability_question(self):
  docs=[('a.md','# Солнечные станции\n## Оборудование\nСолнечная станция и крыша.'),
        ('b.md','# Солнечные станции\n## Не подходит, если\nНекоторые профили потребления ограничивают экономическую эффективность.'),
        ('c.md','# Станции\n## Возможности\nСолнечные панели поддерживаются.')]
  sources=Search(chunks(docs)).find('А сетевая солнечная станция вообще имеет смысл?',k=3)
  self.assertTrue(any(s['source']=='b.md' for s in sources))

if __name__=='__main__':unittest.main()
