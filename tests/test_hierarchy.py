import sys,unittest,tempfile,os
from unittest.mock import patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'submission'))
from hierarchy import HierarchicalSearch
from context import pack_context

DOCS=[('20-products/23-solar.md','# Солнечные станции\n## Монтаж\nПанели на кровле.\n## Ограничения\nСущественная оговорка в конце.'),
('20-products/22-askue.md','# Облачная платформа учёта\n## Возможности\nУчёт по арендаторам и цехам.'),
('20-products/25-meter-reading.md','# Распознавание по фотографии\n## Возможности\nФото приборов, распознавание номера.'),
('10-energy/reference.md','# Энергетика\n## Реактивная энергия\nСначала нужны измерения.')]

class HierarchyTests(unittest.TestCase):
 def test_complete_product_document_retains_last_qualification(self):
  search=HierarchicalSearch(DOCS);found=search.find('Солнечные панели на кровлю')
  whole=[c for c in found if c.get('whole_document')]
  self.assertEqual(len(whole),1);self.assertEqual(whole[0]['text'],DOCS[0][1])
  context,_=pack_context('Солнечные панели на кровлю',found)
  self.assertIn('Существенная оговорка в конце.',context)
 def test_no_unrelated_product_material(self):
  found=HierarchicalSearch(DOCS).find('Солнечные панели на кровлю')
  self.assertFalse(any(c['source'].endswith('25-meter-reading.md') for c in found))
 def test_general_energy_question_stays_with_lexical(self):
  search=HierarchicalSearch(DOCS);q='Нужно измерить реактивную энергию?'
  self.assertEqual(search.route(q)[0],[]);self.assertEqual(search.find(q),search.lexical.find(q))
 def test_two_products_in_one_question(self):
  search=HierarchicalSearch(DOCS)
  selected=search.route('Платформа учёта по арендаторам и распознавание по фото')[0]
  self.assertEqual(set(selected),{'20-products/22-askue.md','20-products/25-meter-reading.md'})
 def test_explicit_topic_switch(self):
  q='Пользователь: Хотим солнечные панели на кровлю.\nПользователь: Теперь другой вопрос: распознавание по фотографии.'
  self.assertEqual(HierarchicalSearch(DOCS).route(q)[0],['20-products/25-meter-reading.md'])
 def test_followup_uses_history(self):
  q='Пользователь: Хотим солнечные панели.\nПользователь: Каковы ограничения?'
  self.assertEqual(HierarchicalSearch(DOCS).route(q)[0],['20-products/23-solar.md'])
 def test_large_document_falls_back_instead_of_partial_whole(self):
  docs=[('20-products/23-solar.md','# Солнечные станции\n'+'Панели. '*5000)]
  found=HierarchicalSearch(docs).find('Солнечные панели')
  self.assertFalse(any(c.get('whole_document') for c in found))
  self.assertTrue(found)
 def test_unknown_question_no_forced_product(self):
  search=HierarchicalSearch(DOCS)
  self.assertEqual(search.route('Динозавры мезозоя')[0],[])
 def test_windows_sources_normalized(self):
  docs=[(p.replace('/','\\'),t) for p,t in DOCS]
  self.assertEqual(HierarchicalSearch(docs).route('Солнечные панели')[0],['20-products/23-solar.md'])

class HierarchyEntryPointTests(unittest.TestCase):
 def test_worker_sends_full_document_without_embedding_calls(self):
  import solution
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);kb=root/'kb'/'20-products';kb.mkdir(parents=True)
   full='# Солнечные станции\n## Монтаж\nПанели на кровле.\n## Ограничения\nКонечная важная оговорка.'
   (kb/'23-solar.md').write_text(full,encoding='utf-8')
   (root/'q.csv').write_text('id,question\n1,Солнечные панели на кровле?\n',encoding='utf-8')
   requests=[]
   def post(api,endpoint,body):
    requests.append((endpoint,body));return {'choices':[{'message':{'content':'Тест ответа'},'finish_reason':'stop'}]}
   config={'LLM_BASE_URL':'http://synthetic.invalid','LLM_API_KEY':'synthetic','LLM_MODEL':'synthetic'}
   argv=['solution.py','--worker','--questions',str(root/'q.csv'),'--kb',str(kb.parent),'--output',str(root/'out.csv')]
   with patch.dict(os.environ,config,clear=True),patch.object(sys,'argv',argv),patch.object(solution.API,'post',post):solution.main()
   self.assertEqual(len(requests),1);self.assertEqual(requests[0][0],'/chat/completions')
   self.assertIn(full,requests[0][1]['messages'][1]['content'])

if __name__=='__main__':unittest.main()

