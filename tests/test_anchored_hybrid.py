import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'submission'))
from hybrid import HybridSearch
from context import select_sources,pack_context

class Lexical:
 def __init__(self):
  self.items=[{'source':str(i)+'.md','heading':'Раздел '+str(i),'title':'Продукт / Раздел '+str(i),'text':'Отдельное доказательство '+str(i)} for i in range(20)]
 def find(self,text,k=12):return self.items[:k]

class AnchoredTests(unittest.TestCase):
 def make(self,policy):
  lexical=Lexical()
  # Dense ranking deliberately disagrees with lexical ranking.
  xs=[0.,1.]+[1-i/30 for i in range(2,20)]
  docs=[[x,(1-x*x)**.5] for x in xs]
  return HybridSearch(lexical,docs,{'q':[1.,0.]},policy=policy)
 def test_rrf_can_change_first_evidence(self):
  search=self.make('rrf')
  self.assertNotEqual(search.find('q',36)[0]['source'],'0.md')
 def test_anchored_retains_entire_primary_set_and_order(self):
  search=self.make('anchored');found=search.find('q',36)
  selected=select_sources('q',found,12)
  primary=[s for s in selected if not s.get('semantic_supplement')]
  self.assertEqual([s['source'] for s in primary],[str(i)+'.md' for i in range(12)])
  self.assertEqual(len([s for s in selected if s.get('semantic_supplement')]),2)
 def test_baseline_evidence_reaches_actual_prompt(self):
  search=self.make('anchored')
  context,report=pack_context('q',search.find('q',36),limit=12)
  self.assertEqual(len(report['included']),14)
  for c in search.lexical.find('q',12):self.assertIn(c['text'],context)
 def test_small_corpus_no_duplicate_supplements(self):
  search=self.make('anchored');search.anchor_k=20
  self.assertFalse(any(s.get('semantic_supplement') for s in search.find('q',36)))

if __name__=='__main__':unittest.main()
