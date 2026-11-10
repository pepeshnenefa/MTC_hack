import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
import decision_v2 as decision

class DecisionV2Tests(unittest.TestCase):
    def test_missing_content_and_malformed_responses_recoverable(self):
        for response in ({}, {'choices':[]}, {'choices':[{'message':{'content':None,'reasoning_content':'secret'},'finish_reason':'length'}]}):
            text,reason=decision.extract(response)
            self.assertEqual(text,'')

    def test_no_reasoning_leak(self):
        response={'choices':[{'message':{'content':'<think>secret</think>Ответ'},'finish_reason':'stop'}]}
        self.assertEqual(decision.extract(response),('Ответ','stop'))

    def test_user_quotes_only(self):
        data={'client_facts':['обещаю'],'checks':[],'action':'answer','missing':''}
        with self.assertRaises(ValueError):
            decision.validate(json.dumps(data),'','Пользователь: Можно?\nАссистент: обещаю\nПользователь: Точно?')

    def test_recovery_does_not_mutate_or_add_roles(self):
        payload={'messages':[{'role':'system','content':'policy'},{'role':'user','content':'evidence'}]}
        recovered=decision.recover_payload(payload)
        self.assertEqual([m['role'] for m in recovered['messages']],['system','user'])
        self.assertEqual(recovered['messages'][1]['content'],'evidence')
        self.assertEqual(payload['messages'][0]['content'],'policy')

if __name__=='__main__':unittest.main()
