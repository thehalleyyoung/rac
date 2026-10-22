import tempfile, unittest, json, os
from unittest.mock import patch
from pathlib import Path
import numpy as np
from run import Provider, gate, select, rac

class ContractTests(unittest.TestCase):
    def test_valid_finance_is_not_sufficient_for_family_or_style(self):
        row={'family':'news'}; label={'valid':True,'style_fit':5,'family':'news'}
        self.assertTrue(gate(row,label))
        self.assertFalse(gate(row,dict(label,valid=False)))
        self.assertFalse(gate(row,dict(label,family='other')))
        self.assertFalse(gate(row,dict(label,style_fit=3)))
    def test_spend_and_token_limits_prevent_request(self):
        with tempfile.TemporaryDirectory() as d:
            p=Provider(Path(d),'unused',0)
            with self.assertRaises(RuntimeError):p.reserve({'input':'test','max_output_tokens':100})
            p.budget=5;p.tokens=500000
            with self.assertRaises(RuntimeError):p.reserve({'input':'test','max_output_tokens':100})
            p.tokens=0;p.calls=80
            with self.assertRaises(RuntimeError):p.reserve({'input':'test'})
    def test_shared_budget_cannot_reset_between_stages(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Path(d)/'budget.json'
            ledger.write_text(json.dumps({'estimated_usd':4.999999,'tokens':1,'calls':1}))
            with patch.dict(os.environ,{'RQ01_BUDGET_LEDGER':str(ledger)}):
                new_stage=Provider(Path(d),'unused',5)
                with self.assertRaises(RuntimeError):
                    new_stage.reserve({'input':'text','max_output_tokens':100})
    def test_selector_rejects_copy_invalid_and_semantic_duplicate(self):
        refs=[{'id':'a','prompt':'original question'}]
        rows=[{'id':str(i),'family':'a','prompt':s} for i,s in enumerate(['original question','off topic','first new','paraphrase','different new'])]
        labels=[{'valid':True,'style_fit':5,'family':'a'} for _ in rows];labels[1]['valid']=False
        E=np.array([[0,0,1],[1,0,0],[1,0,0],[1,0,0],[0,1,0]],dtype=float)
        chosen,decisions=select(rows,labels,E,refs,np.array([[0,0,1]]),quota=3)
        self.assertEqual(len(chosen),2)
        self.assertEqual({d['decision'] for d in decisions},{'exact_reference_copy','filter','selected','near_duplicate'})
    def test_rac_coverage_and_packing_reach_distinct_controls(self):
        E=np.array([[1.,0],[1.,0],[0,1],[-1,0]])
        chosen=rac.coverage_levels(E,3,eps=.1)
        self.assertEqual(len(chosen),3)
        self.assertEqual(set(chosen)&{0,1},{0})
        self.assertEqual(len(rac.farthest_levels(E,4)),4)

if __name__=='__main__':unittest.main()
