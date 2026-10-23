import copy, json, os, random, tempfile, unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from sampler import load_config,make_plan,sample_one,validate_spec,fingerprint,assign_compatible
from engine import reject_reasons,intervention_specs,select_questions,ReplayProvider,judge,textual_premise_guard,request_form_guard,batch_schema,literal_symbols,render_format,run_bank,generate,read_only_action_guard,literal_instrument

ASOF='2026-10-22T00:00:00Z'
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=load_config()
    def test_same_seed_repeats_and_different_seed_changes(self):
        a=make_plan(self.c,47,64,'four_category_reference',ASOF)
        self.assertEqual(a,make_plan(self.c,47,64,'four_category_reference',ASOF))
        b=make_plan(self.c,48,64,'four_category_reference',ASOF)
        aa={fingerprint(s) for p in a['pools'] for s in p['proposals']}
        bb={fingerprint(s) for p in b['pools'] for s in p['proposals']}
        self.assertLess(len(aa&bb),len(aa)*.02)
    def test_all_32_intents_and_four_categories_preserved(self):
        plan=make_plan(self.c,47,64,'four_category_reference',ASOF)
        self.assertEqual(set(plan['source_quotas'].values()),{2})
        self.assertEqual(set(plan['domain_quotas'].values()),{16})
        for pool in plan['pools']:
            cell=next(c for c in self.c['reference_cells'] if c['source_id']==pool['source_id'])
            for s in pool['proposals']:
                validate_spec(self.c,s)
                self.assertIn(s['variant'],cell['allowed_variants'])
                self.assertEqual(s['domain'],cell['category'])
    def test_many_seeds_both_profiles_and_small_batches(self):
        for seed in range(8):
            for profile,n in [('four_category_reference',32),('four_category_explore',64),('four_category_reference',1)]:
                plan=make_plan(self.c,seed,n,profile,ASOF,pool_size=2)
                self.assertEqual(len(plan['pools']),n)
                for pool in plan['pools']:
                    for spec in pool['proposals']:validate_spec(self.c,spec)
    def test_known_bad_cross_products_fail(self):
        s=sample_one(self.c,random.Random(1),'event_outlook','commodities','open_query',ASOF)
        s.update(variant='weather',symbols=['GOLD'])
        with self.assertRaises(ValueError):validate_spec(self.c,s)
        s=sample_one(self.c,random.Random(1),'token_due_diligence','onchain_spot','open_query',ASOF)
        s['symbols']=['NOT_A_SUPPLIED_TOKEN']
        with self.assertRaises(ValueError):validate_spec(self.c,s)
        s['symbols']=['SI'];s['domain']='equities'
        with self.assertRaises(ValueError):validate_spec(self.c,s)
    def test_evidence_source_is_compatible_with_task(self):
        for seed in range(30):
            s=sample_one(self.c,random.Random(seed),'technical_setup','equities','open_query',ASOF)
            self.assertIn(s['evidence_surface'],[None,'price_volume'])
        s['evidence_surface']='issuer_filings'
        with self.assertRaises(ValueError):validate_spec(self.c,s)
    def test_alias_related_asset_exception_is_narrow(self):
        s=sample_one(self.c,random.Random(2),'relative_value','equities','open_query',ASOF,{'source_id':'eq-mstr-premium','asset_rule':'holdings_proxy','allowed_variants':['asset_nav']})
        self.assertEqual(s['symbols'],['MSTR','BTC']);validate_spec(self.c,s)
        s['asset_rule']='equity_any'
        with self.assertRaises(ValueError):validate_spec(self.c,s)
    def test_infeasible_matching_fails_without_invalid_fallback(self):
        with self.assertRaises(ValueError):assign_compatible(['news'],{'fx':1},lambda _:['equities'],random.Random(1))
    def test_controlled_intervention_changes_one_axis_only(self):
        specs,groups=intervention_specs(self.c,47,ASOF)
        for group in groups:
            chosen=[s for s in specs if s['id'] in group['spec_ids']]
            held=[{k:v for k,v in s.items() if k not in ['id',group['axis']]} for s in chosen]
            self.assertTrue(all(s==held[0] for s in held))
            self.assertEqual(len({s[group['axis']] for s in chosen}),3)
    def label(self,s):
        return dict(valid=True,source_fit=True,unsupported_claim=False,effect_request=False,style_fit=4,reason='synthetic control',symbols=s['symbols'],**{k:s[k] for k in ['task','variant','domain','premise_mode','answer_shape','evidence_demand','difficulty','intent_structure']},**{k:s[k] for k in ['asset_class','instrument'] if k in s})
    def test_hard_fidelity_and_premise_gate_not_just_finance_validity(self):
        s=sample_one(self.c,random.Random(3),'price_move','crypto_majors','open_query',ASOF);row={'prompt':'What is moving '+s['symbols'][0]+' '+s['instrument']+' today?'};l=self.label(s)
        self.assertEqual(reject_reasons(self.c,s,row,l),[])
        for field,value in [('variant','other'),('premise_mode','hypothetical'),('domain','equities'),('unsupported_claim',True),('effect_request',True)]:
            self.assertTrue(reject_reasons(self.c,s,row,dict(l,**{field:value})))
        self.assertIn('outside_domain_catalog',reject_reasons(self.c,s,row,dict(l,symbols=['LUNA'])))
    def test_duplicate_selection_reaches_negative_case(self):
        s=sample_one(self.c,random.Random(3),'price_move','crypto_majors','open_query',ASOF);specs=[dict(s,id='s0'),dict(s,id='s1')]
        rows=[{'id':'q0','spec_id':'s0','prompt':s['symbols'][0]+' '+s['instrument']+' first question?'},{'id':'q1','spec_id':'s1','prompt':s['symbols'][0]+' '+s['instrument']+' semantically identical question?'}];labels=[self.label(s),self.label(s)]
        class Fake:
            def embed(self,texts,purpose):return np.array([[1.,0.]]*len(texts))
        records,selected=select_questions(Fake(),rows,labels,specs,self.c,[],random.Random(1))
        self.assertEqual(len(selected),1);self.assertEqual(records[1]['decision'],'duplicate')
    def test_assertion_guard_catches_modal_schedule_but_preserves_real_condition(self):
        self.assertTrue(textual_premise_guard('Should I buy HYPE before the buyback scheduled next Tuesday?'))
        self.assertFalse(textual_premise_guard('If a HYPE buyback is scheduled next Tuesday, should I buy beforehand?'))
        self.assertTrue(textual_premise_guard('BTC dropped 8% today; why?'))
        self.assertFalse(textual_premise_guard('I heard BTC dropped 8%; is that true?'))
        self.assertFalse(textual_premise_guard("When is NVDA's next earnings?"))
    def test_question_and_imperative_forms_differ_from_statement(self):
        self.assertTrue(request_form_guard('Considering buying ETH at some point.'))
        self.assertFalse(request_form_guard('Show ETH order book levels'))
        self.assertFalse(request_form_guard('worth aping SI?'))
    def test_behavior_schema_cannot_mix_parents_and_variants(self):
        schema=batch_schema(self.c,['q'],'judge')
        levels=schema['properties']['items']['items']['properties']['behavior']['enum']
        self.assertIn('token_due_diligence/entry_risk',levels)
        self.assertNotIn('trade_selection/entry_risk',levels)
    def test_literal_binding_blocks_invented_extracted_symbol(self):
        s=sample_one(self.c,random.Random(3),'price_move','crypto_majors','open_query',ASOF)
        label=self.label(s)
        self.assertIn('missing_literal_instrument',reject_reasons(self.c,s,{'prompt':'What is moving today?'},label))
        self.assertIn('invented_symbol_extraction',reject_reasons(self.c,s,{'prompt':'What is moving today?'},label))
        self.assertEqual(literal_symbols(self.c,'gold versus silver?'),{'GOLD','SILVER'})
        self.assertEqual(literal_symbols(self.c,'S&P and Bitcoin?'),{'SPX','BTC'})
    def test_format_renderer_is_idempotent_and_preserves_stem(self):
        stem='Compare GOLD and SILVER?'
        for shape in ['plain','brief_bullets','ranked_list','comparison_table']:
            text=render_format(stem,shape)
            self.assertTrue(text.startswith(stem));self.assertEqual(render_format(text,shape),text)
        self.assertEqual(render_format(stem,'plain'),stem)
    def test_retry_reaches_rejection_then_resamples_within_source_intent(self):
        config=copy.deepcopy(self.c);config['bounds']['max_rounds']=2
        cell=next(c for c in config['reference_cells'] if c['source_id']=='oc-autoton')
        spec=sample_one(config,random.Random(7),cell['task'],cell['category'],'open_query',ASOF,cell);spec['id']='s'
        seen=[]
        def generation(p,specs,config,round_index,avoid,feedback):
            current=copy.deepcopy(specs[0]);seen.append(current)
            return [{'id':'q'+str(round_index),'spec_id':'s','round':round_index,'prompt':'What are the risks of '+', '.join(current['symbols'])+' '+current['instrument']+'?','requested_spec':current}]
        def judging(p,rows,config,purpose):
            label=self.label(rows[0]['requested_spec']);label['unsupported_claim']=len(seen)==1
            return [label]
        with tempfile.TemporaryDirectory() as d:
            class Fake:
                out=Path(d);cost=0.;tokens=0;calls=0
                def embed(self,texts,purpose):return np.array([[1.,0.]]*len(texts))
            with patch('engine.generate',side_effect=generation),patch('engine.judge',side_effect=judging):
                result=run_bank(Fake(),config,[spec],7)
            self.assertTrue(result['complete']);self.assertEqual(result['selected'],1)
            self.assertEqual(result['resampled_slots'],1);self.assertEqual(result['rejection_reasons']['unsupported_market_premise'],1)
            self.assertEqual(seen[0]['source_id'],seen[1]['source_id']);self.assertEqual(seen[0]['task'],seen[1]['task']);self.assertEqual(seen[0]['domain'],seen[1]['domain'])
            history=json.loads((Path(d)/'resampling-history.json').read_text());self.assertEqual(len(history),1)
    def test_generation_masks_reference_asset_but_retains_source_provenance(self):
        cell=next(c for c in self.c['reference_cells'] if c['source_id']=='oc-stonk')
        s=sample_one(self.c,random.Random(7),cell['task'],cell['category'],'open_query',ASOF,cell);s.update(id='s',symbols=['autoton'])
        seen=[]
        class Fake:
            def chat(self,instructions,obj,purpose,schema):
                seen.append(obj)
                return {'items':[{'id':r['id'],'prompt':'Is autoton worth a long?'} for r in obj['requests']]}
        rows=generate(Fake(),[s],self.c,0,[],{})
        self.assertNotIn('stonk',json.dumps(seen).casefold())
        self.assertEqual(rows[0]['requested_spec']['source_id'],'oc-stonk')
        self.assertEqual(rows[0]['requested_spec']['symbols'],['autoton'])
        self.assertNotIn('source_id',seen[0]['requests'][0]['spec'])
    def test_production_rejects_observed_task_mode_coupling(self):
        s=sample_one(self.c,random.Random(2),'full_analysis','crypto_majors','open_query',ASOF)
        s['premise_mode']='hypothetical'
        with self.assertRaises(ValueError):validate_spec(self.c,s)
        # Explicit intervention probes may deliberately leave the production region.
        s['intervention_axis']='premise_mode';validate_spec(self.c,s)
    def test_slack_matrix_separates_all_assets_and_instruments(self):
        for seed in range(6):
            p=make_plan(self.c,seed,64,'slack_matrix',ASOF,pool_size=2)
            cells={}
            for pool in p['pools']:
                for spec in pool['proposals']:
                    validate_spec(self.c,spec)
                    if spec['instrument'] in ['options','prediction_markets']:
                        self.assertEqual(spec['access_mode'],'read_only')
                        self.assertIsNone(spec['contract_context']['contract_reference'])
                s=pool['proposals'][0];key=(s['asset_class'],s['instrument']);cells[key]=cells.get(key,0)+1
            self.assertEqual(len(cells),16);self.assertEqual(set(cells.values()),{4})
    def test_read_only_scope_cannot_be_escalated_or_contract_invented(self):
        s=sample_one(self.c,random.Random(2),'options_analysis','equities','hypothetical',ASOF,instrument='options')
        bad=copy.deepcopy(s);bad['access_mode']='decision_support'
        with self.assertRaises(ValueError):validate_spec(self.c,bad)
        bad=copy.deepcopy(s);bad['contract_context']['strike']=100
        with self.assertRaises(ValueError):validate_spec(self.c,bad)
        bad=copy.deepcopy(s);bad['instrument']='spot'
        with self.assertRaises(ValueError):validate_spec(self.c,bad)
        l=self.label(s);text='Can you execute '+s['symbols'][0]+' options for me?'
        self.assertIn('read_only_effect_request',reject_reasons(self.c,s,{'prompt':text},l))
        self.assertTrue(read_only_action_guard('Redeem my prediction market shares.'))
        self.assertFalse(read_only_action_guard('How does exercise change a hypothetical option payoff?'))
        self.assertTrue(literal_instrument('Compare perpetual funding','perps'))
    def test_instrument_judge_and_literal_binding_are_required(self):
        s=sample_one(self.c,random.Random(2),'prediction_market_analysis','commodities','hypothetical',ASOF,instrument='prediction_markets')
        l=self.label(s);text='If listed, compare '+s['symbols'][0]+' prediction market odds?'
        self.assertEqual(reject_reasons(self.c,s,{'prompt':text},l),[])
        self.assertIn('axis_instrument',reject_reasons(self.c,s,{'prompt':text},dict(l,instrument='spot')))
        self.assertIn('missing_literal_instrument_type',reject_reasons(self.c,s,{'prompt':'What about '+s['symbols'][0]+'?'},l))
    def test_bad_judge_schema_fails_closed(self):
        class Fake:
            def chat(self,*args):return {'items':[{'id':'q','valid':'yes'}]}
        with self.assertRaises(ValueError):judge(Fake(),[{'id':'q','prompt':'anything'}],self.c,'test')
    def test_replay_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';out=root/'out';src.mkdir();out.mkdir()
            (src/'001-test.request.json').write_text(json.dumps({'endpoint':'responses','payload':{'model':'old'}}))
            p=ReplayProvider(out,src)
            with self.assertRaises(ValueError):p.call('responses',{'model':'new'},'test')

if __name__=='__main__':unittest.main()
