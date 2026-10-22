"""Seeded compatible specifications. No LLM, private state or market facts."""
from __future__ import annotations
import collections, copy, hashlib, json, math, random
from pathlib import Path

HERE=Path(__file__).resolve().parent

def load_config(path=None):
    return json.loads(Path(path or HERE/'axes.json').read_text())

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def quotas(weights,n,rng):
    if any(not math.isfinite(float(v)) or v<0 for v in weights.values()):raise ValueError('negative or nonfinite weight')
    weights={k:float(v) for k,v in weights.items() if v>0}
    if not weights or any(not math.isfinite(v) for v in weights.values()):raise ValueError('invalid positive weights')
    total=sum(weights.values());expected={k:n*v/total for k,v in weights.items()}
    result={k:int(v) for k,v in expected.items()};tie={k:rng.random() for k in weights}
    order=sorted(weights,key=lambda k:(-(expected[k]-result[k]),tie[k]))
    for k in order[:n-sum(result.values())]:result[k]+=1
    return result

def pick(rng,weights):
    return rng.choices(list(weights),weights=list(weights.values()),k=1)[0]

def compatible_modes(task,config=None):
    if config and task in config.get("task_mode_constraints",{}):return config["task_mode_constraints"][task]
    if task in ['scenario_analysis','risk_sizing']:return ['hypothetical']
    if task=='source_verification':return ['claim_check']
    if task=='catalyst_entry':return ['hypothetical','claim_check']
    if task=='event_study':return ['open_query','claim_check']
    return ['open_query','hypothetical','claim_check']

def assign_compatible(slots,capacities,allowed,rng):
    """Randomized augmenting-path matching, preserving both quota marginals.
    Fails rather than silently creating an invalid cross-product or changing mix.
    """
    holders={k:[] for k,v in capacities.items() if v>0};assigned={}
    orders={i:rng.sample([k for k in allowed(s) if k in holders],len([k for k in allowed(s) if k in holders])) for i,s in enumerate(slots)}
    def place(i,visited):
        for value in orders[i]:
            if value in visited:continue
            visited.add(value)
            if len(holders[value])<capacities[value]:
                holders[value].append(i);assigned[i]=value;return True
            for old in list(holders[value]):
                if place(old,visited):
                    holders[value].remove(old);holders[value].append(i);assigned[i]=value;return True
        return False
    indices=list(range(len(slots)));rng.shuffle(indices);indices.sort(key=lambda i:len(orders[i]))
    for i in indices:
        if not place(i,set()):raise ValueError('infeasible task/axis quotas; change profile explicitly')
    return [assigned[i] for i in range(len(slots))]

def variants(config,task,mode,domain=None,cell=None):
    result=list(config['tasks'][task]['variants'])
    if mode!='claim_check':
        result=[v for v in result if (task,v) not in [('price_move','premise_check'),('news','rumor_check')]]
    if cell and cell.get('allowed_variants'):result=[v for v in result if v in cell['allowed_variants']]
    if task=='relative_value':
        result=[v for v in result if v not in (['asset_nav'] if domain=='commodities' else ['precious_ratio','crude_spread'])]
    if task=='event_outlook':
        result=[v for v in result if v not in (['calendar_consensus'] if domain=='commodities' else ['weather','news_reaction','supply_demand'])]
    return result

def sample_one(config,rng,task,domain,mode,as_of,cell=None):
    v=rng.choice(variants(config,task,mode,domain,cell));assets=config['catalog'][domain]
    rule=cell['asset_rule'] if cell else ('onchain_any' if domain=='onchain_spot' else 'major_any' if domain=='crypto_majors' else 'commodity_any' if domain=='commodities' else 'equity_any')
    if task=='relative_value' and not cell:rule='holdings_proxy' if domain=='equities' else rng.choice(['precious_pair','crude_pair'])
    if task=='event_outlook' and v=='weather':rule='natgas'
    if task=='event_outlook' and v=='calendar_consensus':rule='issuer'
    if task=='fund_flows':rule='etf_underlying'
    if task=='relative_value' and v=='precious_ratio':rule='precious_pair'
    if task=='relative_value' and v=='crude_spread':rule='crude_pair'
    if task=='relative_value' and v=='asset_nav':rule='holdings_proxy'
    universe=[x['symbol'] for x in assets]
    fixed={'drv_spot':['DRV'],'precious_pair':['GOLD','SILVER'],'crude_pair':['BRENT','WTI'],'natgas':['NATGAS'],'index':['SPX'],'holdings_proxy':['MSTR','BTC']}
    if rule in fixed:symbols=fixed[rule]
    else:
        if rule=='etf_underlying':universe=['BTC','ETH']
        if rule in ['issuer','equity_any','equity_pair','ai_equity']:universe=[s for s in universe if s!='SPX']
        if rule=='crude':universe=['CRUDE','BRENT','WTI']
        count=3 if task in ['relative_strength','portfolio_risk'] else 2 if rule.endswith('_pair') or v in ['cross_asset','relative_divergence','alternatives','compare_candidates','compare_longs'] else 1
        symbols=rng.sample(universe,min(count,len(universe)))
    if task=='theme_exposure' or rule=='ai_equity' and v!='compare_longs':symbols=[]
    if task=='event_study':ht='historical'
    elif task=='book_depth':ht='current'
    elif task in ['catalyst_entry','scenario_analysis','event_outlook']:ht='prospective'
    elif task=='relative_strength':ht='trailing'
    else:ht=pick(rng,{'current':.4,'trailing':.4,'prospective':.2})
    shape_weights=dict(config['axes']['answer_shape'])
    if task=='relative_strength':shape_weights={'ranked_list':1}
    elif task=='book_depth':shape_weights={'plain':.5,'comparison_table':.5}
    elif task not in ['portfolio_risk','event_study','theme_exposure','execution_cost']:
        shape_weights.pop('ranked_list');shape_weights.pop('comparison_table')
    context=None;constraint=None
    if task in ['allocate_capital','manage_position','risk_sizing','execution_cost','portfolio_risk'] or (task=='technical_setup' and v=='stop_invalidation'):
        budget=rng.choice([250,500,1000,1500,2500,5000]);maxloss=rng.choice([25,50,100,200]);pct=rng.choice([2,3,5,10])
        context={'provenance':'synthetic trader scenario, not a real account','budget_usd':budget,'maximum_loss_usd':min(maxloss,budget//5),'percentage':pct,'leverage':rng.choice([2,3,5])}
        constraint=rng.choice(['avoid concentration','prefer lower turnover','preserve some cash','consider liquidity','keep leverage limited'])
    spec={'task':task,'variant':v,'domain':domain,'symbols':symbols,'premise_mode':mode,'horizon_type':ht,'window':rng.choice(config['horizons'][ht]),'as_of':as_of,'voice':pick(rng,config['axes']['voice']),'difficulty':pick(rng,config['axes']['difficulty']),'evidence_demand':pick(rng,config['axes']['evidence_demand']),'answer_shape':pick(rng,shape_weights),'trader_context':context,'decision_constraint':constraint,'theme':('AI stocks' if rule=='ai_equity' else rng.choice(config['themes'][domain])) if task=='theme_exposure' or rule=='ai_equity' else None,'venue_scope':('Hyperliquid spot, user-supplied venue hint; listing unverified' if rule=='drv_spot' else rng.choice(config['venues'][domain])) if task in ['book_depth','execution_cost'] else None,'source_id':cell['source_id'] if cell else None,'asset_rule':rule,'intent_structure':pick(rng,config['axes']['intent_structure']),'identity_status':'unresolved chain/address' if domain=='onchain_spot' else 'underlying unspecified contract' if domain=='commodities' else 'reference identifier; market unresolved'}
    decision=task in ['trade_selection','allocate_capital','manage_position','short_setup','risk_sizing','portfolio_risk'] or task=='technical_setup' and v in ['stop_invalidation','entry_trigger','scenario_levels']
    spec['holding_period']=rng.choice(list(config['holding_periods'])) if decision else None
    spec['risk_appetite']=pick(rng,config['risk_appetites']) if decision else None
    surface_pool=config.get('variant_surface_compatibility',{}).get(task+'/'+v) or config.get('surface_compatibility',{}).get(task,{}).get(domain) or config['evidence_surfaces'][domain]
    spec['evidence_surface']=rng.choice(surface_pool) if spec['evidence_demand']!='ordinary' else None
    if decision and spec['horizon_type']=='prospective':
        spec['horizon_type']='trailing';spec['window']=rng.choice(config['horizons']['trailing'])
    validate_spec(config,spec)
    return spec

def validate_spec(config,s):
    if s['task'] not in config['tasks']:raise ValueError('unknown task')
    t=config['tasks'][s['task']]
    if s['domain'] not in t['domains']:raise ValueError('incompatible task/domain')
    if s.get('intervention_axis')!='premise_mode' and s['premise_mode'] not in compatible_modes(s['task'],config):raise ValueError('incompatible task/premise')
    if s['variant'] not in variants(config,s['task'],s['premise_mode'],s['domain']):raise ValueError('incompatible variant')
    allowed={a['symbol'] for a in config['catalog'][s['domain']]}
    if s.get('asset_rule')=='holdings_proxy':allowed.add('BTC')
    if s['task']=='event_outlook' and s['variant']=='weather' and s['symbols']!=['NATGAS']:raise ValueError('weather variant requires natural gas context')
    if s['task']=='relative_value' and s['variant']=='precious_ratio' and set(s['symbols'])!={'GOLD','SILVER'}:raise ValueError('precious ratio requires gold/silver')
    if s['task']=='relative_value' and s['variant']=='crude_spread' and set(s['symbols'])!={'BRENT','WTI'}:raise ValueError('crude spread requires Brent/WTI')
    if s['task']=='relative_value' and s['variant']=='asset_nav' and s['symbols']!=['MSTR','BTC']:raise ValueError('holdings proxy requires equity/underlying')
    if not set(s['symbols'])<=allowed or len(s['symbols'])!=len(set(s['symbols'])):raise ValueError('unsupported/duplicate symbol')
    if s['task']!='theme_exposure' and s.get('asset_rule')!='ai_equity' and not s['symbols']:raise ValueError('missing symbol scope')
    for axis in ['voice','difficulty','evidence_demand','answer_shape']:
        if s[axis] not in config['axes'][axis]:raise ValueError('unsupported axis level')
    allowed_surfaces=config.get('variant_surface_compatibility',{}).get(s['task']+'/'+s['variant']) or config.get('surface_compatibility',{}).get(s['task'],{}).get(s['domain']) or config['evidence_surfaces'][s['domain']]
    if s.get('evidence_surface') is not None and s['evidence_surface'] not in allowed_surfaces:raise ValueError('incompatible evidence source/task')
    if s['window'] not in config['horizons'][s['horizon_type']]:raise ValueError('incompatible horizon')
    if s['task']=='book_depth' and s['horizon_type']!='current':raise ValueError('book requires a current snapshot')
    if s['task']=='event_study' and s['horizon_type']!='historical':raise ValueError('event study requires historical events')
    if not isinstance(s['as_of'],str) or not s['as_of']:raise ValueError('missing as-of')

def make_plan(config,seed,n,profile,as_of,pool_size=None):
    if not 1<=n<=config['bounds']['max_specs']:raise ValueError('count outside declared bound')
    p=config['profiles'][profile];rng=random.Random(seed);cells=[]
    if p['reference_cells']:
        source_weights=p.get('source_weights') or {r['source_id']:1 for r in config['reference_cells']}
        if not set(source_weights)<=set(r['source_id'] for r in config['reference_cells']):raise ValueError('unknown source weight key')
        sq=quotas(source_weights,n,rng)
        for cell in config['reference_cells']:cells += [cell]*sq.get(cell['source_id'],0)
        rng.shuffle(cells)
    else:
        dq=quotas(p['category_weights'],n,rng);sq={}
        for domain,count in dq.items():
            compatible=[t for t,d in config['tasks'].items() if domain in d['domains']]
            tw=p.get('task_weights_by_category',{}).get(domain) or {t:1 for t in compatible}
            if not set(tw)<=set(compatible):raise ValueError('incompatible task weight key')
            tq=quotas(tw,count,rng)
            for task,amount in tq.items():
                cells += [{'source_id':None,'category':domain,'task':task,'asset_rule':None,'allowed_variants':None}]*amount
        rng.shuffle(cells)
    tasks=[c['task'] for c in cells];domains=[c['category'] for c in cells]
    mq=quotas(config['axes']['premise_mode'],n,rng)
    try:
        modes=assign_compatible(tasks,mq,lambda t:compatible_modes(t,config),rng);mode_policy='exact compatible premise quotas'
    except ValueError:
        modes=[pick(rng,{m:w for m,w in config['axes']['premise_mode'].items() if m in compatible_modes(t,config)}) for t in tasks]
        mode_policy='conditional premise sampling: rounded target was infeasible for the selected small task mix'
    size=pool_size or config['bounds']['spec_pool'];pools=[];seen=set()
    for i,(cell,task,domain,mode) in enumerate(zip(cells,tasks,domains,modes)):
        pool=[]
        for _ in range(size):
            for attempt in range(100):
                s=sample_one(config,rng,task,domain,mode,as_of,cell if p['reference_cells'] else None);fp=fingerprint(s)
                if fp not in seen:break
            else:raise ValueError('spec pool saturated')
            seen.add(fp);pool.append(s)
        pools.append({'slot':f's{i:04d}','task':task,'domain':domain,'premise_mode':mode,'source_id':cell['source_id'],'proposals':pool})
    return {'version':config['version'],'seed':seed,'count':n,'profile':profile,'as_of':as_of,'config_sha256':fingerprint(config),'sampling':'randomized reference-intent/category strata; compatible conditional proposal pools; later RAC representative selection','premise_policy':mode_policy,'source_quotas':sq,'task_quotas':dict(collections.Counter(tasks)),'domain_quotas':dict(collections.Counter(domains)),'premise_quotas':dict(collections.Counter(modes)),'pools':pools}

def spec_text(config,s):
    return '. '.join([config['tasks'][s['task']]['description'],config['tasks'][s['task']]['variants'][s['variant']],s['domain'],s['premise_mode'],s['horizon_type'],s['evidence_demand'],s['answer_shape'],s['difficulty']])

def summarize_specs(specs):
    axes=['task','domain','source_id','premise_mode','horizon_type','voice','difficulty','evidence_demand','answer_shape','intent_structure']
    counts={axis:dict(collections.Counter(s[axis] for s in specs)) for axis in axes}
    counts['task_variant']=dict(collections.Counter(s['task']+'/'+s['variant'] for s in specs))
    counts['symbols']=dict(collections.Counter(x for s in specs for x in s['symbols']))
    return counts
