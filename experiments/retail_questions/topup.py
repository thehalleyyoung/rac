"""Quota repair from retained failures; no filter/threshold relaxation."""
import argparse, collections, hashlib, json, os, re, time
from pathlib import Path
import numpy as np
from run import HERE, Provider, FILTER, candidates, screen, select, stats, dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--key-file',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    key=os.environ.get('OPENAI_API_KEY')
    if not key and a.key_file:key=re.search(r'OPENAI_API_KEY\s*=\s*[\"\x27]?([^\s\"\x27`]+)',a.key_file.read_text())[1]
    p=Provider(a.out,key)
    refs=json.loads((HERE/'prompts.json').read_text());old=json.loads((a.run/'all-candidates.json').read_text());bank=json.loads((a.run/'questions.json').read_text());base=json.loads((a.run/'baseline.json').read_text());counts=collections.Counter(r['family'] for r in bank)
    missing=[r for r in refs if counts[r['id']]<5];conditions=json.loads((a.run/'frozen-tuning.json').read_text())['conditions']
    feedback={'rejected':[{'question':r['prompt'],'reason':r['label']['reason']} for r in old if r['family'] in {r['id'] for r in missing} and r['decision']=='filter'],'requirements':'Each hold/sell question must explicitly concern an already-owned position and ask whether to hold/trim/sell. Each catalyst-buy question must explicitly ask whether to buy or wait AHEAD OF a named prospective catalyst. Always ask a clear question, not a statement of consideration. Use only BTC ETH SOL HYPE DOGE AVAX ZRO as crypto assets. No crypto earnings. HYPE buybacks, ZRO/AVAX unlocks, ETH/SOL protocol upgrades are example categories, not verified upcoming events: use if/assuming/rumored. Keep voice brief and natural; no generic missing-info checklist. Prefer different decision constraints, not just new prices.'}
    freeze={'input_run':a.run.name,'missing_families':[r['id'] for r in missing],'feedback':feedback,'filter':FILTER,'conditions':conditions,'quota':5,'reference_cosine_max':.94,'pair_cosine_max':.90,'code_sha256':hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest(),'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(a.out/'frozen-tuning.json',freeze)
    for file in ['run.py','topup.py','family_axes.json','prompts.json']:(a.out/file).write_bytes((HERE/file).read_bytes())
    fresh=candidates(p,missing,'tuned',8,conditions,[r['prompt'] for r in old],feedback) if missing else []
    for i,r in enumerate(fresh):r['id']=f'topup-{i:03d}'
    dump(a.out/'generated.json',fresh);fl=screen(p,fresh,refs,'topup-screen')
    oldE=np.load(a.run/'embeddings.npz');newE=p.embed([r['prompt'] for r in fresh],'topup-embeddings') if fresh else np.empty((0,oldE['candidates'].shape[1]));E=np.concatenate([oldE['candidates'],newE]);rows=old+fresh;labels=[r['label'] for r in old]+fl
    selected,decisions=select(rows,labels,E,refs,oldE['references'])
    dump(a.out/'all-candidates.json',decisions);dump(a.out/'questions.json',selected);dump(a.out/'baseline.json',base)
    np.savez_compressed(a.out/'embeddings.npz',references=oldE['references'],baseline=oldE['baseline'],candidates=E)
    summary={'candidate_count':len(rows),'topup_count':len(fresh),'topup_stats':stats(fresh,fl,refs) if fresh else None,'selected_count':len(selected),'selected_family_counts':dict(collections.Counter(r['family'] for r in selected)),'selected_src_counts':dict(collections.Counter(r['src'] for r in selected)),'decisions':dict(collections.Counter(r['decision'] for r in decisions)),'estimated_usd':p.cost,'tokens':p.tokens,'calls':p.calls,'source_run':a.run.name}
    dump(a.out/'summary.json',summary)
    text=['# RQ01 generated question bank','','Exploration, not a sealed evaluation set.','',f'{len(selected)} selected questions.','']
    for ref in refs:text+=['## '+ref['id'],'','Reference: '+ref['prompt'],'']+['- '+r['prompt'] for r in selected if r['family']==ref['id']]+['']
    (a.out/'questions.md').write_text('\n'.join(text));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
