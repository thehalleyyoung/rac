"""Resume a recorded development stage after a failed final generation contract.
Does not erase or relabel the failed run. Final protocol is frozen before calls.
"""
import argparse, collections, hashlib, json, os, re, time
from pathlib import Path
import numpy as np
from run import HERE, Provider, FILTER, candidates, screen, select, stats, dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--development',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--key-file',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    key=os.environ.get('OPENAI_API_KEY')
    if not key and a.key_file:key=re.search(r'OPENAI_API_KEY\s*=\s*[\"\x27]?([^\s\"\x27`]+)',a.key_file.read_text())[1]
    p=Provider(a.out,key)
    refs=json.loads((HERE/'prompts.json').read_text());base=json.loads((a.development/'baseline.json').read_text());dev=json.loads((a.development/'development.json').read_text());old=json.loads((a.development/'frozen-tuning.json').read_text())
    conditions=old['conditions']
    families=json.loads((HERE/'family_axes.json').read_text())
    flat=[v for levels in families.values() for v in levels]
    axisE=p.embed(flat,'conditional-axis-embeddings')
    offset=0
    from run import rac
    for name,levels in families.items():
        vectors=axisE[offset:offset+len(levels)]; offset+=len(levels)
        order=list(dict.fromkeys(rac.coverage_levels(vectors,len(levels))+rac.farthest_levels(vectors,len(levels))+list(range(len(levels)))))
        conditions[name]=[levels[i] for i in order]
    feedback={'failures':old['feedback']['rejected_examples'],'reference_mean_words':float(np.mean([len(r['prompt'].split()) for r in refs])),'observed_failure':'Run-04 universal evidence/framing axes drifted into generic missing-information questions and asset-move explanations, rather than preserving some news, exposure and full-analysis tasks. Replace those with family-conditional axes.','required_change':'Previous generation stayed too close to reference assets/events and wording. Change the reasoning/evidence need within each family. Do not merely paraphrase or swap tickers. Compact, casual voice. Each assigned cell produces one question; omit no cells.'}
    freeze={'development_artifact':a.development.name,'development_config_sha256':hashlib.sha256((a.development/'config.json').read_bytes()).hexdigest(),'conditions':conditions,'feedback':feedback,'filter':FILTER,'code_sha256':hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest(),'per_family_candidates':8,'quota':5,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(a.out/'frozen-tuning.json',freeze)
    for source in ['run.py','finish.py','family_axes.json','prompts.json']:
        (a.out/source).write_bytes((HERE/source).read_bytes())
    rows=candidates(p,refs,'tuned',8,conditions,[r['prompt'] for r in dev],feedback)
    dump(a.out/'generated.json',rows)
    labels=screen(p,rows,refs,'final-screen')
    E=p.embed([r['prompt'] for r in refs+base+rows],'question-embeddings')
    refE,bE,qE=E[:12],E[12:36],E[36:]
    chosen,decisions=select(rows,labels,qE,refs,refE)
    dump(a.out/'all-candidates.json',decisions);dump(a.out/'questions.json',chosen);dump(a.out/'baseline.json',base)
    np.savez_compressed(a.out/'embeddings.npz',references=refE,baseline=bE,candidates=qE)
    summary={'baseline':stats(base,[x['label'] for x in base],refs,bE,refE),'development':stats(dev,[x['label'] for x in dev],refs),'final_candidates':stats(rows,labels,refs,qE,refE),'selected_count':len(chosen),'selected_family_counts':dict(collections.Counter(x['family'] for x in chosen)),'selected_src_counts':dict(collections.Counter(x['src'] for x in chosen)),'decisions':dict(collections.Counter(x['decision'] for x in decisions)),'estimated_usd':p.cost,'tokens':p.tokens,'calls':p.calls,'controls_reused_from':a.development.name,'limitations':['Same-model generation/filter bias; no human annotation.','12 seen examples; no universal distribution or third-party independence claim.','Family frequencies matched by construction.','Final generation revised after failed output-count contract; both attempts retained.','No GTE answer evaluation or market observations.']}
    dump(a.out/'summary.json',summary)
    text=['# RQ01 generated question bank','','Exploration, not a sealed evaluation set.','',f'{len(chosen)} selected questions.','']
    for r in refs:
        text+=['## '+r['id'],'','Reference: '+r['prompt'],'']+['- '+x['prompt'] for x in chosen if x['family']==r['id']]+['']
    (a.out/'questions.md').write_text('\n'.join(text))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
