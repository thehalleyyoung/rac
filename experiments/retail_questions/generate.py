"""Single entry point for fresh third-party development-question generation.
Sequential subprocesses share a cost/token/call ledger. No UI or human labels.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def commands(root,key_file=None):
    extra=['--key-file',str(key_file)] if key_file else []
    # Child helpers accept env credentials; key-file is optional in all CLIs.
    return [
      [sys.executable,str(HERE/'run.py'),'--out',str(root/'development')]+extra,
      [sys.executable,str(HERE/'finish.py'),'--development',str(root/'development'),'--out',str(root/'conditional')]+extra,
      [sys.executable,str(HERE/'topup.py'),'--run',str(root/'conditional'),'--out',str(root/'final')]+extra,
      [sys.executable,str(HERE/'audit.py'),'--run',str(root/'final'),'--out',str(root/'audit')]+extra,
    ]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--key-file',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,RQ01_BUDGET_LEDGER=str((a.out/'budget.json').resolve()))
    for command in commands(a.out,a.key_file):subprocess.run(command,env=env,check=True)
    result=json.loads((a.out/'final/summary.json').read_text())
    if result['selected_count']!=60 or any(v!=5 for v in result['selected_family_counts'].values()):
        raise RuntimeError('bank incomplete: retained all artifacts; do not claim target distribution')
    print('Complete: final/questions.json and audit/audit.json; 60 selected, five per family.')

if __name__=='__main__':main()
