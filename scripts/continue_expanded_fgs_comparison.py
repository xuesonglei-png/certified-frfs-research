"""Prospective native/uniform controls, complete audit and scoped review."""
import json
import subprocess
import sys
import time
from run_expanded_fgs_comparison import ROOT,digest


def main():
    status=ROOT/'results/expanded-fgs-comparison-chain-status.json'
    def update(phase,**kwargs):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**kwargs},indent=2),encoding='utf-8')
    try:
        frozen=json.loads((ROOT/'research/expanded-fgs-comparison-source-freeze.json').read_text())
        for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha
        for phase,script in [('running','run_expanded_fgs_comparison.py'),('auditing','audit_expanded_fgs_comparison.py'),('reviewing','review_expanded_fgs_comparison.py')]:
            with (ROOT/f'results/expanded-fgs-comparison-{phase}.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                update(phase,child_pid=child.pid)
                if child.wait():raise RuntimeError(f'{phase} failed; inspect retained log')
        update('complete_audited_reviewed')
    except Exception as error:
        update('failed',error=str(error));raise


if __name__=='__main__':main()
