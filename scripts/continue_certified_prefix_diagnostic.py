"""Frozen complete inner-only cohort -> independent replay -> scoped review."""
import json
import subprocess
import sys
import time
from run_certified_prefix_diagnostic import ROOT,digest


def main():
    status=ROOT/'results/certified-prefix-diagnostic-chain-status.json'
    def update(phase,**kwargs):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**kwargs},indent=2),encoding='utf-8')
    try:
        frozen=json.loads((ROOT/'research/certified-prefix-diagnostic-source-freeze.json').read_text())
        for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha
        for phase,script in [('running','run_certified_prefix_diagnostic.py'),('auditing','audit_certified_prefix_diagnostic.py'),('reviewing','review_certified_prefix_diagnostic.py')]:
            with (ROOT/f'results/certified-prefix-diagnostic-{phase}.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                update(phase,child_pid=child.pid)
                if child.wait():raise RuntimeError(f'{phase} failed; inspect retained log')
        update('complete_audited_inner_diagnostic')
    except Exception as error:
        update('failed',error=str(error));raise


if __name__=='__main__':main()
