"""Execute and audit ALL15 training-only pruning without query scores."""
import json
import subprocess
import sys
import time
from run_postmetric_pruning_admission import ROOT


def main():
    status=ROOT/'results/postmetric-pruning-admission-chain-status.json'
    def update(phase,**values):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**values},indent=2),encoding='utf-8')
    try:
        for phase,script in [('running','run_postmetric_pruning_admission.py'),('auditing','audit_postmetric_pruning_admission.py')]:
            with (ROOT/'results'/f'postmetric-pruning-admission-{phase}.log').open('w',encoding='utf-8') as stream:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                update(phase,child_pid=child.pid)
                if child.wait(): raise RuntimeError(f'{phase} failed; see log')
        update('complete_audited',scope='all15_training_pruning_not_test_gain')
    except Exception as error:
        update('failed',error=str(error)); raise


if __name__=='__main__': main()
