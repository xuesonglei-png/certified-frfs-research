"""Corrected cohort with separate full audit and stronger baseline review."""
import json
import subprocess
import sys
import time
from run_atomic_pruning_classifier_v2 import ROOT,FOLDER,digest
from continue_soft_topk_knn_pilot import verify


def main():
    status=ROOT/'results/atomic-pruning-classifier-v2-chain-status.json'
    def update(phase,**values):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),
            'goal_achieved':False,**values},indent=2),encoding='utf-8')
    try:
        frozen=json.loads((ROOT/'research/atomic-pruning-classifier-v2-source-freeze.json').read_text())
        for rel,sha in frozen['fingerprints'].items():
            assert digest(ROOT/rel)==sha,rel
        for phase,script in [('running','run_atomic_pruning_classifier_v2.py'),
                             ('auditing','audit_atomic_pruning_classifier_v2.py'),
                             ('reviewing','review_atomic_pruning_classifier_v2.py')]:
            with (ROOT/f'results/atomic-pruning-classifier-v2-{phase}.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                update(phase,child_pid=child.pid)
                if child.wait(): raise RuntimeError(f'{phase} failed; see retained log')
            if phase=='auditing': verify(FOLDER)
        update('complete_audited_reviewed',audit_sha256=digest(FOLDER/'complete-audit.json'))
    except Exception as error:
        update('failed',error=str(error))
        raise


if __name__=='__main__':main()
