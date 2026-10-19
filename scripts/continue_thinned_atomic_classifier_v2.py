"""Corrected cohort with separate full audit and stronger baseline review."""
import json
import subprocess
import sys
import time
from run_thinned_atomic_classifier_v2 import ROOT,FOLDER,digest
from continue_soft_topk_knn_pilot import verify
from continue_joint_guarded_soft5_knn_pilot import wait_verified


def main():
    status=ROOT/'results/thinned-atomic-classifier-v2-chain-status.json'
    def update(phase,**values):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),
            'goal_achieved':False,**values},indent=2),encoding='utf-8')
    try:
        frozen=json.loads((ROOT/'research/thinned-atomic-classifier-v2-source-freeze.json').read_text())
        for rel,sha in frozen['fingerprints'].items():
            assert digest(ROOT/rel)==sha,rel
        for phase,script in [('reusing_verified_raw','reuse_thinned_atomic_classifier_v2.py'),('running','run_thinned_atomic_classifier_v2.py'),
                             ('auditing','audit_thinned_atomic_classifier_v2.py'),
                             ('reviewing','review_thinned_atomic_classifier_v2.py')]:
            if phase=='reviewing':
                update('waiting_atomic_v2_complete_audit',prerequisite_pid=42572)
                wait_verified(ROOT/'results/atomic-pruning-classifier-v2',42572)
            with (ROOT/f'results/thinned-atomic-classifier-v2-{phase}.log').open('w',encoding='utf-8') as log:
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
