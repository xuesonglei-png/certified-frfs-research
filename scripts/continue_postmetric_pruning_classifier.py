"""Run/audit registered ALL15 cohort, then wait for audited recent comparator."""
import json
import subprocess
import sys
import time
from run_postmetric_pruning_classifier import ROOT,FOLDER,digest
from continue_joint_guarded_soft5_knn_pilot import wait_verified
from continue_soft_topk_knn_pilot import verify


def main():
    status=ROOT/'results/postmetric-pruning-classifier-chain-status.json'
    def update(phase,**values):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**values},indent=2),encoding='utf-8')
    def phase(name,script):
        with (ROOT/'results'/f'postmetric-pruning-classifier-{name}.log').open('w',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            update(name,child_pid=child.pid)
            if child.wait(): raise RuntimeError(f'{name} failed; see log')
    try:
        snapshot=json.loads((ROOT/'research/postmetric-pruning-classifier-source-freeze.json').read_text())
        for rel,sha in snapshot['fingerprints'].items(): assert digest(ROOT/rel)==sha
        verify(ROOT/'results/postmetric-pruning-admission')
        phase('running','run_postmetric_pruning_classifier.py')
        phase('auditing','audit_postmetric_pruning_classifier.py'); verify(FOLDER)
        update('waiting_recent_DSN_complete_audit',prerequisite_pid=49216)
        wait_verified(ROOT/'results/dsnfrs-formula-expanded-pilot',49216)
        phase('reviewing','review_postmetric_pruning_classifier.py')
        result=json.loads((FOLDER/'complete.json').read_text())
        update('complete_audited_reviewed',pilot_gate_passed=result['pilot_gate_passed'],audit_sha256=digest(FOLDER/'complete-audit.json'))
    except Exception as error:
        update('failed',error=str(error)); raise


if __name__=='__main__': main()
