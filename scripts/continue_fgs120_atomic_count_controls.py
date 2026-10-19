"""Wait for actual atomicV2 audit then run and audit stronger count controls."""
import argparse
import json
import subprocess
import sys
import time
from run_fgs120_atomic_count_controls import ROOT,SOURCE,FOLDER,digest
from continue_joint_guarded_soft5_knn_pilot import wait_verified


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--atomic-chain-pid',type=int,required=True);args=parser.parse_args()
    status=ROOT/'results/fgs120-atomic-count-controls-chain-status.json'
    def update(phase,**kw):status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**kw},indent=2),encoding='utf-8')
    try:
        frozen=json.loads((ROOT/'research/fgs120-atomic-count-controls-source-freeze.json').read_text())
        for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
        update('waiting_atomic_v2_independent_audit',prerequisite_pid=args.atomic_chain_pid)
        wait_verified(SOURCE,args.atomic_chain_pid)
        for phase,script in [('running','run_fgs120_atomic_count_controls.py'),('auditing','audit_fgs120_atomic_count_controls.py')]:
            with (ROOT/f'results/fgs120-atomic-count-controls-{phase}.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                update(phase,child_pid=child.pid)
                if child.wait():raise RuntimeError(f'{phase} failed; see retained log')
        update('complete_audited')
    except Exception as error:
        update('failed',error=str(error));raise


if __name__=='__main__':main()
