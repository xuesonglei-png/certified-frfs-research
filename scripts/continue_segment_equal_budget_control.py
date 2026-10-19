"""Wait for full independent segment audit, then run fixed16 comparison."""
import json
import subprocess
import sys
import time
from run_segment_equal_budget_control import ROOT,SEGMENT
from continue_joint_guarded_soft5_knn_pilot import wait_verified


def main():
    status=ROOT/'results/segment-equal-budget-control-chain-status.json'
    def update(phase,**values):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**values},indent=2),encoding='utf-8')
    try:
        update('waiting_segment_complete_audit',prerequisite_pid=50456)
        wait_verified(SEGMENT,50456)
        with (ROOT/'results/segment-equal-budget-control-running.log').open('w',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_segment_equal_budget_control.py')],cwd=ROOT,
                                   stdout=stream,stderr=subprocess.STDOUT)
            update('running',child_pid=child.pid)
            if child.wait(): raise RuntimeError('fixed16 control failed; see log')
        update('complete_scalar_replayed',scope='training_budget_control_not_test_gain')
    except Exception as error:
        update('failed',error=str(error)); raise


if __name__=='__main__': main()
