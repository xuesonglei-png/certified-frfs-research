"""Separate V2 diagnostic observer; dead prerequisite is a failure, not a hang."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from continue_joint_guarded_soft5_knn_pilot import wait_verified
from run_simplex_soft5_knn_pilot import ROOT,digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['atomic','thinned'],required=True)
    parser.add_argument('--chain-pid',type=int,required=True)
    args=parser.parse_args()
    stem='atomic' if args.kind=='atomic' else 'thinned-atomic'
    folder=ROOT/f'results/{"atomic-pruning" if args.kind=="atomic" else "thinned-atomic"}-classifier-v2'
    script=ROOT/'scripts'/('diagnose_atomic_pruning_transfer_v2.py' if args.kind=='atomic' else 'diagnose_thinned_atomic_transfer_v2.py')
    status=ROOT/f'results/{stem}-transfer-v2-chain-status.json';frozen=digest(script)
    def update(phase,**kw):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),
            'goal_achieved':False,'diagnostic_sha256':frozen,**kw},indent=2),encoding='utf-8')
    try:
        update('waiting_v2_complete_independent_audit',prerequisite_pid=args.chain_pid)
        wait_verified(folder,args.chain_pid)
        assert frozen==digest(script)
        with (ROOT/f'results/{stem}-transfer-v2.log').open('w',encoding='utf-8') as log:
            update('running_inner_only_diagnostic')
            result=subprocess.run([sys.executable,str(script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('diagnostic failed; see retained log')
        update('complete_inner_only_diagnostic')
    except Exception as error:
        update('failed',error=str(error));raise


if __name__=='__main__':main()
