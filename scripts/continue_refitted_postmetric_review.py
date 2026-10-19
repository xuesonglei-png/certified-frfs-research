"""Wait on an actual prerequisite process then review the audited full cohort."""
import json
from pathlib import Path
import subprocess
import sys
import time
from run_refitted_postmetric_deletion import ROOT,FOLDER,digest
from continue_joint_guarded_soft5_knn_pilot import wait_verified


def main():
    script=ROOT/'scripts/review_refitted_postmetric_deletion.py'; frozen=digest(script)
    status=ROOT/'results/refitted-postmetric-review-chain-status.json'
    status.write_text(json.dumps({'phase':'waiting_all117_audit','prerequisite_pid':49916,
        'updated_unix':time.time(),'goal_achieved':False,'review_script_sha256':frozen},indent=2),encoding='utf-8')
    wait_verified(FOLDER,49916)
    assert digest(script)==frozen
    with (ROOT/'results/refitted-postmetric-deletion-review.log').open('w',encoding='utf-8') as log:
        rc=subprocess.run([sys.executable,str(script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    status.write_text(json.dumps({'phase':'review_complete' if rc==0 else 'failed','exit_code':rc,
        'updated_unix':time.time(),'goal_achieved':False,'review_script_sha256':frozen},indent=2),encoding='utf-8')
    if rc: raise RuntimeError('review failed; retain log and do not interpret partial output')


if __name__=='__main__': main()
