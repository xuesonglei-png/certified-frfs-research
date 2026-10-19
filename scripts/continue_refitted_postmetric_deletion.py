"""Run fixed-candidate relearning then audit; terminal failures stay visible."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'results/refitted-postmetric-deletion-chain-status.json'


def main():
    for phase,script in [('running','run_refitted_postmetric_deletion.py'),
                         ('auditing','audit_refitted_postmetric_deletion.py')]:
        with (ROOT/f'results/refitted-postmetric-deletion-{phase}.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            STATUS.write_text(json.dumps({'phase':phase,'child_pid':child.pid,'updated_unix':time.time(),
                'goal_achieved':False},indent=2),encoding='utf-8')
            rc=child.wait()
        if rc:
            STATUS.write_text(json.dumps({'phase':'failed','failed_phase':phase,'exit_code':rc,
                'updated_unix':time.time(),'goal_achieved':False},indent=2),encoding='utf-8')
            raise RuntimeError(f'{phase} failed; do not treat partial files as complete')
    STATUS.write_text(json.dumps({'phase':'complete_audited_training_diagnostic',
        'updated_unix':time.time(),'goal_achieved':False},indent=2),encoding='utf-8')


if __name__=='__main__': main()
