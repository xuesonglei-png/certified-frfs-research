"""Atomic fixed-candidate run then full audit, with terminal failure status."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'results/atomic-postmetric-deletion-chain-status.json'


def main():
    for phase,script in [('running','run_atomic_postmetric_deletion.py'),('auditing','audit_atomic_postmetric_deletion.py')]:
        with (ROOT/f'results/atomic-postmetric-deletion-{phase}.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            STATUS.write_text(json.dumps({'phase':phase,'child_pid':child.pid,'updated_unix':time.time(),
                'goal_achieved':False},indent=2),encoding='utf-8'); rc=child.wait()
        if rc:
            STATUS.write_text(json.dumps({'phase':'failed','failed_phase':phase,'exit_code':rc,
                'updated_unix':time.time(),'goal_achieved':False},indent=2),encoding='utf-8')
            raise RuntimeError(f'{phase} failed; retain partial artifacts, not a complete result')
    STATUS.write_text(json.dumps({'phase':'complete_audited_training_diagnostic','updated_unix':time.time(),
        'goal_achieved':False},indent=2),encoding='utf-8')


if __name__=='__main__': main()
