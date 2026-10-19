"""Full atomic path run, audit and transparent comparison to old training path."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'results/thinned-atomic-pruning-admission-chain-status.json'


def main():
    from run_simplex_soft5_knn_pilot import digest
    frozen=json.loads((ROOT/'research/thinned-atomic-pruning-admission-source-freeze.json').read_text())
    for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha
    for phase,script in [('running','run_thinned_atomic_pruning_admission.py'),('auditing','audit_thinned_atomic_pruning_admission.py'),
                         ('reviewing','review_thinned_atomic_pruning_admission.py')]:
        with (ROOT/f'results/thinned-atomic-pruning-admission-{phase}.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            STATUS.write_text(json.dumps({'phase':phase,'child_pid':child.pid,'updated_unix':time.time(),'goal_achieved':False},indent=2),encoding='utf-8')
            rc=child.wait()
        if rc:
            STATUS.write_text(json.dumps({'phase':'failed','failed_phase':phase,'exit_code':rc,'updated_unix':time.time(),
                'goal_achieved':False},indent=2),encoding='utf-8');raise RuntimeError(f'{phase} failed; retain partial evidence')
    STATUS.write_text(json.dumps({'phase':'complete_audited_training_path','updated_unix':time.time(),
        'goal_achieved':False},indent=2),encoding='utf-8')


if __name__=='__main__':main()
