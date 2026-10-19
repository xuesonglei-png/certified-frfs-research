"""Run stronger ALL15 native stopping control after original cohort audit."""
import json
import subprocess
import sys
import time

from run_dsnfrs_formula_expanded_pilot import ROOT
from continue_joint_guarded_soft5_knn_pilot import wait_verified


def main():
    status = ROOT/'results/dsnfrs-formula-expanded-chain-status.json'
    def update(phase, **values):
        status.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(),
                                     'goal_achieved': False, **values}, indent=2), encoding='utf-8')
    try:
        update('waiting_original_complete_audit', prerequisite_pid=50500)
        wait_verified(ROOT/'results/dsnfrs-formula-native-pilot', 50500)
        for phase, script in [('running', 'run_dsnfrs_formula_expanded_pilot.py'),
                              ('auditing', 'audit_dsnfrs_formula_expanded_pilot.py')]:
            with (ROOT/'results'/f'dsnfrs-formula-expanded-{phase}.log').open('w', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                update(phase, child_pid=child.pid)
                if child.wait(): raise RuntimeError(f'{phase} failed; see log')
        update('complete_audited', scope='formula_controlled_expanded_development_not_author_reproduction')
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
