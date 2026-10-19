"""Run/audit/review ALL15 after audited Iris and prior DC completion."""
import json
import subprocess
import sys
import time
import traceback
from run_frs_cell_inner_v2 import ROOT, FOLDER, PREVIOUS, digest, frozen

STATUS = ROOT/'results/frs-cell-inner-v2-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), 'goal_achieved': False, **values}, indent=2), encoding='utf-8')


def main():
    frozen()
    smoke = json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256'] == digest(FOLDER/'iris.json')
    status('waiting_for_previous_independent_DC_audit', prerequisite='results/third-enemy-inner/complete-audit.json')
    while not (PREVIOUS/'complete-audit.json').exists():
        prior_status = json.loads((ROOT/'results/third-enemy-inner-chain-status.json').read_text())
        if prior_status['phase'] == 'failed':
            raise RuntimeError('Prior DC audit failed; no new cohort started')
        time.sleep(10)
    audit = json.loads((PREVIOUS/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(PREVIOUS/'complete.json')
    for phase, script in [('running', 'run_frs_cell_inner_v2.py'), ('auditing', 'audit_frs_cell_inner_v2.py'),
                          ('reviewing', 'review_frs_cell_inner.py')]:
        frozen(); log = ROOT/'results'/f'frs-cell-inner-v2-{phase}.log'
        with log.open('a', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status(phase, child_pid=child.pid, log=log.relative_to(ROOT).as_posix(), executed_script_sha256=digest(ROOT/'scripts'/script))
            if child.wait():
                raise RuntimeError(f'{script} failed; preserve every artifact/log')
    summary = json.loads((FOLDER/'complete.json').read_text()); audit = json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(FOLDER/'complete.json')
    status('complete_audited_reviewed', pilot_gate_passed=summary['pilot_gate_passed'], audit_sha256=digest(FOLDER/'complete-audit.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
