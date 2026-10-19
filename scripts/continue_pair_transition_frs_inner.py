"""Frozen smoke/full independent classifier development; never outer promotion."""
import json
import subprocess
import sys
import time
import traceback
from run_pair_transition_frs_inner import ROOT, FOLDER, frozen, prerequisites, digest
from continue_soft_topk_knn_pilot import verify

STATUS = ROOT/'results/pair-transition-frs-inner-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase':phase, 'updated_unix':time.time(),
        'goal_achieved':False, 'new_outer_evaluation_performed':False, **values}, indent=2), encoding='utf-8')


def execute(phase, script, args=()):
    frozen(); log = ROOT/'results'/f'pair-transition-frs-inner-{phase}.log'
    with log.open('a', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script), *args], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT)
        status(phase, child_pid=child.pid, log=log.relative_to(ROOT).as_posix(), script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; retain all logs and artifacts')


def main():
    prerequisites(); frozen()
    execute('smoke_scoring', 'run_pair_transition_frs_inner.py', ('--dataset', 'iris'))
    execute('smoke_auditing', 'audit_pair_transition_frs_inner.py', ('--dataset', 'iris'))
    smoke = json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256'] == digest(FOLDER/'iris.json')
    execute('inner_scoring', 'run_pair_transition_frs_inner.py')
    execute('auditing', 'audit_pair_transition_frs_inner.py')
    execute('reviewing', 'review_pair_transition_frs_inner.py')
    verify(FOLDER); review = json.loads((ROOT/'research/pair-transition-FRS-inner-review.json').read_text())
    assert review['audit_sha256'] == digest(FOLDER/'complete-audit.json')
    status('complete_audited_reviewed', pilot_gate_passed=review['pilot_gate_passed'],
        audit_sha256=review['audit_sha256'], classification_superiority_established=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
