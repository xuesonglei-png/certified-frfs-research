"""Wait for original star independent audit, then run/audit signed refinement."""
import json
import subprocess
import sys
import time
import traceback
from diagnose_signed_classwise_admission import ROOT, STARS, FOLDER, FREEZE, digest
from continue_hardaware_soft5_knn_pilot import wait_verified

STATUS = ROOT/'results/signed-classwise-admission-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), 'goal_achieved': False, **values}, indent=2), encoding='utf-8')


def main(pid):
    freeze = json.loads(FREEZE.read_text())
    for rel, sha in freeze['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    status('waiting_for_star_independent_audit', prerequisite_pid=pid); wait_verified(STARS, pid)
    for phase, script in [('running', 'diagnose_signed_classwise_admission.py'), ('auditing', 'audit_signed_classwise_admission.py')]:
        log = ROOT/'results'/f'signed-classwise-admission-{phase}.log'
        with log.open('a', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status(phase, child_pid=child.pid, log=log.relative_to(ROOT).as_posix())
            if child.wait():
                raise RuntimeError(f'{script} failed; preserve all failure evidence')
    summary = json.loads((FOLDER/'complete.json').read_text()); audit = json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(FOLDER/'complete.json')
    review = {'counts': summary['counts'], 'audit_sha256': digest(FOLDER/'complete-audit.json'), 'goal_achieved': False,
        'classification_scored': False, 'scope': 'stronger_classwise_FRS_bound_not_prediction_accuracy',
        'decision': 'Quantify identical-rule classifier opportunity for all newly feasible metrics before a fresh nested study.'
                    if summary['counts']['beyond_star_admissions'] else 'No expansion beyond stars; do not claim extra experimental utility.',
        'limits': ['Signed compensation occurs only within each class.', 'Saturation fallback and original majority unchanged.',
                   'Finite reference/class-prior guarantee is not population risk or held-out accuracy.', 'Originality remains unverified.']}
    (ROOT/'research/signed-classwise-admission-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    status('complete_audited_reviewed', counts=summary['counts'])


if __name__ == '__main__':
    try:
        main(int(sys.argv[1]))
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
