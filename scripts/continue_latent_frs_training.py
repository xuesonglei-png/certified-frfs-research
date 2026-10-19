"""Execute frozen matched training smoke, full cohort, audits and review."""
import json
import subprocess
import sys
import time
import traceback
from run_latent_frs_training import ROOT, FOLDER, frozen, prerequisite, digest
from continue_soft_topk_knn_pilot import verify

STATUS = ROOT/'results/latent-frs-training-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), 'goal_achieved': False,
        'classification_scored': False, **values}, indent=2), encoding='utf-8')


def execute(phase, script, args=()):
    frozen(); log = ROOT/'results'/f'latent-frs-training-{phase}.log'
    with log.open('a', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script), *args], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT)
        status(phase, child_pid=child.pid, log=log.relative_to(ROOT).as_posix(), script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; preserve artifacts and logs')


def main():
    prerequisite(); frozen()
    execute('smoke_training', 'run_latent_frs_training.py', ('--dataset', 'iris'))
    execute('smoke_auditing', 'audit_latent_frs_training.py', ('--dataset', 'iris'))
    smoke = json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256'] == digest(FOLDER/'iris.json')
    execute('training', 'run_latent_frs_training.py')
    execute('auditing', 'audit_latent_frs_training.py')
    execute('reviewing', 'review_latent_frs_training.py')
    verify(FOLDER); review = json.loads((ROOT/'research/latent-FRS-training-review.json').read_text())
    assert review['audit_sha256'] == digest(FOLDER/'complete-audit.json')
    status('complete_audited_reviewed', audit_sha256=review['audit_sha256'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
