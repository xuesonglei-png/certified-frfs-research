"""Run/audit/review the frozen all15 DC diagnostic; retain every failure."""
import json
import subprocess
import sys
import time
import traceback
from run_third_enemy_inner import ROOT, FOLDER, digest, frozen

STATUS = ROOT/'results/third-enemy-inner-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), 'goal_achieved': False, **values}, indent=2), encoding='utf-8')


def main():
    frozen()
    smoke = json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256'] == digest(FOLDER/'iris.json')
    for phase, script in [('running', 'run_third_enemy_inner.py'), ('auditing', 'audit_third_enemy_inner.py')]:
        log = ROOT/'results'/f'third-enemy-inner-{phase}.log'
        with log.open('a', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status(phase, child_pid=child.pid, log=log.relative_to(ROOT).as_posix())
            if child.wait():
                raise RuntimeError(f'{script} failed; preserve log/artifacts')
    summary = json.loads((FOLDER/'complete.json').read_text()); audit = json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(FOLDER/'complete.json')
    review = {'mean_anchor_gain_pp': summary['mean_anchor_gain_pp'], 'mean_vs_convex_pp': summary['mean_vs_convex_pp'],
              'improved_vs_convex_datasets': summary['improved_vs_convex_datasets'], 'target_optimizer_failures': summary['target_optimizer_failures'],
              'pilot_gate_passed': summary['pilot_gate_passed'], 'audit_sha256': digest(FOLDER/'complete-audit.json'),
              'goal_achieved': False, 'classification_superiority_established': False, 'new_outer_evaluation_performed': False,
              'decision': 'Evaluate fresh complete-grid and fully nested common-control classifier.' if summary['pilot_gate_passed']
                          else 'Do not promote DC diagnostic. Review rank-loss/guard/support mechanism with all failures.',
              'limits': ['Inner reselection and inclusion of anchor do not prove accuracy advantages.',
                         'All15 repeatedly exposed development; final independent data required.',
                         'CCCP fixed-budget descent is not global optimization or verified novel theory.',
                         'Frozen prior support selection is sequential, not fresh joint nested selection.']}
    (ROOT/'research/third-enemy-inner-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    status('complete_audited_reviewed', pilot_gate_passed=summary['pilot_gate_passed'], audit_sha256=digest(FOLDER/'complete-audit.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
