"""Complete all training fits, independent audits and all-row review."""
import json
import subprocess
import sys
import time
import traceback
import numpy as np
from run_frs_one_sided_training import ROOT, FOLDER, frozen, digest

STATUS = ROOT/'results/frs-one-sided-training-v2-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), 'goal_achieved': False,
        'classification_scored': False, **values}, indent=2), encoding='utf-8')


def main():
    frozen(); smoke = json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256'] == digest(FOLDER/'iris.json')
    for phase, script in [('training', 'run_frs_one_sided_training.py'), ('auditing', 'audit_frs_one_sided_training.py')]:
        frozen(); log = ROOT/'results'/f'frs-one-sided-training-v2-{phase}.log'
        with log.open('a', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status(phase, child_pid=child.pid, log=log.relative_to(ROOT).as_posix(), executed_script_sha256=digest(ROOT/'scripts'/script))
            if child.wait():
                raise RuntimeError(f'{script} failed; preserve log and artifacts')
    summary = json.loads((FOLDER/'complete.json').read_text()); audit = json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(FOLDER/'complete.json')
    comparisons = [c for row in summary['rows'] for c in row['comparisons']]
    gains = [c['objective_decrease_vs_symmetric'] for c in comparisons]
    review = {'audit_sha256': digest(FOLDER/'complete-audit.json'), 'all_dataset_rows': summary['rows'],
        'available_cells': summary['available_cells'], 'optimizer_failures': summary['optimizer_failures'],
        'mean_training_objective_decrease': float(np.mean(gains)) if gains else None,
        'better_folds': sum(v > 1e-9 for v in gains), 'worse_folds': sum(v < -1e-9 for v in gains),
        'target_feasible_set_expansions': sum(c['new_target_outside_symmetric_cell'] for c in comparisons),
        'goal_achieved': False, 'classification_scored': False, 'classification_superiority_established': False,
        'decision': 'Review all symmetric inner results before adding this family equally to all weighted arms.',
        'limits': ['Training loss is not test accuracy.', 'All15 are exposed development data.',
                   'Feasible-set inclusion does not guarantee CCCP finds the best trajectory.',
                   'Monotonicity and generic optimization do not establish new-theorem originality.']}
    (ROOT/'research/frs-one-sided-training-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    status('complete_audited_reviewed', audit_sha256=review['audit_sha256'], better_folds=review['better_folds'], worse_folds=review['worse_folds'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
