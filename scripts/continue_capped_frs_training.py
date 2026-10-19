"""Wait for audited training predecessor, complete capped fits/audit/review."""
import json
import subprocess
import sys
import time
import traceback
import numpy as np
from run_capped_frs_training import ROOT, FOLDER, SOURCE, frozen, digest
from continue_soft_topk_knn_pilot import verify

STATUS = ROOT/'results/capped-frs-training-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), 'goal_achieved': False,
        'classification_scored': False, **values}, indent=2), encoding='utf-8')


def main():
    frozen(); smoke = json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256'] == digest(FOLDER/'iris.json')
    status('waiting_for_previous_independent_training_audit')
    while not (SOURCE/'complete-audit.json').exists():
        old_status = json.loads((ROOT/'results/frs-one-sided-training-v2-chain-status.json').read_text())
        if old_status['phase'] == 'failed':
            raise RuntimeError('Prerequisite audit failed; capped ALL15 not started')
        time.sleep(10)
    verify(SOURCE)
    for phase, script in [('training', 'run_capped_frs_training.py'), ('auditing', 'audit_capped_frs_training.py')]:
        frozen(); log = ROOT/'results'/f'capped-frs-training-{phase}.log'
        with log.open('a', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status(phase, child_pid=child.pid, log=log.relative_to(ROOT).as_posix(), executed_script_sha256=digest(ROOT/'scripts'/script))
            if child.wait():
                raise RuntimeError(f'{script} failed; preserve artifacts/logs')
    summary = json.loads((FOLDER/'complete.json').read_text()); audit = verify(FOLDER)
    comparisons = [c for r in summary['rows'] for c in r['comparisons']]
    review = {'audit_sha256': digest(FOLDER/'complete-audit.json'), 'all_dataset_rows': summary['rows'],
        'mean_dataset_clipped_training_bound_decrease_pp': summary['mean_dataset_clipped_training_bound_decrease_pp'],
        'mean_dataset_clipped_objective_decrease': summary['mean_dataset_clipped_objective_decrease'],
        'optimizer_failures': summary['optimizer_failures'], 'available_cells': summary['available_cells'],
        'better_bound_folds': sum(c['clipped_training_bound_decrease_pp'] > 1e-8 for c in comparisons),
        'worse_bound_folds': sum(c['clipped_training_bound_decrease_pp'] < -1e-8 for c in comparisons),
        'strict_CCCP_improvement_folds': sum(c['start_to_final_objective_decrease'] > 1e-9 for c in comparisons),
        'warm_selected_folds': sum(c['selected_start'] == 'warm' for c in comparisons),
        'goal_achieved': False, 'classification_scored': False, 'classification_superiority_established': False,
        'decision': 'Compare objective alignment and stalls; review symmetric classifier results before adding any family equally to controls.',
        'limits': ['Training upper bounds do not imply test accuracy.', 'All15 are exposed.',
                   'Warm-start improvement is recorded separately from CCCP improvement.', 'Capped/DC generic theory is prior art.']}
    (ROOT/'research/capped-frs-training-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    status('complete_audited_reviewed', audit_sha256=review['audit_sha256'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
