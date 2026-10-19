"""Frozen paired residual-cell smoke/ALL15 run/audit/all-row review."""
import json
import subprocess
import sys
import time
import traceback
from run_pair_transition_frs_training import ROOT, FOLDER, frozen, prerequisite, digest
from continue_soft_topk_knn_pilot import verify

STATUS = ROOT/'results/pair-transition-frs-training-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase':phase, 'updated_unix':time.time(), 'goal_achieved':False,
        'classification_scored':False, **values}, indent=2), encoding='utf-8')


def execute(phase, script, args=()):
    frozen(); log = ROOT/'results'/f'pair-transition-frs-training-{phase}.log'
    with log.open('a', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script), *args], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT)
        status(phase, child_pid=child.pid, log=log.relative_to(ROOT).as_posix(), script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; retain all logs/results')


def main():
    frozen(); prerequisite()
    execute('smoke_training', 'run_pair_transition_frs_training.py', ('--dataset','iris'))
    execute('smoke_auditing', 'audit_pair_transition_frs_training.py', ('--dataset','iris'))
    smoke = json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256'] == digest(FOLDER/'iris.json')
    execute('training', 'run_pair_transition_frs_training.py')
    execute('auditing', 'audit_pair_transition_frs_training.py')
    verify(FOLDER); summary = json.loads((FOLDER/'complete.json').read_text())
    folds = [f for r in summary['rows'] for f in r['comparisons']]
    review = {'audit_sha256':digest(FOLDER/'complete-audit.json'), 'all_dataset_rows':summary['rows'],
        'dataset_balanced_metrics':summary['dataset_balanced_metrics'], 'optimizer_failures':summary['optimizer_failures'],
        'available_cells':summary['available_cells'],
        'new_cell_expansion_folds':sum(f['new_target_outside_one_sided_cell'] for f in folds),
        'better_training_objective_folds':sum(f['objective_decrease']>1e-9 for f in folds),
        'worse_training_objective_folds':sum(f['objective_decrease']< -1e-9 for f in folds),
        'better_clipped_training_bound_folds':sum(f['clipped_training_bound_decrease_pp']>1e-8 for f in folds),
        'worse_clipped_training_bound_folds':sum(f['clipped_training_bound_decrease_pp']< -1e-8 for f in folds),
        'new_folds_with_zero_weight':sum(f['new_zero_weights']>0 for f in folds),
        'classification_scored':False, 'classification_superiority_established':False, 'goal_achieved':False,
        'decision':'Review cell expansion, optimization, sparsity, computational costs and every failure before any equal-budget classifier extension.',
        'limits':['Existing pair-envelope theory; no verified novelty.', 'Training gains do not prove test superiority.',
            'All15 are exposed; no p-values or confirmation.', 'No automatic outer experiment promotion.',
            'Stable cell restrictions still apply; pair rows may be quadratic.', 'Floating point is not directed interval certification.']}
    (ROOT/'research/pair-transition-FRS-training-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    status('complete_audited_reviewed', audit_sha256=review['audit_sha256'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
