import json
import subprocess
import sys
import time
import traceback
from run_sparse_global_frs_training import ROOT, FOLDER, frozen, prerequisites, digest
from continue_soft_topk_knn_pilot import verify

STATUS = ROOT/'results/sparse-global-frs-training-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase':phase, 'updated_unix':time.time(),
        'goal_achieved':False, 'classification_scored':False, **values}, indent=2), encoding='utf-8')


def execute(phase, script, args=()):
    frozen(); log = ROOT/'results'/f'sparse-global-frs-training-{phase}.log'
    with log.open('a', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script), *args], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT)
        status(phase, child_pid=child.pid, script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; preserve all artifacts')


def main():
    prerequisites(); frozen()
    execute('smoke_training', 'run_sparse_global_frs_training.py', ('--dataset', 'iris'))
    execute('smoke_auditing', 'audit_sparse_global_frs_training.py', ('--dataset', 'iris'))
    smoke = json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256'] == digest(FOLDER/'iris.json')
    execute('training', 'run_sparse_global_frs_training.py')
    execute('auditing', 'audit_sparse_global_frs_training.py')
    verify(FOLDER); summary = json.loads((FOLDER/'complete.json').read_text())
    review = {'audit_sha256':digest(FOLDER/'complete-audit.json'), 'all_dataset_rows':summary['rows'],
        'dataset_balanced_metrics':summary['dataset_balanced_metrics'],
        'optimizer_failures':summary['optimizer_failures'], 'available_regions':summary['available_regions'],
        'folds_with_exact_zero_weights':summary['folds_with_exact_zero_weights'],
        'goal_achieved':False, 'classification_scored':False, 'classification_superiority_established':False,
        'decision':'Review sparsity versus raw training-bound tradeoff on every dataset before any matched classifier family.',
        'limits':summary['limits']}
    (ROOT/'research/sparse-global-FRS-training-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    status('complete_audited_reviewed', audit_sha256=review['audit_sha256'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
