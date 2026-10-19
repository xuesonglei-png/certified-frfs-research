import json
import subprocess
import sys
import time
import traceback
from run_frs_coordinate_diagnostic import ROOT, FOLDER, frozen, prerequisites, digest
from continue_soft_topk_knn_pilot import verify

STATUS = ROOT/'results/frs-coordinate-diagnostic-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase':phase, 'updated_unix':time.time(),
        'goal_achieved':False, 'classification_scored':False, **values}, indent=2), encoding='utf-8')


def execute(phase, script, args=()):
    frozen(); log = ROOT/'results'/f'frs-coordinate-diagnostic-{phase}.log'
    with log.open('a', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script), *args], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT)
        status(phase, child_pid=child.pid, script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; preserve all artifacts and logs')


def main():
    prerequisites(); frozen()
    execute('smoke_solving', 'run_frs_coordinate_diagnostic.py', ('--dataset', 'iris'))
    execute('smoke_auditing', 'audit_frs_coordinate_diagnostic.py', ('--dataset', 'iris'))
    smoke = json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256'] == digest(FOLDER/'iris.json')
    execute('solving', 'run_frs_coordinate_diagnostic.py')
    execute('auditing', 'audit_frs_coordinate_diagnostic.py')
    audit = verify(FOLDER); summary = json.loads((FOLDER/'complete.json').read_text())
    review = {'audit_sha256':digest(FOLDER/'complete-audit.json'), 'all_dataset_rows':summary['rows'],
        'counts':summary['counts'], 'independent_LP_replays':audit['counts']['independent_coordinate_LP_primal_dual_replays'],
        'goal_achieved':False, 'classification_scored':False,
        'decision':'Locate fixed-cell/majority/class-envelope obstruction separately from learned-loss sparsity before redesign.',
        'limits':summary['limits']}
    (ROOT/'research/FRS-coordinate-minimum-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    status('complete_audited_reviewed', audit_sha256=review['audit_sha256'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
