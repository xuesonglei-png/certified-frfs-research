"""Await audited source, freeze, run all15 repair and full independent replay."""
import json
import subprocess
import sys
import time
import traceback
from run_frs_zero_face_repair_experiment import ROOT, SOURCE, FOLDER, FREEZE, PROTOCOL, digest, prerequisites, frozen
from continue_soft_topk_knn_pilot import verify

STATUS = ROOT/'results/frs-zero-face-repair-chain-status.json'
TESTS = ROOT/'research/frs-zero-face-repair-tests.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase':phase, 'updated_unix':time.time(),
        'goal_achieved':False, 'classification_scored':False, **values}, indent=2), encoding='utf-8')


def execute(phase, script, args=()):
    frozen(); log = ROOT/'results'/f'frs-zero-face-repair-{phase}.log'
    with log.open('a', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script), *args], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT)
        status(phase, child_pid=child.pid, script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; all artifacts and logs retained')


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    tests = json.loads(TESTS.read_text()); assert tests['passed'] and tests['tests_passed'] == 2
    for rel, sha in tests['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    status('waiting_for_full_coordinate_source_audit')
    review_path = ROOT/'research/FRS-coordinate-minimum-review-v2.json'
    while not review_path.exists():
        p = ROOT/'results/frs-coordinate-diagnostic-v2-chain-status.json'
        if p.exists() and json.loads(p.read_text())['phase'] == 'failed':
            raise RuntimeError('Source audit failed; no repair attempted')
        time.sleep(10)
    parent = prerequisites()
    base = json.loads((ROOT/'research/frs-coordinate-diagnostic-v2-source-freeze.json').read_text())['fingerprints']
    for rel, sha in base.items():
        assert digest(ROOT/rel) == sha, rel
    paths = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT/folder).rglob('*.py')]
    paths += [PROTOCOL, TESTS, review_path, SOURCE/'complete.json', SOURCE/'complete-audit.json']
    paths += [SOURCE/(n+'.json') for n in sorted(parent['artifact_hashes'])]
    hashes = {**base, **{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    FREEZE.write_text(json.dumps({'created_unix':time.time(), 'fingerprints':hashes,
        'scope':'ALL15FIRSTfoldall4regimesoriginal_constraints_exact_zero_numerical_repair',
        'goal_achieved':False}, indent=2), encoding='utf-8')
    print('Froze zero-face repair', len(hashes), 'files before fitting', flush=True)
    execute('smoke_solving', 'run_frs_zero_face_repair_experiment.py', ('--dataset', 'iris'))
    execute('smoke_auditing', 'audit_frs_zero_face_repair_experiment.py', ('--dataset', 'iris'))
    smoke = json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256'] == digest(FOLDER/'iris.json')
    execute('solving', 'run_frs_zero_face_repair_experiment.py')
    execute('auditing', 'audit_frs_zero_face_repair_experiment.py')
    audit = verify(FOLDER); summary = json.loads((FOLDER/'complete.json').read_text())
    review = {'audit_sha256':digest(FOLDER/'complete-audit.json'), 'all_dataset_rows':summary['rows'],
        'counts':summary['counts'], 'independent_LP_duals':audit['counts'].get('independent_zero_repair_LP_duals', 0),
        'goal_achieved':False, 'classification_scored':False, 'limits':summary['limits'],
        'decision':'Review numerical repair and actual per-vector support; no classifier promotion or replacement of queued sparse starts.'}
    (ROOT/'research/FRS-zero-face-repair-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    status('complete_audited_reviewed', audit_sha256=review['audit_sha256'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
