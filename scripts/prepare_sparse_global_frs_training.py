"""Wait for full prior audits, retain every row, freeze prospective attribution."""
import json
import subprocess
import sys
import time
import traceback
from run_sparse_global_frs_training import ROOT, CAPPED, DIAGNOSTIC, FREEZE, prerequisites, digest

STATUS = ROOT/'results/sparse-global-frs-training-preparation-status.json'
EVIDENCE = ROOT/'research/sparse-global-FRS-prerequisite-review.json'
TESTS = ROOT/'research/sparse-global-FRS-prototype-tests.json'
REVIEWS = ('capped-frs-training-review.json', 'FRS-coordinate-minimum-review-v2.json',
           'pair-transition-FRS-training-review.json', 'latent-FRS-training-review.json')


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase':phase, 'updated_unix':time.time(),
        'goal_achieved':False, 'classification_scored':False, **values}, indent=2), encoding='utf-8')


def main():
    assert not FREEZE.exists() and not EVIDENCE.exists()
    tests = json.loads(TESTS.read_text()); assert tests['passed'] and tests['tests_passed'] == 12
    for rel, sha in tests['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    status('waiting_for_complete_coordinate_pair_latent_reviews')
    while not all((ROOT/'research'/p).exists() for p in REVIEWS):
        for name in ('frs-coordinate-diagnostic-v2-chain-status.json', 'pair-transition-frs-training-chain-status.json',
                     'latent-frs-training-chain-status.json'):
            p = ROOT/'results'/name
            if p.exists() and json.loads(p.read_text())['phase'] == 'failed':
                raise RuntimeError(f'{name} failed; no new fitting started')
        time.sleep(10)
    prerequisites()
    for rel, sha in tests['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    reviews = {name:json.loads((ROOT/'research'/name).read_text()) for name in REVIEWS}
    assert all(len(r['all_dataset_rows']) == 15 for r in reviews.values())
    evidence = {'created_unix':time.time(), 'all_prerequisites_ready':True,
        'all_prior_dataset_rows':{name:r['all_dataset_rows'] for name, r in reviews.items()},
        'review_sha256':{name:digest(ROOT/'research'/name) for name in REVIEWS},
        'tests_sha256':digest(TESTS), 'goal_achieved':False, 'classification_scored':False,
        'decision':'Run predesigned ALL15firstfold3arm3start training ablation; retain every adverse tradeoff and unavailable region.'}
    EVIDENCE.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    for phase, script in (('freezing', 'freeze_sparse_global_frs_training.py'),
                          ('running_frozen_chain', 'continue_sparse_global_frs_training.py')):
        log = ROOT/'results'/f'sparse-global-frs-training-preparation-{phase}.log'
        with log.open('a', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
            status(phase, child_pid=child.pid, script_sha256=digest(ROOT/'scripts'/script))
            if child.wait():
                raise RuntimeError(f'{script} failed; retain all artifacts')
    status('complete_audited_reviewed')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
