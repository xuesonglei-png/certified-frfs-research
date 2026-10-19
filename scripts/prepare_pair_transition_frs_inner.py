"""Wait for full reviews, preserve every row, freeze and run prospective design.

This queue never accesses outer scores and never changes the preset screen.
Every prerequisite error stops the chain rather than bypassing the audit.
"""
import json
import subprocess
import sys
import time
import traceback
from run_pair_transition_frs_inner import (ROOT, SOURCE, PREVIOUS, CANONICAL,
    TRAINING, FREEZE, prerequisites, digest)

STATUS = ROOT/'results/pair-transition-frs-inner-preparation-status.json'
EVIDENCE = ROOT/'research/pair-transition-FRS-inner-prerequisite-review.json'
TESTS = ROOT/'research/pair-transition-FRS-inner-tests.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase':phase, 'updated_unix':time.time(), 'goal_achieved':False,
        'new_outer_evaluation_performed':False, **values}, indent=2), encoding='utf-8')


def ready():
    paths = [folder/'complete-audit.json' for folder in (SOURCE, PREVIOUS, CANONICAL, TRAINING)]
    paths += [ROOT/'research/monotone-frs-inner-review.json',
        ROOT/'research/pair-transition-FRS-training-review.json',
        ROOT/'research/capped-frs-training-review.json',
        ROOT/'research/pair-transition-native-control-alignment.json']
    return all(p.exists() for p in paths)


def main():
    assert not FREEZE.exists() and not EVIDENCE.exists(), 'Do not rerun or overwrite an existing scientific preparation'
    tests = json.loads(TESTS.read_text()); assert tests['passed'] and tests['tests_passed'] == 4
    for rel, sha in tests['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    status('waiting_for_complete_training_classifier_and_native_reviews')
    chains = ('monotone-frs-inner-chain-status.json', 'pair-transition-frs-training-chain-status.json',
              'majority-hinge-canonical-benchmark-chain-status.json', 'capped-frs-training-chain-status.json')
    while not ready():
        for name in chains:
            path = ROOT/'results'/name
            if path.exists() and json.loads(path.read_text())['phase'] == 'failed':
                raise RuntimeError(f'Prerequisite {name} failed; no new real-data fitting started')
        time.sleep(10)
    prerequisites()
    for rel, sha in tests['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    mono = json.loads((ROOT/'research/monotone-frs-inner-review.json').read_text())
    pair = json.loads((ROOT/'research/pair-transition-FRS-training-review.json').read_text())
    capped = json.loads((ROOT/'research/capped-frs-training-review.json').read_text())
    assert len(mono['all_dataset_rows']) == len(pair['all_dataset_rows']) == len(capped['all_dataset_rows']) == 15
    evidence = {'created_unix':time.time(), 'all_prerequisites_ready':True,
        'prototype_tests_passed':True, 'tests_sha256':digest(TESTS),
        'all_monotone_classifier_rows':mono['all_dataset_rows'],
        'all_pair_training_rows':pair['all_dataset_rows'], 'all_capped_training_rows':capped['all_dataset_rows'],
        'monotone_screen_passed':mono['pilot_gate_passed'],
        'monotone_mean_proposed_minus_controls_pp':mono['mean_proposed_minus_controls_pp'],
        'pair_training_metrics':pair['dataset_balanced_metrics'],
        'pair_training_failures':pair['optimizer_failures'],
        'pair_training_folds_with_zero_weights':pair['new_folds_with_zero_weight'],
        'canonical_audit_sha256':digest(CANONICAL/'complete-audit.json'),
        'canonical_scope':'only full native INNER candidate searches will be reused; no outer scores copied',
        'goal_achieved':False, 'new_outer_evaluation_performed':False,
        'decision':'Run prospectively specified append-only inner screen despite prior losses; no outer promotion.'}
    EVIDENCE.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    for phase, script in (('freezing', 'freeze_pair_transition_frs_inner.py'),
                          ('running_frozen_chain', 'continue_pair_transition_frs_inner.py')):
        log = ROOT/'results'/f'pair-transition-frs-inner-preparation-{phase}.log'
        with log.open('a', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
            status(phase, child_pid=child.pid, script_sha256=digest(ROOT/'scripts'/script))
            if child.wait():
                raise RuntimeError(f'{script} failed; preserve all files')
    status('complete_audited_reviewed')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
