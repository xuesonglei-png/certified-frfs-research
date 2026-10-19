"""Freeze new classifier protocol only after every prerequisite audit/review."""
import json
import time
from run_pair_transition_frs_inner import (ROOT, SOURCE, PREVIOUS, CANONICAL, TRAINING,
    FOLDER, FREEZE, prerequisites, digest)


def main():
    assert not FREEZE.exists() and not FOLDER.exists(), 'Never overwrite a scientific freeze or fitted cohort'
    audits = prerequisites(); protocol = ROOT/'research/pair-transition-FRS-inner-protocol.md'
    assert protocol.exists()
    evidence = ROOT/'research/pair-transition-FRS-inner-prerequisite-review.json'
    assert evidence.exists(), 'Review all prerequisite rows before freezing'
    review = json.loads(evidence.read_text())
    assert review['all_prerequisites_ready'] and not review['goal_achieved']
    assert review['prototype_tests_passed']
    base = json.loads((ROOT/'research/pair-transition-frs-training-source-freeze.json').read_text())['fingerprints']
    for rel, sha in base.items():
        assert digest(ROOT/rel) == sha, rel
    paths = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT/folder).rglob('*.py')]
    paths += [protocol, evidence, ROOT/'research/pair-transition-native-control-alignment.json',
        ROOT/'research/monotone-frs-inner-review.json', ROOT/'research/pair-transition-FRS-training-review.json',
        ROOT/'research/pair-transition-FRS-inner-tests.json',
        ROOT/'research/capped-frs-training-review.json', ROOT/'results/capped-frs-training/complete-audit.json']
    for folder in (SOURCE, PREVIOUS, CANONICAL, TRAINING):
        paths += [folder/'complete.json', folder/'complete-audit.json']
        paths += [folder/(name+'.json') for name in sorted(audits[folder.name]['artifact_hashes'])]
    hashes = {**base, **{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    FREEZE.write_text(json.dumps({'created_unix':time.time(), 'fingerprints':hashes,
        'scope':'append-only equal35 refined arms full native controls exposed INNER development',
        'goal_achieved':False}, indent=2), encoding='utf-8')
    print('Prospectively froze', len(hashes), 'files; no new real-data classifier fitting yet', flush=True)


if __name__ == '__main__':
    main()
