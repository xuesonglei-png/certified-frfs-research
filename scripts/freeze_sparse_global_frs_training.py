import json
import time
from run_sparse_global_frs_training import ROOT, CAPPED, DIAGNOSTIC, FOLDER, FREEZE, prerequisites, digest


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    audits = prerequisites(); evidence = ROOT/'research/sparse-global-FRS-prerequisite-review.json'
    assert evidence.exists()
    base = json.loads((ROOT/'research/frs-coordinate-diagnostic-v2-source-freeze.json').read_text())['fingerprints']
    for rel, sha in base.items():
        assert digest(ROOT/rel) == sha, rel
    paths = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT/folder).rglob('*.py')]
    paths += [evidence, ROOT/'research/sparse-global-FRS-training-protocol.md',
        ROOT/'research/sparse-global-FRS-prototype-tests.json',
        ROOT/'research/FRS-coordinate-minimum-review-v2.json', ROOT/'research/latent-FRS-training-review.json',
        ROOT/'research/pair-transition-FRS-training-review.json', ROOT/'research/capped-frs-training-review.json']
    for folder in (CAPPED, DIAGNOSTIC, ROOT/'results/pair-transition-frs-training', ROOT/'results/latent-frs-training'):
        paths += [folder/'complete.json', folder/'complete-audit.json']
        paths += [folder/(n+'.json') for n in sorted(audits[folder.name]['artifact_hashes'])]
    hashes = {**base, **{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    FREEZE.write_text(json.dumps({'created_unix':time.time(), 'fingerprints':hashes,
        'scope':'ALL15FIRSTfoldmatched3arms3starts_globalcell_and_supportobjective_trainingonly',
        'goal_achieved':False}, indent=2), encoding='utf-8')
    print('Froze sparse/global training attribution', len(hashes), 'files', flush=True)


if __name__ == '__main__':
    main()
