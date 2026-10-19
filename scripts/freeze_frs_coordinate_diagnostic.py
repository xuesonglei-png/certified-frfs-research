import json
import time
from run_frs_coordinate_diagnostic import ROOT, SOURCE, FOLDER, FREEZE, prerequisites, digest


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    audit = prerequisites()
    base = json.loads((ROOT/'research/latent-frs-training-source-freeze.json').read_text())['fingerprints']
    for rel, sha in base.items():
        assert digest(ROOT/rel) == sha, rel
    paths = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT/folder).rglob('*.py')]
    paths += [ROOT/'research/FRS-coordinate-minimum-protocol.md', SOURCE/'complete.json',
        SOURCE/'complete-audit.json', ROOT/'research/capped-frs-training-review.json']
    paths += [SOURCE/(name+'.json') for name in sorted(audit['artifact_hashes'])]
    hashes = {**base, **{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    FREEZE.write_text(json.dumps({'created_unix':time.time(), 'fingerprints':hashes,
        'scope':'ALL15FIRSTfold per_coordinate_fixed_cell_diagnostic not_classifier_or_joint_reduct',
        'goal_achieved':False}, indent=2), encoding='utf-8')
    print('Froze coordinate diagnostic', len(hashes), 'files before real-data solves', flush=True)


if __name__ == '__main__':
    main()
