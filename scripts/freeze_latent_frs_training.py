"""Freeze only after complete capped review; never overwrite a freeze."""
import json
import time
from run_latent_frs_training import ROOT, SOURCE, FREEZE, REVIEW, prerequisite, digest


def main():
    assert not FREEZE.exists(), 'Existing scientific freeze must not be overwritten'
    audit = prerequisite(); protocol = ROOT/'research/latent-FRS-training-protocol.md'
    assert protocol.exists(), 'Fix prospective protocol after capped review first'
    assert not (ROOT/'results/latent-frs-training').exists(), 'No real-data latent artifacts before freeze'
    base = json.loads((ROOT/'research/capped-frs-training-source-freeze.json').read_text())['fingerprints']
    for rel, sha in base.items():
        assert digest(ROOT/rel) == sha, rel
    paths = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT/folder).rglob('*.py')]
    paths += [protocol, REVIEW, SOURCE/'complete.json', SOURCE/'complete-audit.json',
        ROOT/'research/strict-capped-control-provenance.json', ROOT/'research/latent-friend-FRS-prototype.md',
        ROOT/'research/latent-friend-exact-mechanism.md']
    paths += [SOURCE/(name+'.json') for name in sorted(audit['artifact_hashes'])]
    hashes = {**base, **{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    result = {'created_unix': time.time(), 'fingerprints': hashes,
        'capped_audit_sha256': digest(SOURCE/'complete-audit.json'), 'capped_review_sha256': digest(REVIEW),
        'scope': 'ALL15 matched strict solver two-start training ablation; no classifier scoring', 'goal_achieved': False}
    FREEZE.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Prospectively froze', len(hashes), 'files; no latent real-data fits yet', flush=True)


if __name__ == '__main__':
    main()
