import json
import time
from run_frs_coordinate_diagnostic_v2 import ROOT, SOURCE, FOLDER, FREEZE, prerequisites, digest


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    audit = prerequisites()
    base = json.loads((ROOT/'research/frs-coordinate-diagnostic-source-freeze.json').read_text())['fingerprints']
    for rel, sha in base.items():
        assert digest(ROOT/rel) == sha, rel
    paths = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT/folder).rglob('*.py')]
    paths += [ROOT/'research/FRS-coordinate-minimum-protocol.md',
        ROOT/'research/frs-coordinate-diagnostic-v2-provenance.json',
        ROOT/'research/FRS-coordinate-minimum-v2-protocol.md',
        ROOT/'results/frs-coordinate-diagnostic/iris.json',
        ROOT/'results/frs-coordinate-diagnostic/iris.smoke-audit.json',
        SOURCE/'complete.json', SOURCE/'complete-audit.json']
    paths += [SOURCE/(n+'.json') for n in sorted(audit['artifact_hashes'])]
    hashes = {**base, **{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    FREEZE.write_text(json.dumps({'created_unix':time.time(), 'fingerprints':hashes,
        'scope':'same_ALL15FIRSTfoldLPdiagnostic_numpy_scalar_JSON_conversion_only', 'goal_achieved':False}, indent=2), encoding='utf-8')
    print('Froze serialization-v2 diagnostic', len(hashes), 'files before resumed full cohort', flush=True)


if __name__ == '__main__':
    main()
