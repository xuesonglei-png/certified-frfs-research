"""Package current code and evidence, without manuscript/LaTeX artifacts."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from datetime import datetime,timezone
from run_decision_iteration import ROOT


def main(name='prediction-development-code-evidence.zip'):
    if Path(name).name!=name or not name.endswith('.zip'):raise ValueError('ZIP basename required')
    paths=set()
    for pattern in ('src/certified_frfs/*.py','scripts/*.py','tests/*.py','data/manifest.json',
                    'data/processed/*','pyproject.toml','requirements-lock.txt'):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    for result_directory in ('predictive-iteration28','predictive-iteration29','predictive-iteration30',
                 'predictive-iteration31','predictive-iteration32','fgs-nested-comparison',
                 'matched-size-comparison','matched-uncertified-ablation','shared-metric-certification',
                 'adaptive-witness-pilot','flexible-witness-pilot','predictive-iteration33','majority-query-audit33'):
        paths.update(p for p in (ROOT/'results'/result_directory).glob('*') if p.is_file())
    paths.update(ROOT/'research'/name for name in ('iteration28-31-method-rationale.md',
        'next-experiment-gates.md','code-experiment-entrypoints.md',
        'utility-search-iteration33-protocol.md','utility-search-iteration33-review.md'))
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'development_evidence_not_submission_or_confirmation',
        'limitations':'current code snapshot; generation hashes may refer to earlier compatible sources; no claim of byte-identical historical source recovery',
        'files':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}}
    directory=ROOT/'output';directory.mkdir(exist_ok=True)
    target=directory/name
    if target.exists():raise ValueError('preserve existing evidence package')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):archive.write(p,str(p.relative_to(ROOT)))
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for archive_name,digest in manifest['files'].items():
            assert hashlib.sha256(archive.read(archive_name)).hexdigest()==digest
    print(json.dumps({'package':str(target),'verified_files':len(paths),'bytes':target.stat().st_size}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='prediction-development-code-evidence.zip')
    main(p.parse_args().name)
