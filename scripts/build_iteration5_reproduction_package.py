"""Create an isolated, non-overwriting historical iteration-5 reproduction bundle."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'output' / 'iteration5-reproduction-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    if DEST.exists():
        raise FileExistsError('Preserve existing bundle: ' + str(DEST))
    DEST.mkdir(parents=True)
    historical = ROOT / 'results/predictive-iteration5'
    records = [json.loads(p.read_text()) for p in sorted(historical.glob('*-s*-f*.json'))]
    assert len(records) == 42
    datasets = sorted({o['dataset'] for o in records})
    source_files = list((ROOT/'src/certified_frfs').glob('*.py')) + [
        ROOT/'scripts/run_decision_iteration.py',
        ROOT/'scripts/run_predictive_certified_iteration.py',
    ]
    for p in source_files:
        q = DEST/p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    shutil.copytree(historical, DEST/'reference/results')
    for name in ['predictive-iteration4-protocol.md', 'predictive-iteration5-protocol.md',
                 'predictive-iteration5-review.md', 'decision-iterations-status.md']:
        (DEST/'reference/research').mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'research'/name, DEST/'reference/research'/name)
    manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    dump(DEST/'data/manifest.json', {n:manifest[n] for n in datasets})
    for n in datasets:
        p = ROOT/'data/processed'/f'{n}.npz'
        assert digest(p) == manifest[n]['processed_sha256'], n
        (DEST/'data/processed').mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, DEST/'data/processed'/p.name)

    # Outer indices are loaded from saved results; missing inner indices are
    # reconstructed from the documented seed and frozen explicitly in the bundle.
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    splits = []
    for o in records:
        with np.load(DEST/'data/processed'/f"{o['dataset']}.npz") as z:
            y = z['y']; X = z['X']
        tr, te = np.array(o['train_indices']), np.array(o['test_indices'])
        assert sorted(np.r_[tr, te].tolist()) == list(range(len(y)))
        generated = list(StratifiedKFold(3, shuffle=True, random_state=o['seed']).split(X,y))[o['fold']]
        assert np.array_equal(tr, generated[0]) and np.array_equal(te, generated[1])
        inner = StratifiedKFold(min(3, np.unique(y[tr],return_counts=True)[1].min()),
                               shuffle=True, random_state=o['seed']+1)
        splits.append({'dataset':o['dataset'], 'seed':o['seed'], 'fold':o['fold'],
                       'train_indices':tr.tolist(), 'test_indices':te.tolist(),
                       'inner_indices_relative_to_train':[
                           {'train':a.tolist(), 'validation':b.tolist()} for a,b in inner.split(X[tr],y[tr])]})
    dump(DEST/'splits.json', splits)

    archived_path = ROOT/'output/prediction-development-code-evidence.zip'
    comparisons = []
    with zipfile.ZipFile(archived_path) as archive:
        for p in source_files:
            rel = p.relative_to(ROOT).as_posix()
            old = archive.read(rel) if rel in archive.namelist() else None
            comparisons.append({'path':rel,'current_sha256':digest(p),
                'archived_sha256':hashlib.sha256(old).hexdigest() if old is not None else None,
                'status':'unchanged_since_available_archive' if old == p.read_bytes() else
                         'not_in_available_archive' if old is None else 'changed_since_available_archive'})
    current_aggregate = hashlib.sha256((DEST/'scripts/run_decision_iteration.py').read_bytes()+b''.join(
        p.read_bytes() for p in sorted((DEST/'src/certified_frfs').glob('*.py')))).hexdigest()
    dump(DEST/'source-audit.json', {
        'historical_aggregate_sha256':sorted({o['code_sha256'] for o in records}),
        'frozen_current_aggregate_sha256':current_aggregate,
        'exact_historical_source_snapshot_recovered':False,
        'available_archive':archived_path.name, 'archive_sha256':digest(archived_path),
        'interpretation':'Archive is a later development snapshot, not proof of exact iteration-5 source identity. '
                         'Aggregate hash includes all package modules, even unused later additions. '
                         'Behavioral reproduction must be established separately.',
        'files':comparisons})
    versions = {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    dump(DEST/'environment.json', {'python':sys.version,'platform':platform.platform(),
                                  'packages':dict(sorted(versions.items()))})
    core = ['numpy','scipy','scikit-learn','joblib','threadpoolctl','pytest']
    (DEST/'requirements.txt').write_text(''.join(f'{n}=={importlib.metadata.version(n)}\n' for n in core),encoding='utf-8')
    for name in ['test_decision_coverage.py','test_margin_certificates.py','test_recent_sfrip.py']:
        (DEST/'tests').mkdir(exist_ok=True)
        shutil.copy2(ROOT/'tests'/name,DEST/'tests'/name)
    (DEST/'pytest.ini').write_text('[pytest]\npythonpath = src\ntestpaths = tests\n',encoding='utf-8')
    dump(DEST/'original-results-sha256.json', {
        p.relative_to(ROOT).as_posix():digest(p) for p in sorted(historical.glob('*')) if p.is_file()})
    dump(DEST/'frozen-inputs-sha256.json', {
        p.relative_to(DEST).as_posix():digest(p) for p in sorted(DEST.rglob('*')) if p.is_file()})
    print(json.dumps({'package':str(DEST),'splits':len(splits),'datasets':datasets,
                      'sources':len(source_files)},ensure_ascii=False))


if __name__ == '__main__':
    main()
