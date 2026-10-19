"""Rebuild baseline choices and scores without trusting summary files."""
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from compare_trajectory_recent import ROOT, FOLDER, fingerprints, GRIDS, SELECTORS, score, utility
from certified_frfs.datasets import load_dataset


def scaling(A, B):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = MinMaxScaler(clip=True)
    return scaler.fit_transform(imputer.fit_transform(A)), scaler.transform(imputer.transform(B))


def main():
    threadpool_limits(1); counts = {'inner_refits': 0, 'final_refits': 0, 'scores': 0, 'splits': 0}
    files = sorted(FOLDER.glob('*.json'))
    for path in files:
        if path.name.startswith('complete'): continue
        p = json.loads(path.read_text()); name = p['dataset']; assert p['fingerprints'] == fingerprints(name)
        prior = json.loads((ROOT/'results/trajectory-certificate-diagnostic'/f'{name}.json').read_text())
        X, y = load_dataset(name)
        for index, s in enumerate(p['splits']):
            ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
            old = prior['splits'][index]
            assert ri.tolist() == old['reference_indices'] and vi.tolist() == old['validation_indices']
            assert not set(ri).intersection(vi)
            raw, labels = X[ri], y[ri]
            for method, grid in GRIDS.items():
                saved = s['baselines'][method]; assert [c['config'] for c in saved['trace']] == grid
                for j, inner in enumerate(old['fit']['training_splits']):
                    ti, qi = np.array(inner['reference']), np.array(inner['query'])
                    assert set(ti).isdisjoint(qi) and set(ti).union(qi) == set(range(len(ri)))
                    A, B = scaling(raw[ti], raw[qi])
                    for candidate in saved['trace']:
                        selection = SELECTORS[method](A, labels[ti], **candidate['config'])
                        record = candidate['folds'][j]
                        assert selection['features'] == record['features']
                        metrics = score(A, labels[ti], B, labels[qi], selection['features'], 5000+j)
                        assert all(abs(metrics[k]-record['metrics'][k]) < 1e-12 for k in metrics)
                        counts['inner_refits'] += 1; counts['scores'] += 2
                for candidate in saved['trace']:
                    assert abs(candidate['utility']-np.mean([utility(f['metrics']) for f in candidate['folds']])) < 1e-12
                    assert abs(candidate['mean_size']-np.mean([len(f['features']) for f in candidate['folds']])) < 1e-12
                chosen = max(range(len(grid)), key=lambda j: (saved['trace'][j]['utility'], -saved['trace'][j]['mean_size'], -j))
                assert chosen == saved['chosen_index']
                A, B = scaling(raw, X[vi]); selection = SELECTORS[method](A, labels, **grid[chosen])
                assert json.loads(json.dumps(selection)) == saved['selection']
                metrics = score(A, labels, B, y[vi], selection['features'], 144+index)
                assert all(abs(metrics[k]-saved['metrics'][k]) < 1e-12 for k in metrics)
                counts['final_refits'] += 1; counts['scores'] += 2
            assert s['proposed_metrics'] == old['metrics']['certified']
            assert s['proposed_features'] == old['fit']['selected']['features']
            counts['splits'] += 1
        print(name, 'recent baseline audit complete', flush=True)
    result = {**counts, 'scope': 'independent_scaling_selector_refits_choice_score_checks_not_author_fidelity_validation'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/trajectory-recent-evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'): archive.write(path, str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'): archive.write(path, str(path.relative_to(ROOT)))
        for rel in ('scripts/compare_trajectory_recent.py', 'scripts/audit_trajectory_recent.py', 'research/trajectory-recent-comparison-protocol.md'):
            archive.write(ROOT/rel, rel)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
