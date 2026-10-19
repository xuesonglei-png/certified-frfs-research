"""Independent split/scoring/configuration/certificate audit of completed pilots."""
import json, hashlib, argparse
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma


def preprocessing(A, B):
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    sc = MinMaxScaler(clip=True); F = sc.fit_transform(imp.fit_transform(A))
    return F, sc.transform(imp.transform(B))


def scoring(A, y, B, truth, fs, weights, seed):
    fs = np.asarray(fs); w = np.asarray(weights)[fs]
    model = KNeighborsClassifier(n_neighbors=min(5, len(y)), metric='manhattan', algorithm='brute')
    model.fit(A[:, fs]*w, y)
    clean = float(balanced_accuracy_score(truth, model.predict(B[:, fs]*w)))
    rng = np.random.default_rng(seed); noisy = []
    for _ in range(3):
        ref = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
        query = np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1)
        model.fit(ref[:, fs]*w, y)
        noisy.append(balanced_accuracy_score(truth, model.predict(query[:, fs]*w)))
    return {'clean': clean, 'noise01': float(np.mean(noisy))}


def equal(a, b):
    assert all(abs(a[k]-b[k]) < 1e-12 for k in ('clean', 'noise01')), (a, b)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory'); args = parser.parse_args()
    threadpool_limits(1); folder = ROOT/'results'/args.directory
    expected = ['iris', 'wine', 'sonar', 'wdbc', 'ionosphere']; rows = []; outer_count = inner_count = sampled = 0
    for name in expected:
        path = folder/f'{name}-s101-f0.json'; p = json.loads(path.read_text())
        X, y = load_dataset(name); tr = np.array(p['train_indices']); te = np.array(p['test_indices'])
        assert set(tr).isdisjoint(te) and set(tr) | set(te) == set(range(len(y)))
        source = ROOT/'results/predictive-iteration36'/path.name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == p['baseline_source_sha256']
        A, B = preprocessing(X[tr], X[te]); rebuilt = {}
        for index, split in enumerate(p['inner_splits']):
            ii = np.array(split['train']); jj = np.array(split['validation'])
            assert set(ii).isdisjoint(jj) and set(ii) | set(jj) == set(range(len(tr)))
            F, Q = preprocessing(X[tr][ii], X[tr][jj])
            for r in split['fits']:
                metrics = scoring(F, y[tr][ii], Q, y[tr][jj], r['features'], r['weights'], 144+index)
                equal(metrics, r['validation']); inner_count += 1
                key = json.dumps(r['config'], sort_keys=True)
                rebuilt.setdefault(key, []).append({'score': (metrics['clean']+metrics['noise01'])/2,
                    'size': len(r['features']), 'feasible': r['certificate']['feasible']})
        for c in p['configurations']:
            records = rebuilt[json.dumps(c['config'], sort_keys=True)]
            assert abs(c['score']-np.mean([r['score'] for r in records])) < 1e-12
            assert abs(c['mean_size']-np.mean([r['size'] for r in records])) < 1e-12
            assert c['all_inner_feasible'] == all(r['feasible'] for r in records)
        if args.directory == 'sparse-neighbor-pilot':
            key = lambda c: (-c['score'], c['mean_size'], c['config']['eps'], c['config']['fraction'])
        else: key = lambda c: (-c['score'], c['mean_size'], json.dumps(c['config'], sort_keys=True))
        metrics_by_method = {}; sizes = {}
        for r in p['rows']:
            eligible = p['configurations']
            if r['method'] == 'certified_sparse': eligible = [c for c in eligible if c['all_inner_feasible']]
            elif r['method'] in ('dense_joint_metric', 'unlimited_joint_metric'): eligible = [c for c in eligible if c['config']['fraction'] == 1.]
            assert r['chosen'] == min(eligible, key=key)
            fit = r['fit']; fs = fit['features']; w = np.asarray(fit['weights']); full = np.asarray(fit['full_weights'])
            assert abs(w.sum()-full.sum()) < 1e-10 and np.all(w >= 0)
            assert np.array_equal(np.flatnonzero(w > 0), fs)
            cert = weight_certificate(A, y[tr], fs, full, w[fs])
            assert cert['feasible'] == fit['certificate']['feasible']
            assert cert['retained_counts'] == fit['certificate']['retained_counts']
            assert abs(cert['dependency_certificate']['upper_bound']-fit['certificate']['dependency_certificate']['upper_bound']) < 1e-12
            if args.directory == 'regularized-sparse-pilot': np.testing.assert_allclose(full, np.ones(A.shape[1])/np.sqrt(A.shape[1]), atol=1e-12)
            if r['method'] == 'certified_sparse': assert cert['feasible']
            rng = np.random.default_rng(990); losses = []
            for _ in range(10):
                Z = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1)
                losses.append(gamma(Z, y[tr], weights=full)-gamma(Z, y[tr], fs, weights=w))
            assert max(losses) <= cert['dependency_certificate']['upper_bound']+1e-10; sampled += len(losses)
            metrics = scoring(A, y[tr], B, y[te], fs, w, 104); equal(metrics, r['metrics']); outer_count += 1
            metrics_by_method[r['method']] = metrics; sizes[r['method']] = len(fs)
        for r in p['baseline_rows']:
            metrics = scoring(A, y[tr], B, y[te], r['features'], r['weights'], 104)
            equal(metrics, r['metrics']); outer_count += 1; metrics_by_method[r['method']] = metrics
        rows.append({'dataset': name, 'metrics': metrics_by_method, 'sizes': sizes, 'd': A.shape[1]})
        print(name, 'audited', flush=True)
    methods = list(rows[0]['metrics']); means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in ('clean', 'noise01')} for m in methods}
    gain = means['certified_sparse']['clean']-means['fgs']['clean']
    gates = {'clean_gain_over_fgs_at_least_half_point': gain >= .005,
        'clean_at_least_all': means['certified_sparse']['clean'] >= means['all']['clean'],
        'at_least_three_dataset_wins_over_fgs': sum(r['metrics']['certified_sparse']['clean'] > r['metrics']['fgs']['clean']+1e-12 for r in rows) >= 3}
    out = folder/'verified-analysis.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps({'rows': rows, 'means': means, 'gates': gates, 'gate_passed': all(gates.values()),
        'outer_scores_checked': outer_count, 'inner_scores_checked': inner_count,
        'sampled_loss_checks': sampled, 'observed_violations': 0,
        'scope': 'five_reused_development_tasks_descriptive_not_significance_or_confirmation',
        'audit_limits': 'selected final certificates checked;inner certificate eligibility read from saved fits;optimizers not independently refitted'}, indent=2), encoding='utf-8')
    print(json.dumps(means), json.dumps(gates), flush=True)


if __name__ == '__main__': main()
