"""Incremental immutable-result audit; safe while generation continues."""
import json, hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing, scoring, equal
from certified_frfs.datasets import load_dataset
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma


def audit(path):
    p = json.loads(path.read_text()); X, y = load_dataset(p['dataset'])
    assert hashlib.sha256((ROOT/'data/processed'/f"{p['dataset']}.npz").read_bytes()).hexdigest() == p['data_sha256']
    tr, te = np.array(p['train_indices']), np.array(p['test_indices'])
    assert set(tr).isdisjoint(te) and set(tr) | set(te) == set(range(len(y)))
    source = ROOT/'results/predictive-iteration36'/path.name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == p['baseline_source_sha256']
    for rel, digest in p['source_hashes'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    trace = {}; inner_count = outer_count = 0
    for index, split in enumerate(p['inner_splits']):
        ii, jj = np.array(split['train']), np.array(split['validation'])
        assert set(ii).isdisjoint(jj) and set(ii) | set(jj) == set(range(len(tr)))
        A, B = preprocessing(X[tr][ii], X[tr][jj])
        for r in split['fits']:
            metrics = scoring(A, y[tr][ii], B, y[tr][jj], r['features'], r['weights'], 144+index)
            equal(metrics, r['validation']); inner_count += 1
            trace.setdefault(json.dumps(r['config'], sort_keys=True), []).append({
                'score': (metrics['clean']+metrics['noise01'])/2, 'size': len(r['features']), 'feasible': r['certificate']['feasible']})
    for c in p['configurations']:
        rows = trace[json.dumps(c['config'], sort_keys=True)]
        assert abs(c['score']-np.mean([r['score'] for r in rows])) < 1e-12
        assert abs(c['mean_size']-np.mean([r['size'] for r in rows])) < 1e-12
        assert c['all_inner_feasible'] == all(r['feasible'] for r in rows)
    ranking = lambda c: (-c['score'], c['mean_size'], json.dumps(c['config'], sort_keys=True))
    F, Q = preprocessing(X[tr], X[te]); checks = []
    for r in p['rows']:
        eligible = p['configurations']
        if r['method'] == 'certified_sparse': eligible = [c for c in eligible if c['all_inner_feasible']]
        elif r['method'] == 'unlimited_joint_metric': eligible = [c for c in eligible if c['config']['fraction'] == 1.]
        assert r['chosen'] == min(eligible, key=ranking)
        c = r['fit']; fs = c['features']; w = np.asarray(c['weights']); full = np.ones(F.shape[1])/np.sqrt(F.shape[1])
        np.testing.assert_allclose(c['full_weights'], full, atol=1e-12)
        assert abs(w.sum()-full.sum()) < 1e-10 and np.all(w >= 0)
        cert = weight_certificate(F, y[tr], fs, full, w[fs])
        assert cert['retained_counts'] == c['certificate']['retained_counts'] and cert['feasible'] == c['certificate']['feasible']
        assert abs(cert['dependency_certificate']['upper_bound']-c['certificate']['dependency_certificate']['upper_bound']) < 1e-12
        if r['method'] == 'certified_sparse': assert cert['feasible']
        if r['full_fallback']: assert c['config']['ridge'] == 'original'
        equal(scoring(F, y[tr], Q, y[te], fs, w, 104), r['metrics']); outer_count += 1
        rng = np.random.default_rng(1006); losses = []
        for _ in range(10):
            Z = np.clip(F+rng.uniform(-.01, .01, F.shape), 0, 1)
            losses.append(gamma(Z, y[tr], weights=full)-gamma(Z, y[tr], fs, weights=w))
        assert max(losses) <= cert['dependency_certificate']['upper_bound']+1e-10
        checks.append({'method': r['method'], 'bound': cert['dependency_certificate']['upper_bound'], 'sampled_losses': losses})
    for r in p['baseline_rows']:
        equal(scoring(F, y[tr], Q, y[te], r['features'], r['weights'], 104), r['metrics']); outer_count += 1
    return {'dataset': p['dataset'], 'fold': p['fold'], 'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'inner_scores_checked': inner_count, 'outer_scores_checked': outer_count, 'selected_metric_checks': checks,
        'sampled_loss_checks': 30, 'observed_violations': 0,
        'limits': 'selected certificates and scoring;inner certificate flags read from saved fits;no independent optimizer rebuild'}


def main():
    threadpool_limits(1); folder = ROOT/'results/predictive-iteration41'; target = folder/'audits'; target.mkdir(exist_ok=True)
    count = 0
    for path in sorted(folder.glob('*-s101-f*.json')):
        out = target/path.name
        if out.exists():
            assert json.loads(out.read_text())['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            continue
        result = audit(path); out.write_text(json.dumps(result, indent=2), encoding='utf-8'); count += 1
        print(path.name, 'audited', flush=True)
    print('new audits', count, 'total', len(list(target.glob('*-s101-f*.json'))), flush=True)


if __name__ == '__main__': main()
