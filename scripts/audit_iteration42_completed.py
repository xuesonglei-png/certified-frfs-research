"""Audit available42files against immutable cached41inputs and independent scores."""
import json, hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing, scoring, equal
from certified_frfs.datasets import load_dataset
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma


def audit(path):
    p = json.loads(path.read_text()); source = ROOT/'results/predictive-iteration41'/path.name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == p['source41_sha256']
    old = json.loads(source.read_text()); assert old['train_indices'] == p['train_indices'] and old['test_indices'] == p['test_indices']
    for rel, digest in p['source_hashes'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    X, y = load_dataset(p['dataset']); tr, te = np.array(p['train_indices']), np.array(p['test_indices'])
    assert hashlib.sha256((ROOT/'data/processed'/f"{p['dataset']}.npz").read_bytes()).hexdigest() == p['data_sha256']
    assert set(tr).isdisjoint(te) and set(tr) | set(te) == set(range(len(y)))
    trace = {}; inner_count = outer_count = sampled = 0
    for index, split in enumerate(p['inner_splits']):
        assert split['train'] == old['inner_splits'][index]['train'] and split['validation'] == old['inner_splits'][index]['validation']
        ii, jj = np.array(split['train']), np.array(split['validation']); A, B = preprocessing(X[tr][ii], X[tr][jj])
        for r in split['fits']:
            scores = scoring(A, y[tr][ii], B, y[tr][jj], r['features'], r['weights'], 144+index)
            equal(scores, r['validation']); inner_count += 1
            assert r['certificate']['feasible']
            trace.setdefault(json.dumps(r['config'], sort_keys=True), []).append({'score': (scores['clean']+scores['noise01'])/2, 'size': len(r['features'])})
    for c in p['configurations']:
        rows = trace[json.dumps(c['config'], sort_keys=True)]
        assert abs(c['score']-np.mean([r['score'] for r in rows])) < 1e-12
        assert abs(c['mean_size']-np.mean([r['size'] for r in rows])) < 1e-12
    chosen = min(p['configurations'], key=lambda c: (-c['score'], c['mean_size'], json.dumps(c['config'], sort_keys=True)))
    primary = next(r for r in p['rows'] if r['method'] == 'certified_anchored'); assert primary['chosen'] == chosen
    F, Q = preprocessing(X[tr], X[te]); checks = []
    for r in p['rows']:
        c = r['fit']; fs = c['features']; w = np.asarray(c['weights']); full = np.ones(F.shape[1])/np.sqrt(F.shape[1])
        np.testing.assert_allclose(c['full_weights'], full, atol=1e-12)
        assert abs(w.sum()-full.sum()) < 1e-10 and np.all(w >= 0)
        cert = weight_certificate(F, y[tr], fs, full, w[fs]); assert cert['feasible'] == c['certificate']['feasible']
        assert cert['retained_counts'] == c['certificate']['retained_counts']
        assert abs(cert['dependency_certificate']['upper_bound']-c['certificate']['dependency_certificate']['upper_bound']) < 1e-12
        if r['method'] == 'certified_anchored': assert cert['feasible']
        else:
            original = next(q for q in old['rows'] if q['method'] == r['method']); assert original['fit'] == c and original['chosen'] == r['chosen']
        equal(scoring(F, y[tr], Q, y[te], fs, w, 104), r['metrics']); outer_count += 1
        rng = np.random.default_rng(1008); losses = []
        for _ in range(10):
            Z = np.clip(F+rng.uniform(-.01, .01, F.shape), 0, 1)
            losses.append(gamma(Z, y[tr], weights=full)-gamma(Z, y[tr], fs, weights=w))
        assert max(losses) <= cert['dependency_certificate']['upper_bound']+1e-10; sampled += 10
        checks.append({'method': r['method'], 'bound': cert['dependency_certificate']['upper_bound'], 'sampled_losses': losses})
    assert p['baseline_rows'] == old['baseline_rows']
    for r in p['baseline_rows']:
        equal(scoring(F, y[tr], Q, y[te], r['features'], r['weights'], 104), r['metrics']); outer_count += 1
    uniform = p['uniform_on_selected_support']; assert uniform['features'] == primary['fit']['features']
    equal(scoring(F, y[tr], Q, y[te], uniform['features'], np.ones(F.shape[1])/np.sqrt(F.shape[1]), 104), uniform['metrics']); outer_count += 1
    return {'dataset': p['dataset'], 'fold': p['fold'], 'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'inner_scores_checked': inner_count, 'outer_scores_checked': outer_count, 'selected_metric_checks': checks,
        'sampled_loss_checks': sampled, 'observed_violations': 0,
        'limits': 'selected final certificates;inner feasibility read from saved fits;optimizer/repair not independently regenerated'}


def main():
    threadpool_limits(1); folder = ROOT/'results/predictive-iteration42'; target = folder/'audits'; target.mkdir(exist_ok=True); count = 0
    for path in sorted(folder.glob('*-s101-f*.json')):
        out = target/path.name
        if out.exists():
            assert json.loads(out.read_text())['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest(); continue
        result = audit(path); out.write_text(json.dumps(result, indent=2), encoding='utf-8'); count += 1; print(path.name, 'audited', flush=True)
    print('new audits', count, 'total', len(list(target.glob('*-s101-f*.json'))), flush=True)


if __name__ == '__main__': main()
