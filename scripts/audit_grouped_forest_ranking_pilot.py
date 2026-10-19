"""Independent objective/KKT and raw-substitution group ranking audit."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import cvxpy as cp
import numpy as np
from scipy.special import logsumexp, expit
from threadpoolctl import threadpool_limits
from audit_nested_geometry_forest import (ROOT, Q, membership, score, BA, check_model, equivalent, load_dataset,
    preprocess_fit, interaction_tree, weight_certificate, transfer_weights, capped_sparse_projection, paired)
from certified_frfs.majority_metric import MajorityTerms

FOLDER = ROOT/'results/grouped-forest-ranking-pilot'
ANCHOR = ROOT/'results/nested-geometry-forest'
METHODS = ('old', 'grouped', 'matched', 'full')


def group_objective_kkt(A, y, fit):
    classes = np.unique(y); q, d = len(classes), A.shape[1]; n = len(A)
    W = np.asarray(fit['coef']); V = np.asarray(fit['pair_coef']).reshape(q, len(fit['edges']), 3, 3)
    theta = [(W @ Q).reshape(q, -1)]
    theta += [(np.einsum('ki,qkl,lj->qij', Q, V[:, e], Q)*np.sqrt(fit['pair_penalty'])).reshape(q, 4) for e in range(len(fit['edges']))]
    beta = np.column_stack(theta); bias = np.asarray(fit['intercept'])
    if q == 2: beta, bias = beta[1:], bias[1:]
    else: beta = beta-beta.mean(axis=0); bias = bias-bias.mean()
    Z = membership(A) @ Q
    blocks = [Z.reshape(n, -1)]
    blocks += [(Z[:, u, :, None]*Z[:, v, None, :]).reshape(n, 4)/np.sqrt(fit['pair_penalty']) for u, v in fit['edges']]
    design = np.column_stack(blocks); logits = design @ beta.T+bias
    own = np.argmax(y[:, None] == classes[None], axis=1)
    weights = np.array([1/(q*sum(y == c)) for c in y])
    if q == 2:
        loss = float(weights @ (np.logaddexp(0, logits[:, 0])-own*logits[:, 0]))
        error = (expit(logits[:, 0])-own)[:, None]
        initial = (.5-own)[:, None]
    else:
        loss = float(weights @ (logsumexp(logits, axis=1)-logits[np.arange(n), own]))
        error = np.exp(logits-logsumexp(logits, axis=1)[:, None])-np.eye(q)[own]
        initial = 1/q-np.eye(q)[own]
    g0 = (weights[:, None]*initial).T @ design
    gradient = (weights[:, None]*error).T @ design+beta/(fit['C']*n)
    bias_gradient = weights @ error
    groups = []
    for a in range(d):
        cols = [2*a, 2*a+1]; factors = [1., 1.]
        for e, (u, v) in enumerate(fit['edges']):
            if a in (u, v): cols += list(range(2*d+4*e, 2*d+4*(e+1))); factors += [1/np.sqrt(2)]*4
        groups.append((cols, np.array(factors)))
    threshold = max(np.linalg.norm(g0[:, c]*f[None]) for c, f in groups)
    norms = [np.linalg.norm(beta[:, c]*f[None]) for c, f in groups]
    np.testing.assert_allclose(fit['node_group_norms'], norms, atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(fit['sufficient_zero_threshold'], threshold, atol=1e-12)
    np.testing.assert_allclose(fit['strength'], .1*threshold, atol=1e-12)
    assert fit['relative_strength'] == .1
    objective = loss+np.sum(beta**2)/(2*fit['C']*n)+fit['strength']*sum(norms)
    np.testing.assert_allclose(fit['objective'], objective, atol=1e-8, rtol=1e-8)
    # Numerical stationarity: for near-zero groups, seek admissible SOC
    # subgradients rather than divide by their tiny norms.
    unknown = []; constraints = []; residual = gradient.copy(); strength = fit['strength']
    for (cols, factors), norm in zip(groups, norms):
        if norm > 1e-6:
            residual[:, cols] += strength*beta[:, cols]*factors[None]**2/norm
        else:
            dual = cp.Variable((len(beta), len(cols))); constraints.append(cp.norm(dual, 'fro') <= 1)
            embed = np.zeros((len(cols), design.shape[1])); embed[np.arange(len(cols)), cols] = factors
            unknown.append(strength*dual @ embed)
    if unknown:
        residual_expression = residual+sum(unknown)
        problem = cp.Problem(cp.Minimize(cp.norm(residual_expression, 'fro')), constraints)
        problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
        assert problem.status == 'optimal'
        residual = residual_expression.value
    error_norm = float(np.linalg.norm(residual)); assert error_norm < 1e-4, error_norm
    assert np.max(abs(bias_gradient)) < 1e-5
    return error_norm


def audit(name):
    threadpool_limits(1); counts = Counter(); residuals = []
    saved = json.loads((FOLDER/(name+'.json')).read_text()); anchor = json.loads((ANCHOR/(name+'.json')).read_text())
    assert saved['complete']
    for rel, value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    X, y = load_dataset(name); expected = [(i, j, s, inner) for i, s in enumerate(anchor['splits']) for j, inner in enumerate(s['inner'])]
    assert len(expected) == len(saved['records'])
    for r, (i, j, s, inner) in zip(saved['records'], expected):
        assert r['outer_index'] == i and r['inner_index'] == j and r['outer_reference_indices'] == s['reference_indices']
        for key in ('reference_indices', 'validation_indices'): assert r[key] == inner[key]
        ri, a, b = (np.array(r[k]) for k in ('outer_reference_indices', 'reference_indices', 'validation_indices'))
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
        A, B = preprocess_fit(X[ri[a]], X[ri[b]]); labels = y[ri[a]]; d = A.shape[1]
        original = np.ones(d)/np.sqrt(d); protected = MajorityTerms(A, labels, weights=original).bounds(range(d), allowed_enemies=2) > 1e-10
        stage = r['stage']; path = stage['path']; assert stage['C'] == stage['pair_penalty'] == 1. and not stage['teacher_guards_enforced']
        teacher = stage['group_teacher']; assert teacher['usable'] == (not stage['group_selector_fallback'])
        if teacher['usable']:
            assert teacher['status'] == 'optimal' and teacher['features'] == list(range(d)) and teacher['C'] == teacher['pair_penalty'] == 1.
            assert teacher['edges'] == interaction_tree(A, labels)['edges']
            residuals.append(group_objective_kkt(A, labels, teacher)); counts['group_objective_kkt_checks'] += 1
        else:
            teacher = r['fits']['full']; counts['selector_fallbacks'] += 1
        classes = np.asarray(teacher['classes']); balance = np.array([1/(len(classes)*sum(labels == c)) for c in labels])
        center = balance @ A; np.testing.assert_allclose(path['center'], center, atol=1e-12)
        target = score(A, teacher); target -= target.mean(axis=1, keepdims=True); probes = {}; geometries = {}
        def loss(fs):
            key = tuple(fs)
            if key not in probes:
                replaced = A.copy(); omitted = [a for a in range(d) if a not in fs]; replaced[:, omitted] = center[omitted]
                predicted = score(replaced, teacher); predicted -= predicted.mean(axis=1, keepdims=True)
                probes[key] = float(balance @ np.mean((predicted-target)**2, axis=1)); counts['raw_projection_probes'] += 1
            return probes[key]
        def candidate(record, fs, value):
            assert record['features'] == fs; np.testing.assert_allclose(record['projection_loss'], value, atol=1e-11, rtol=1e-9)
            key = tuple(fs)
            if key not in geometries:
                weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A, fs, original)/np.sqrt(d), len(fs), 2/d)*np.sqrt(d)
                embedded = np.zeros(d); embedded[fs] = weights
                geometries[key] = (embedded, weight_certificate(A, labels, fs, original, weights, protected=protected)); counts['geometry'] += 1
            embedded, cert = geometries[key]
            np.testing.assert_allclose(record['weights'], embedded, atol=1e-12); equivalent(record['certificate'], cert)
            return cert['feasible']
        active = list(range(d)); ti = ai = 0; accepted = False; assert candidate(path['trace'][0], active, 0.)
        for ranking in path['rankings']:
            assert ranking['features'] == active and len(active) > int(np.ceil(d/2))
            ordered = sorted([(loss([a for a in active if a != removed]), removed) for removed in active], key=lambda v: (round(v[0], 12), v[1]))
            assert [v['removed'] for v in ranking['choices']] == [removed for _, removed in ordered]
            np.testing.assert_allclose([v['loss'] for v in ranking['choices']], [value for value, _ in ordered], atol=1e-11, rtol=1e-9)
            accepted = False
            for value, removed in ordered:
                attempted = path['attempts'][ai]; ai += 1; fs = [a for a in active if a != removed]
                assert attempted['removed'] == removed
                feasible = candidate(attempted, fs, value); assert attempted['accepted'] == feasible
                counts['attempts'] += 1; counts['geometry_rejections'] += not feasible
                if feasible:
                    ti += 1; assert candidate(path['trace'][ti], fs, value); active = fs; accepted = True; break
            if not accepted: assert ranking is path['rankings'][-1]
        assert ai == len(path['attempts']) and ti+1 == len(path['trace'])
        assert len(active) == int(np.ceil(d/2)) or not accepted
        for f in (.5, .65, .8, 1.):
            selected = next(v for v in reversed(path['trace']) if len(v['features']) >= int(np.ceil(f*d)))
            policy = stage['policies'][str(f)]; equivalent(policy['candidate'], selected)
            assert policy['fit']['features'] == selected['features']; check_model(A, labels, policy['fit'], counts)
        old = inner['stages'][7]
        assert r['fits']['old'] == inner['models'][old['policies']['0.8']['model_id']]
        assert r['fits']['full'] == inner['models'][old['policies']['1.0']['model_id']]
        assert r['fits']['grouped'] == stage['policies']['0.8']['fit']
        matched = next(v for v in reversed(path['trace']) if len(v['features']) >= len(r['fits']['old']['features']))
        assert r['fits']['matched']['features'] == matched['features']
        assert r['matched_count_exact'] == (len(matched['features']) == len(r['fits']['old']['features']))
        counts['matched_count_mismatches'] += not r['matched_count_exact']
        rng = np.random.default_rng(144+i); assert r['query_seed'] == 144+i
        queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
        for m, fit in r['fits'].items():
            assert fit['C'] == fit['pair_penalty'] == 1.
            fs = fit['features']; assert fit['edges'] == [sorted([fs[u], fs[v]]) for u, v in interaction_tree(A[:, fs], labels)['edges']]
            check_model(A, labels, fit, Counter())
            values = [BA(score(Qi, fit), fit['classes'], y[ri[b]]) for Qi in queries]
            equivalent(r['metrics'][m], {'clean': values[0], 'noise': float(np.mean(values[1:])), 'removed': 1-len(fs)/d})
            counts['query_predictions'] += 4; counts['final_fit_caps'] += not fit['optimizer_converged']
        counts['records'] += 1; print(name, i, j, 'group audit passed', flush=True)
    return {'dataset': name, 'counts': dict(counts), 'max_kkt_residual': max(residuals, default=0.),
        'methods': {m: {metric: float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean', 'noise', 'removed')} for m in METHODS}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text())
    names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]; rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(audit, name) for name in names]): rows.append(f.result())
    if args.only: print(rows, flush=True); return
    lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]
    equivalent(summary['rows'], [{'dataset':r['dataset'], 'methods':r['methods']} for r in rows])
    counts = sum((Counter(r['counts']) for r in rows), Counter())
    for k, v in summary['counts'].items(): assert counts[k] == v
    assert counts['records'] == 128
    for metric in ('clean', 'noise'):
        for m in METHODS[1:]: equivalent(summary['contrasts'][metric][m], paired([r['methods'][m][metric]-r['methods']['old'][metric] for r in rows]))
    for m in METHODS:
        for metric in ('clean', 'noise', 'removed'): equivalent(summary['means'][m][metric], float(np.mean([r['methods'][m][metric] for r in rows])))
    gate = summary['contrasts']['clean']['grouped']['mean_pp'] >= .5 and summary['means']['grouped']['removed'] >= summary['means']['old']['removed'] and counts['selector_fallbacks'] == counts['final_fit_caps'] == 0
    assert gate == summary['gate_passed']
    result = {'passed': True, 'counts':dict(counts), 'max_kkt_residual':max(r['max_kkt_residual'] for r in rows),
        'scope':'independent_objective_numerical_KKT_raw_projection_geometry_fits_and_predictions',
        'limits':'Numerical_KKT_check_not_certified_optimum; exposed_fixed_config_development_not_nested_SOTA_evidence',
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(result, flush=True)


if __name__ == '__main__': main()
