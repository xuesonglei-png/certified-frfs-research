"""Verify learned reference identity, new weights/guards and query replay."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from audit_nested_geometry_forest import (ROOT, score, BA, check_model, equivalent, load_dataset,
    preprocess_fit, interaction_tree, weight_certificate, transfer_weights, capped_sparse_projection, paired)
from certified_frfs.majority_metric import MajorityTerms, learn_majority_weights

FOLDER = ROOT/'results/learned-reference-geometry-pilot'
ANCHOR = ROOT/'results/nested-geometry-forest'
METHODS = ('old', 'learned', 'matched', 'full')


def audit(name):
    threadpool_limits(1); counts = Counter(); protections = []
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
        original = learn_majority_weights(A, labels, eps=.01, reg=.01)
        stage = r['stage']; path = stage['path']; metric = stage['reference_metric']
        assert stage['C'] == stage['pair_penalty'] == 1. and not stage['teacher_guards_enforced']
        assert metric['eps'] == metric['reg'] == .01 and 'not_preserved_uniform_relation' in metric['scope']
        np.testing.assert_allclose(metric['weights'], original, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(original), 1., atol=1e-12); assert np.all(original > 0)
        mass = float(original.sum()); cap = max(float(original.max()), 2*mass/d)
        np.testing.assert_allclose(metric['mass'], mass, atol=1e-12); np.testing.assert_allclose(metric['cap'], cap, atol=1e-12)
        assert cap >= max(original) and int(np.ceil(d/2))*cap >= mass-1e-12
        protected = MajorityTerms(A, labels, weights=original).bounds(range(d), allowed_enemies=2) > 1e-10
        expected_counts = {str(c):int(sum(protected & (labels == c))) for c in np.unique(labels)}
        assert stage['original_majority_protected_counts'] == expected_counts
        old_stage = inner['stages'][7]
        assert r['old_original_majority_protected_counts'] == old_stage['original_majority_protected_counts']
        protections.append({'outer_index':i, 'inner_index':j, 'old':r['old_original_majority_protected_counts'], 'learned':expected_counts})
        teacher = stage['policies']['1.0']['fit']; assert teacher['features'] == list(range(d))
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
                weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A, fs, original)/mass, len(fs), cap/mass)*mass
                embedded = np.zeros(d); embedded[fs] = weights
                np.testing.assert_allclose(sum(weights), mass, atol=1e-10); assert max(weights) <= cap+1e-10
                geometries[key] = (embedded, weight_certificate(A, labels, fs, original, weights, protected=protected)); counts['geometry'] += 1
            embedded, cert = geometries[key]
            np.testing.assert_allclose(record['weights'], embedded, atol=1e-12); equivalent(record['certificate'], cert)
            return cert['feasible']
        active = list(range(d)); ti = ai = 0; accepted = False; assert candidate(path['trace'][0], active, 0.)
        for ranking in path['rankings']:
            assert ranking['features'] == active and len(active) > int(np.ceil(d/2))
            ordered = sorted([(loss([a for a in active if a != removed]), removed) for removed in active], key=lambda v:(round(v[0],12),v[1]))
            assert [v['removed'] for v in ranking['choices']] == [removed for _, removed in ordered]
            np.testing.assert_allclose([v['loss'] for v in ranking['choices']], [value for value,_ in ordered], atol=1e-11, rtol=1e-9)
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
        for f in (.5,.65,.8,1.):
            selected = next(v for v in reversed(path['trace']) if len(v['features']) >= int(np.ceil(f*d)))
            policy = stage['policies'][str(f)]; equivalent(policy['candidate'], selected)
            assert policy['fit']['features'] == selected['features']; check_model(A, labels, policy['fit'], counts)
        assert r['fits']['old'] == inner['models'][old_stage['policies']['0.8']['model_id']]
        assert r['fits']['full'] == inner['models'][old_stage['policies']['1.0']['model_id']]
        assert r['fits']['learned'] == stage['policies']['0.8']['fit']
        matched = next(v for v in reversed(path['trace']) if len(v['features']) >= len(r['fits']['old']['features']))
        assert r['fits']['matched']['features'] == matched['features']
        assert r['matched_count_exact'] == (len(matched['features']) == len(r['fits']['old']['features']))
        counts['matched_count_mismatches'] += not r['matched_count_exact']
        counts['unchanged_supports'] += r['fits']['old']['features'] == r['fits']['learned']['features']
        rng = np.random.default_rng(144+i); assert r['query_seed'] == 144+i
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for m, fit in r['fits'].items():
            assert fit['C'] == fit['pair_penalty'] == 1.
            fs = fit['features']; assert fit['edges'] == [sorted([fs[u],fs[v]]) for u,v in interaction_tree(A[:,fs],labels)['edges']]
            check_model(A, labels, fit, Counter())
            values = [BA(score(Qi,fit),fit['classes'],y[ri[b]]) for Qi in queries]
            equivalent(r['metrics'][m], {'clean':values[0],'noise':float(np.mean(values[1:])),'removed':1-len(fs)/d})
            counts['query_predictions'] += 4; counts['final_fit_caps'] += not fit['optimizer_converged']
        counts['records'] += 1; print(name,i,j,'learned reference audit passed',flush=True)
    return {'dataset':name,'counts':dict(counts),'protection_counts':protections,
        'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean','noise','removed')} for m in METHODS}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text())
    names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]; rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(audit,name) for name in names]): rows.append(f.result())
    if args.only: print(rows,flush=True); return
    lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]
    equivalent(summary['rows'],[{'dataset':r['dataset'],'methods':r['methods']} for r in rows])
    counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in summary['counts'].items(): assert counts[k] == v
    assert counts['records'] == 128
    for metric in ('clean','noise'):
        for m in METHODS[1:]: equivalent(summary['contrasts'][metric][m],paired([r['methods'][m][metric]-r['methods']['old'][metric] for r in rows]))
    for m in METHODS:
        for metric in ('clean','noise','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
    gate = summary['contrasts']['clean']['learned']['mean_pp'] >= .5 and summary['means']['learned']['removed'] >= summary['means']['old']['removed'] and counts['final_fit_caps'] == 0
    assert gate == summary['gate_passed']
    result = {'passed':True,'counts':dict(counts),'scope':'train_only_weight_replay_changed_relation_certificates_projection_refits_and_queries',
        'limits':'Preserves_learned_reference_relation_not_uniform_relation; weight_learning_replay_not_global_optimality_verification; exposed_development',
        'protection_counts':{r['dataset']:r['protection_counts'] for r in rows},'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print({k:v for k,v in result.items() if k != 'protection_counts'},flush=True)


if __name__ == '__main__': main()
