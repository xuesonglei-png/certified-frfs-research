"""Replay joint phases, certify new masks, reuse audited unchanged old masks."""
import argparse
import hashlib
import json
from pathlib import Path
from itertools import combinations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from audit_nested_geometry_forest import (ROOT,score,BA,check_model,equivalent,load_dataset,
    preprocess_fit,interaction_tree,weight_certificate,transfer_weights,capped_sparse_projection,paired)
from certified_frfs.majority_metric import MajorityTerms

FOLDER = ROOT/'results/joint-deletion-geometry-pilot'
ANCHOR = ROOT/'results/nested-geometry-forest'
METHODS = ('old','joint','matched','full')


def audit(name):
    threadpool_limits(1); counts = Counter()
    artifact = ANCHOR/(name+'.json'); anchor = json.loads(artifact.read_text())
    saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
    prerequisite = json.loads((ANCHOR/'complete-audit.json').read_text()); assert prerequisite['passed']
    checkpoint = json.loads((ANCHOR/(name+'-audit.json')).read_text())
    assert checkpoint['artifact_sha256'] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert checkpoint['auditor_sha256'] == hashlib.sha256((ROOT/'scripts/audit_nested_geometry_forest.py').read_bytes()).hexdigest()
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    X,y = load_dataset(name); expected = [(i,j,s,inner) for i,s in enumerate(anchor['splits']) for j,inner in enumerate(s['inner'])]
    assert len(saved['records']) == len(expected)
    for r,(i,j,s,inner) in zip(saved['records'],expected):
        assert r['outer_index'] == i and r['inner_index'] == j and r['outer_reference_indices'] == s['reference_indices']
        for k in ('reference_indices','validation_indices'): assert r[k] == inner[k]
        ri,a,b = (np.array(r[k]) for k in ('outer_reference_indices','reference_indices','validation_indices'))
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
        A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels = y[ri[a]]; d = A.shape[1]
        original = np.ones(d)/np.sqrt(d); protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10
        stage = r['stage']; path = stage['path']; old = inner['stages'][7]
        assert stage['C'] == stage['pair_penalty'] == 1. and not stage['teacher_guards_enforced']
        assert path['pair_pool_size'] == 8 and path['tie_decimals'] == 12
        assert stage['original_majority_protected_counts'] == old['original_majority_protected_counts']
        assert r['fits']['old'] == inner['models'][old['policies']['0.5']['model_id']]
        assert r['fits']['full'] == inner['models'][old['policies']['1.0']['model_id']]
        teacher = stage['policies']['1.0']['fit']; assert teacher == r['fits']['full']
        classes = np.asarray(teacher['classes']); balance = np.array([1/(len(classes)*sum(labels == c)) for c in labels])
        center = balance @ A; np.testing.assert_allclose(path['center'],center,atol=1e-12)
        target = score(A,teacher); target -= target.mean(axis=1,keepdims=True)
        losses = {}; verified = {tuple(g['features']):g for g in inner['geometry']}
        checked = {}
        def loss(fs):
            key = tuple(fs)
            if key not in losses:
                replaced = A.copy(); omitted = [a for a in range(d) if a not in fs]; replaced[:,omitted] = center[omitted]
                scores = score(replaced,teacher); scores -= scores.mean(axis=1,keepdims=True)
                losses[key] = float(balance @ np.mean((scores-target)**2,axis=1)); counts['raw_projection_probes'] += 1
            return losses[key]
        def candidate(record,fs,value):
            assert record['features'] == fs; np.testing.assert_allclose(record['projection_loss'],value,atol=1e-11,rtol=1e-9)
            key = tuple(fs)
            if key not in checked:
                if key in verified:
                    checked[key] = verified[key]; counts['reused_previously_independently_verified_geometries'] += 1
                else:
                    weights = capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
                    embedded = np.zeros(d); embedded[fs] = weights
                    checked[key] = {'features':fs,'weights':embedded.tolist(),'certificate':weight_certificate(A,labels,fs,original,weights,protected=protected)}
                    counts['new_independently_recomputed_geometries'] += 1
            geometry = checked[key]; np.testing.assert_allclose(record['weights'],geometry['weights'],atol=1e-12)
            equivalent(record['certificate'],geometry['certificate'])
            return geometry['certificate']['feasible']
        active = list(range(d)); minimum = int(np.ceil(d/2)); ai = ti = gi = jumps = 0
        assert candidate(path['trace'][0],active,0.)
        while len(active) > minimum:
            singles = sorted([(loss([a for a in active if a != z]),(z,)) for z in active],key=lambda v:(round(v[0],12),v[1]))
            phases = [(1,singles)]; accepted = None
            for size,ordered in phases:
                ranking = path['rankings'][gi]; gi += 1
                assert ranking['features'] == active and ranking['deletion_size'] == size
                assert [v['removed'] for v in ranking['choices']] == [list(removed) for _,removed in ordered]
                np.testing.assert_allclose([v['loss'] for v in ranking['choices']],[value for value,_ in ordered],atol=1e-11,rtol=1e-9)
                for value,removed in ordered:
                    attempt = path['attempts'][ai]; ai += 1; fs = [a for a in active if a not in removed]
                    assert attempt['removed'] == list(removed)
                    feasible = candidate(attempt,fs,value); assert attempt['accepted'] == feasible
                    counts['attempts'] += 1; counts['pair_attempts'] += size == 2; counts['geometry_rejections'] += not feasible
                    if feasible:
                        ti += 1; assert candidate(path['trace'][ti],fs,value)
                        accepted = fs; jumps += size == 2; break
                if accepted is not None: break
                if size == 1 and len(active)-2 >= minimum:
                    pool = sorted(removed[0] for _,removed in singles[:8])
                    ordered_pairs = sorted([(loss([a for a in active if a not in removed]),removed) for removed in combinations(pool,2)],key=lambda v:(round(v[0],12),v[1]))
                    phases.append((2,ordered_pairs))
            if accepted is None: break
            active = accepted
        assert ai == len(path['attempts']) and gi == len(path['rankings']) and ti+1 == len(path['trace'])
        assert jumps == path['accepted_pair_jumps']; counts['accepted_pair_jumps'] += jumps; counts['records_with_jumps'] += jumps > 0
        # Joint search extends the old path exactly; it changes no earlier
        # certified single deletion. Same-old-count control is therefore old.
        old_trace = [inner['geometry'][v['geometry_id']]['features'] for v in old['path']['trace']]
        assert [v['features'] for v in path['trace'][:len(old_trace)]] == old_trace
        for f in (.5,.65,.8,1.):
            selected = next(v for v in reversed(path['trace']) if len(v['features']) >= int(np.ceil(f*d)))
            policy = stage['policies'][str(f)]; equivalent(policy['candidate'],selected)
            assert policy['fit']['features'] == selected['features']; check_model(A,labels,policy['fit'],counts)
        assert r['fits']['joint'] == stage['policies']['0.5']['fit']
        matched = next(v for v in reversed(path['trace']) if len(v['features']) >= len(r['fits']['old']['features']))
        assert r['fits']['matched']['features'] == matched['features'] == r['fits']['old']['features']
        assert r['matched_count_exact'] is True
        rng = np.random.default_rng(144+i); assert r['query_seed'] == 144+i
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for m,fit in r['fits'].items():
            assert fit['C'] == fit['pair_penalty'] == 1.
            fs = fit['features']; assert fit['edges'] == [sorted([fs[u],fs[v]]) for u,v in interaction_tree(A[:,fs],labels)['edges']]
            check_model(A,labels,fit,Counter()); values = [BA(score(Q,fit),fit['classes'],y[ri[b]]) for Q in queries]
            equivalent(r['metrics'][m],{'clean':values[0],'noise':float(np.mean(values[1:])),'removed':1-len(fs)/d})
            counts['query_predictions'] += 4; counts['final_fit_caps'] += not fit['optimizer_converged']
        counts['records'] += 1; print(name,i,j,'joint audit passed',flush=True)
    return {'dataset':name,'counts':dict(counts),'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean','noise','removed')} for m in METHODS}}


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
    gate = summary['contrasts']['clean']['joint']['mean_pp'] >= .5 and summary['means']['joint']['removed'] >= summary['means']['old']['removed'] and counts['final_fit_caps'] == 0
    assert gate == summary['gate_passed']
    result = {'passed':True,'counts':dict(counts),'scope':'independent_phases_raw_projection_new_geometry_fits_queries_audited_source_geometry_reuse',
        'limits':'Old_geometry_reused_only_with_matching_independent_audit_artifact_hash; bounded_pair_pool_not_complete; exposed_development',
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__ == '__main__': main()
