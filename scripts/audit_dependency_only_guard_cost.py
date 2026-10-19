"""Dependency-only admission audit without relabeling majority certificates."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from audit_nested_geometry_forest import (ROOT,score,BA,check_model,equivalent,load_dataset,
    preprocess_fit,interaction_tree,weight_certificate,transfer_weights,capped_sparse_projection,paired)
from certified_frfs.majority_metric import MajorityTerms

SOURCE = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/dependency-only-guard-cost'
METHODS = ('old','dependency_only','full')


def audit(name):
    threadpool_limits(1); counts = Counter(); violations = []
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text()); old_path = SOURCE/(name+'.json'); source = json.loads(old_path.read_text())
    assert saved['complete'] and source['complete']
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    assert json.loads((SOURCE/'complete-audit.json').read_text())['passed']
    verified_source = json.loads((SOURCE/(name+'-audit.json')).read_text())
    assert verified_source['artifact_sha256'] == hashlib.sha256(old_path.read_bytes()).hexdigest()
    assert verified_source['auditor_sha256'] == hashlib.sha256((ROOT/'scripts/audit_nested_geometry_forest.py').read_bytes()).hexdigest()
    hashes = {'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == hashes: return previous['result']
    X,y = load_dataset(name); expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(saved['records']) == len(expected)
    for r,(i,j,s,inner) in zip(saved['records'],expected):
        assert r['outer_index'] == i and r['inner_index'] == j and r['outer_reference_indices'] == s['reference_indices']
        for k in ('reference_indices','validation_indices'): assert r[k] == inner[k]
        ri,a,b = (np.array(r[k]) for k in ('outer_reference_indices','reference_indices','validation_indices'))
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
        A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels = y[ri[a]]; d = A.shape[1]
        stage = r['stage']; path = stage['path']; old = inner['stages'][7]
        assert stage['C'] == stage['pair_penalty'] == 1. and not stage['majority_guard_enforced'] and not stage['teacher_guards_enforced']
        assert r['fits']['old'] == inner['models'][old['policies']['0.8']['model_id']]
        assert r['fits']['full'] == inner['models'][old['policies']['1.0']['model_id']]
        target_count = len(r['fits']['old']['features']); assert r['target_count'] == stage['target_count'] == target_count
        original = np.ones(d)/np.sqrt(d); protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10
        teacher = r['fits']['full']; classes = np.array(teacher['classes']); balance = np.array([1/(len(classes)*sum(labels == c)) for c in labels])
        center = balance @ A; np.testing.assert_allclose(path['center'],center,atol=1e-12); assert path['tie_decimals'] == 12
        target = score(A,teacher); target -= target.mean(axis=1,keepdims=True); probes = {}; checked = {}; source_geometry = {tuple(g['features']):g for g in inner['geometry']}
        def loss(fs):
            key = tuple(fs)
            if key not in probes:
                substituted = A.copy(); omitted = [a for a in range(d) if a not in fs]; substituted[:,omitted] = center[omitted]
                values = score(substituted,teacher); values -= values.mean(axis=1,keepdims=True)
                probes[key] = float(balance @ np.mean((values-target)**2,axis=1)); counts['raw_projection_probes'] += 1
            return probes[key]
        def candidate(record,fs,value):
            assert record['features'] == fs; np.testing.assert_allclose(record['projection_loss'],value,atol=1e-11,rtol=1e-9); key = tuple(fs)
            if key not in checked:
                if key in source_geometry: checked[key] = source_geometry[key]; counts['reused_verified_geometries'] += 1
                else:
                    weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
                    embedded = np.zeros(d); embedded[fs] = weights
                    checked[key] = {'features':fs,'weights':embedded.tolist(),'certificate':weight_certificate(A,labels,fs,original,weights,protected=protected)}
                    counts['new_geometry_replays'] += 1
            geometry = checked[key]; np.testing.assert_allclose(record['weights'],geometry['weights'],atol=1e-12); equivalent(record['certificate'],geometry['certificate'])
            admissible = geometry['certificate']['dependency_certificate']['upper_bound'] <= .1
            assert record['dependency_only_admissible'] == admissible
            # Preserve original combined and majority meanings even when this
            # diagnostic accepts that originally infeasible certificate.
            assert geometry['certificate']['feasible'] == (admissible and geometry['certificate']['majority_feasible'])
            return admissible
        active = list(range(d)); ai = ti = 0; accepted = False; assert candidate(path['trace'][0],active,0.)
        for ranking in path['rankings']:
            assert ranking['features'] == active and len(active) > target_count
            ordered = sorted([(loss([a for a in active if a != j]),j) for j in active],key=lambda v:(round(v[0],12),v[1]))
            assert [v['removed'] for v in ranking['choices']] == [j for _,j in ordered]
            np.testing.assert_allclose([v['loss'] for v in ranking['choices']],[value for value,_ in ordered],atol=1e-11,rtol=1e-9)
            accepted = False
            for value,removed in ordered:
                attempt = path['attempts'][ai]; ai += 1; fs = [a for a in active if a != removed]
                assert attempt['removed'] == removed
                admissible = candidate(attempt,fs,value); assert attempt['accepted'] == admissible
                counts['attempts'] += 1; counts['dependency_rejections'] += not admissible
                if admissible:
                    ti += 1; assert candidate(path['trace'][ti],fs,value); active = fs; accepted = True; break
            if not accepted: assert ranking is path['rankings'][-1]
        assert ai == len(path['attempts']) and ti+1 == len(path['trace']) and (len(active) == target_count or not accepted)
        equivalent(stage['candidate'],path['trace'][-1]); assert r['fits']['dependency_only'] == stage['fit'] and stage['fit']['features'] == active
        assert r['count_exact'] == (len(active) == target_count); counts['count_mismatches'] += not r['count_exact']
        counts['unchanged_supports'] += active == r['fits']['old']['features']; cert = stage['candidate']['certificate']
        assert cert['dependency_certificate']['upper_bound'] <= .1
        counts['deployed_majority_violations'] += not cert['majority_feasible']
        if not cert['majority_feasible']:
            violations.append({'outer_index':i,'inner_index':j,'required':cert['required'],'retained_counts':cert['retained_counts'],'features':active})
        rng = np.random.default_rng(144+i); assert r['query_seed'] == 144+i
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for m,fit in r['fits'].items():
            assert fit['C'] == fit['pair_penalty'] == 1.
            fs = fit['features']; assert fit['edges'] == [sorted([fs[u],fs[v]]) for u,v in interaction_tree(A[:,fs],labels)['edges']]
            check_model(A,labels,fit,Counter()); values = [BA(score(Q,fit),fit['classes'],y[ri[b]]) for Q in queries]
            equivalent(r['metrics'][m],{'clean':values[0],'noise':float(np.mean(values[1:])),'removed':1-len(fs)/d})
            counts['query_predictions'] += 4; counts['final_fit_caps'] += not fit['optimizer_converged']
        counts['records'] += 1; print(name,i,j,'dependency only audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'majority_violations':violations,
        'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean','noise','removed')} for m in METHODS}}
    checkpoint.write_text(json.dumps({'fingerprints':hashes,'result':result},indent=2),encoding='utf-8'); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text()); rows = []
    names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(audit,name) for name in names]): rows.append(f.result())
    if args.only: print(rows,flush=True); return
    lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]
    equivalent(summary['rows'],[{'dataset':r['dataset'],'methods':r['methods']} for r in rows]); counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in summary['counts'].items(): assert counts[k] == v
    assert counts['records'] == 128
    for metric in ('clean','noise'):
        for m in METHODS[1:]: equivalent(summary['contrasts'][metric][m],paired([r['methods'][m][metric]-r['methods']['old'][metric] for r in rows]))
    for m in METHODS:
        for metric in ('clean','noise','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
    gate = summary['contrasts']['clean']['dependency_only']['mean_pp'] >= .5 and counts['count_mismatches'] == counts['final_fit_caps'] == 0
    assert gate == summary['gate_passed']
    result = {'passed':True,'counts':dict(counts),'scope':'unchanged_original_certificate_meanings_dependency_only_admission_same_counts_fits_and_queries',
        'limits':'No_original_majority_retention_guarantee; auxiliary_guard_cost_development_not_SOTA_or_population_accuracy_claim',
        'majority_violations':{r['dataset']:r['majority_violations'] for r in rows},'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print({k:v for k,v in result.items() if k != 'majority_violations'},flush=True)


if __name__ == '__main__': main()
