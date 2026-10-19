"""Independent raw-score sensitivity, deletion, geometry, and refit audit."""
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
FOLDER = ROOT/'results/task-reference-geometry-pilot'
METHODS = ('old','full','uniform_native','task_native','task_matched')


def raw_probe_sensitivity(fit):
    """Recover coordinate slope extrema from direct score evaluations.

    Fix other neighbors at .5, then independently vary each incident neighbor
    over 0,.5,1. Add the minimum/maximum *changes* to the baseline contrast
    slope. No coefficient-derivative implementation is imported.
    """
    d = len(fit['coef'][0]); q = len(fit['classes'])
    pairs = [(c,r) for c in range(q) for r in range(c+1,q)]
    lower = np.zeros((len(pairs),d,2)); upper = lower.copy()
    for a in range(d):
        neighbors = sorted({v if u == a else u for u,v in fit['edges'] if a in (u,v)})
        for branch in range(2):
            points = np.full((2+6*len(neighbors),d),.5)
            points[:,a] = np.tile([branch/2,(branch+1)/2],len(points)//2)
            for k,neighbor in enumerate(neighbors):
                points[2+6*k:2+6*(k+1),neighbor] = np.repeat([0.,.5,1.],2)
            values = score(points,fit); slopes = 2*(values[1::2]-values[::2])
            contrasts = np.column_stack([slopes[:,c]-slopes[:,r] for c,r in pairs])
            base = contrasts[0]; lo,hi = base.copy(),base.copy()
            for k in range(len(neighbors)):
                changes = contrasts[1+3*k:1+3*(k+1)]-base
                lo += changes.min(axis=0); hi += changes.max(axis=0)
            lower[:,a,branch],upper[:,a,branch] = lo,hi
    per_contrast = np.maximum(abs(lower),abs(upper)).max(axis=2)
    constants = per_contrast.max(axis=0); total = float(constants.sum())
    return {'coordinate_constants':constants.tolist(),'contrast_constants':per_contrast.tolist(),
        'class_index_pairs':[list(p) for p in pairs],'slope_lower':lower.tolist(),'slope_upper':upper.tolist(),
        'total_constant':total,'unit_mass_weights':(constants/total if total > 0 else np.ones(d)/d).tolist(),
        'uniform_fallback':total == 0}


def audit(name):
    threadpool_limits(1); counts = Counter(); artifact = FOLDER/(name+'.json')
    saved = json.loads(artifact.read_text()); old_path = SOURCE/(name+'.json'); source = json.loads(old_path.read_text())
    assert saved['complete'] and source['complete']
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    assert json.loads((SOURCE/'complete-audit.json').read_text())['passed']
    per = json.loads((SOURCE/(name+'-audit.json')).read_text())
    assert per['artifact_sha256'] == hashlib.sha256(old_path.read_bytes()).hexdigest()
    assert per['auditor_sha256'] == hashlib.sha256((ROOT/'scripts/audit_nested_geometry_forest.py').read_bytes()).hexdigest()
    fingerprints = {'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == fingerprints: return previous['result']
    X,y = load_dataset(name)
    expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(expected) == len(saved['records'])
    for record,(i,j,s,inner) in zip(saved['records'],expected):
        assert record['outer_index'] == i and record['inner_index'] == j
        assert record['outer_reference_indices'] == s['reference_indices']
        for key in ('reference_indices','validation_indices'): assert record[key] == inner[key]
        ri,a,b = (np.array(record[k]) for k in ('outer_reference_indices','reference_indices','validation_indices'))
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
        A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels = y[ri[a]]; d = A.shape[1]
        oldstage = inner['stages'][7]; fits = record['fits']; teacher = fits['full']
        assert fits['old'] == inner['models'][oldstage['policies']['0.8']['model_id']]
        assert teacher == inner['models'][oldstage['policies']['1.0']['model_id']]
        exact_sensitivity = raw_probe_sensitivity(teacher); counts['raw_sensitivity_replays'] += 1
        classes = np.unique(labels); balance = np.array([1/(len(classes)*sum(labels == c)) for c in labels])
        center = balance @ A; teacher_scores = score(A,teacher); teacher_scores -= teacher_scores.mean(axis=1,keepdims=True)
        probes = {}
        def loss(fs):
            key = tuple(fs)
            if key not in probes:
                Q = A.copy(); omitted = [a for a in range(d) if a not in fs]; Q[:,omitted] = center[omitted]
                values = score(Q,teacher); values -= values.mean(axis=1,keepdims=True)
                probes[key] = float(balance @ np.mean((values-teacher_scores)**2,axis=1))
                counts['raw_projection_probes'] += 1
            return probes[key]
        for family,stage in record['stages'].items():
            assert family in ('task_unit','uniform_unit') and stage['family'] == family
            assert stage['reference_teacher'] == teacher and stage['C'] == stage['pair_penalty'] == 1.
            assert stage['fraction'] == .8 and stage['target_count'] == int(np.ceil(.8*d))
            for key,value in exact_sensitivity.items(): equivalent(stage['sensitivity'][key],value)
            w = np.array(exact_sensitivity['unit_mass_weights']) if family == 'task_unit' else np.ones(d)/d
            np.testing.assert_allclose(stage['reference_weights'],w,atol=1e-12)
            cap = max(float(w.max()),2/d); np.testing.assert_allclose(stage['cap'],cap,atol=1e-12)
            protected = MajorityTerms(A,labels,weights=w).bounds(range(d),allowed_enemies=2) > 1e-10
            assert stage['protected_mask'] == protected.tolist()
            path = stage['path']; np.testing.assert_allclose(path['center'],center,atol=1e-12)
            assert path['tie_decimals'] == 12; geometries = {}
            def verify(point,fs,value):
                assert point['features'] == fs
                np.testing.assert_allclose(point['projection_loss'],value,atol=1e-11,rtol=1e-9)
                key = tuple(fs)
                if key not in geometries:
                    local = w if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,w),len(fs),cap)
                    embedded = np.zeros(d); embedded[fs] = local
                    assert abs(sum(local)-1) < 1e-10 and max(local) <= cap+1e-10
                    geometries[key] = {'weights':embedded.tolist(),
                        'certificate':weight_certificate(A,labels,fs,w,local,protected=protected)}
                    counts['new_geometry_replays'] += 1
                g = geometries[key]; equivalent(point['weights'],g['weights']); equivalent(point['certificate'],g['certificate'])
                return g['certificate']['feasible']
            active = list(range(d)); ai = ti = 0; accepted = False
            assert verify(path['trace'][0],active,0.)
            for ranking in path['rankings']:
                assert ranking['features'] == active and len(active) > stage['target_count']
                ordered = sorted([(loss([a for a in active if a != removed]),removed) for removed in active],key=lambda v:(round(v[0],12),v[1]))
                assert [r['removed'] for r in ranking['choices']] == [a for _,a in ordered]
                np.testing.assert_allclose([r['loss'] for r in ranking['choices']],[v for v,_ in ordered],atol=1e-11,rtol=1e-9)
                accepted = False
                for value,removed in ordered:
                    point = path['attempts'][ai]; ai += 1; fs = [a for a in active if a != removed]
                    assert point['removed'] == removed; feasible = verify(point,fs,value)
                    assert point['accepted'] == feasible; counts['attempts'] += 1; counts['geometry_rejections'] += not feasible
                    if feasible:
                        ti += 1; assert verify(path['trace'][ti],fs,value); active = fs; accepted = True; break
                if not accepted: assert ranking is path['rankings'][-1]
            assert ai == len(path['attempts']) and ti+1 == len(path['trace'])
            assert len(active) == stage['target_count'] or not accepted
            equivalent(stage['candidate'],path['trace'][-1]); assert stage['fit']['features'] == active
            key = 'task_native' if family == 'task_unit' else 'uniform_native'
            assert fits[key] == stage['fit'] and stage['candidate']['certificate']['feasible']
        target = len(fits['old']['features']); assert record['target_count'] == target
        point = next(p for p in reversed(record['stages']['task_unit']['path']['trace']) if len(p['features']) >= target)
        equivalent(point,record['matched_candidate']); assert fits['task_matched']['features'] == point['features']
        assert record['count_exact'] == (len(point['features']) == target)
        counts['count_mismatches'] += not record['count_exact']
        counts['matched_unchanged_supports'] += fits['task_matched']['features'] == fits['old']['features']
        for m in ('task_native','uniform_native'): counts[m+'_unchanged_supports'] += fits[m]['features'] == fits['old']['features']
        counts['task_uniform_same_supports'] += fits['task_native']['features'] == fits['uniform_native']['features']
        counts['teacher_fallbacks'] += exact_sensitivity['uniform_fallback']
        assert record['query_seed'] == 144+i; rng = np.random.default_rng(144+i)
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for m,fit in fits.items():
            assert fit['C'] == fit['pair_penalty'] == 1.
            fs = fit['features']; assert fit['edges'] == [sorted([fs[u],fs[v]]) for u,v in interaction_tree(A[:,fs],labels)['edges']]
            check_model(A,labels,fit,counts)
            values = [BA(score(Q,fit),fit['classes'],y[ri[b]]) for Q in queries]
            equivalent(record['metrics'][m],{'clean':values[0],'noise':float(np.mean(values[1:])),'removed':1-len(fs)/d})
            counts['query_predictions'] += 4; counts['final_fit_caps'] += not fit['optimizer_converged']
        counts['records'] += 1; print(name,i,j,'task reference audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']]))
        for metric in ('clean','noise','removed')} for m in METHODS}}
    checkpoint.write_text(json.dumps({'fingerprints':fingerprints,'result':result},indent=2),encoding='utf-8'); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text())
    names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    if args.only: print(rows,flush=True); return
    lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]
    equivalent(summary['rows'],[{'dataset':r['dataset'],'methods':r['methods']} for r in rows])
    counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in summary['counts'].items(): assert counts[k] == v
    assert counts['records'] == 128
    for metric in ('clean','noise'):
        for m in METHODS[1:]: equivalent(summary['contrasts'][metric][m],paired([r['methods'][m][metric]-r['methods']['old'][metric] for r in rows]))
        equivalent(summary['task_vs_uniform'][metric],paired([r['methods']['task_native'][metric]-r['methods']['uniform_native'][metric] for r in rows]))
    for m in METHODS:
        for metric in ('clean','noise','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
    gate = (summary['contrasts']['clean']['task_native']['mean_pp'] >= .5
        and summary['contrasts']['clean']['task_native']['sign_flip_two_sided_p'] < .05
        and summary['contrasts']['clean']['task_matched']['mean_pp'] >= .5
        and counts['count_mismatches'] == counts['final_fit_caps'] == 0)
    assert gate == summary['gate_passed']
    result = {'passed':True,'counts':dict(counts),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'independent_raw_sensitivity_projection_new_reference_certificates_fits_queries_all15_all128',
        'limits':'Changed_full_reference_relation; exposed_development_not_modern_baseline_superiority_or_population_guarantee'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__ == '__main__': main()
