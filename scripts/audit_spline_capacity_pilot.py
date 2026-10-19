"""Independent interpolation/refit/tree/geometry/LP audit of five-knot pilot."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from sklearn.metrics import mutual_info_score
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified
from run_nested_geometry_forest import GRIDS,verified_native,holm
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from audit_forest_compact_transfer import equivalent
from audit_nested_geometry_forest import BA
from spline_forest_audit_oracle import membership,score,refit,leaf_margins,lp_margins_first
from run_forest_robustness_curve import coverage
from review_box_iterations import paired

FOLDER = ROOT/'results/spline-capacity-pilot'
METHODS = ('old3','full3','full5_fixed3','full5','old_support5','proposed5','same_new_support3',
    'fgs_native5','sfrip_native5','fgs_matched5','sfrip_matched5')


def independent_tree(A,y,fs,knots):
    if not fs: return []
    bins = membership(A,knots)[:,fs].argmax(axis=-1); k = np.array(knots).shape[1]; n = len(fs)
    singles = [mutual_info_score(bins[:,a],y) for a in range(n)]; choices = []
    for u in range(n):
        for v in range(u+1,n):
            gain = max(0.,mutual_info_score(k*bins[:,u]+bins[:,v],y)-max(singles[u],singles[v]))
            choices.append((gain,u,v))
    choices.sort(key=lambda r:(-r[0],r[1],r[2])); components = [{a} for a in range(n)]; selected = []
    for _,u,v in choices:
        left = next(group for group in components if u in group); right = next(group for group in components if v in group)
        if left is right: continue
        selected.append(sorted([fs[u],fs[v]])); components.remove(left); components.remove(right); components.append(left|right)
        if len(selected) == n-1: break
    return selected


def audit(name):
    threadpool_limits(1); counts = Counter(); precision = {k:0. for k in ('coefficient_error','intercept_error','score_error','gradient_inf','objective_gap')}
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    source = source_verified(name); native = verified_native(name,source); assert saved['complete']
    for rel,value in saved['fingerprints'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == value
    fingerprints = {'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'oracle_sha256':hashlib.sha256((ROOT/'scripts/spline_forest_audit_oracle.py').read_bytes()).hexdigest()}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == fingerprints: return previous['result']
    X,y = load_dataset(name)
    expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(expected) == len(saved['records'])
    for r,(i,j,s,inner) in zip(saved['records'],expected):
        assert r['outer_index'] == i and r['inner_index'] == j and r['outer_reference_indices'] == s['reference_indices']
        for key in ('reference_indices','validation_indices'): assert r[key] == inner[key]
        ri,a,b = (np.array(r[k]) for k in ('outer_reference_indices','reference_indices','validation_indices'))
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(ri)
        A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels,truth = y[ri[a]],y[ri[b]]; d = A.shape[1]
        fits = r['fits']; assert set(fits) == set(METHODS); oldstage = inner['stages'][7]
        def original3(fit): return {k:v for k,v in fit.items() if k != 'knots'}
        assert original3(fits['old3']) == inner['models'][oldstage['policies']['0.8']['model_id']]
        assert original3(fits['full3']) == inner['models'][oldstage['policies']['1.0']['model_id']]
        stage = r['stage']; assert stage['fraction'] == .8 and not stage['teacher_guards_enforced']
        assert stage['teacher'] == fits['full5'] and stage['fit'] == fits['proposed5']
        teacher = stage['teacher']; classes = np.array(teacher['classes']); q = len(classes)
        balance = np.array([1/(q*sum(labels == c)) for c in labels]); center = balance@A
        np.testing.assert_allclose(stage['path']['center'],center,atol=1e-12); assert stage['path']['tie_decimals'] == 12
        target_scores = score(A,teacher); target_scores -= target_scores.mean(axis=1,keepdims=True); probes = {}; geometries = {}
        fullweights = np.ones(d)/np.sqrt(d)
        protected = MajorityTerms(A,labels,weights=fullweights).bounds(range(d),allowed_enemies=2) > 1e-10
        sourcegeometry = {tuple(g['features']):g for g in inner['geometry']}
        def loss(fs):
            key = tuple(fs)
            if key not in probes:
                query = A.copy(); omitted = [a for a in range(d) if a not in fs]; query[:,omitted] = center[omitted]
                values = score(query,teacher); values -= values.mean(axis=1,keepdims=True)
                probes[key] = float(balance @ np.mean((values-target_scores)**2,axis=1)); counts['raw_projection_probes'] += 1
            return probes[key]
        def point(record,fs,value):
            assert record['features'] == fs
            np.testing.assert_allclose(record['projection_loss'],value,atol=1e-11,rtol=1e-9); key = tuple(fs)
            if key not in geometries:
                if key in sourcegeometry:
                    geometries[key] = sourcegeometry[key]; counts['verified_source_geometries'] += 1
                else:
                    weights = fullweights if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,fullweights)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
                    embedded = np.zeros(d); embedded[fs] = weights
                    geometries[key] = {'features':fs,'weights':embedded.tolist(),
                        'certificate':weight_certificate(A,labels,fs,fullweights,weights,protected=protected)}
                    counts['new_geometry_replays'] += 1
            geometry = geometries[key]; equivalent(record['weights'],geometry['weights']); equivalent(record['certificate'],geometry['certificate'])
            return geometry['certificate']['feasible']
        path = stage['path']; active = list(range(d)); ai = ti = 0; accepted = False
        assert point(path['trace'][0],active,0.)
        for ranking in path['rankings']:
            assert ranking['features'] == active and len(active) > int(np.ceil(.8*d))
            ordered = sorted([(loss([a for a in active if a != removed]),removed) for removed in active],key=lambda row:(round(row[0],12),row[1]))
            assert [p['removed'] for p in ranking['choices']] == [removed for _,removed in ordered]
            np.testing.assert_allclose([p['loss'] for p in ranking['choices']],[value for value,_ in ordered],atol=1e-11,rtol=1e-9)
            accepted = False
            for value,removed in ordered:
                attempt = path['attempts'][ai]; ai += 1; fs = [a for a in active if a != removed]
                assert attempt['removed'] == removed; feasible = point(attempt,fs,value); assert attempt['accepted'] == feasible
                counts['attempts'] += 1
                if feasible:
                    ti += 1; assert point(path['trace'][ti],fs,value); active = fs; accepted = True; break
            if not accepted: assert ranking is path['rankings'][-1]
        assert ai == len(path['attempts']) and ti+1 == len(path['trace']) and (len(active) == int(np.ceil(.8*d)) or not accepted)
        equivalent(stage['candidate'],path['trace'][-1]); assert fits['proposed5']['features'] == active
        assert stage['candidate']['certificate']['feasible'] and r['target_count'] == len(active)
        assert fits['same_new_support3']['features'] == active and fits['old_support5']['features'] == fits['old3']['features']
        assert fits['full5_fixed3']['edges'] == fits['full3']['edges']
        for method,ci in [('fgs2023',4),('sfrip2025',3)]:
            sourcepath = native['splits'][i]['inner'][j]['paths'][method][ci]; path = r['selector_paths'][method]
            assert path['selector_index'] == ci and path['config'] == GRIDS[method][ci]
            assert path['ranking'] == sourcepath['ranking'] and sorted(path['ranking']) == list(range(d))
            assert path['native_features'] == sorted(sourcepath['native']['features'])
            assert path['matched_features'] == sorted(sourcepath['ranking'][:len(active)])
            prefix = 'fgs' if method == 'fgs2023' else 'sfrip'
            assert fits[prefix+'_native5']['features'] == path['native_features']
            assert fits[prefix+'_matched5']['features'] == path['matched_features']
        exact = all(len(fits[m]['features']) == len(active) for m in ('fgs_matched5','sfrip_matched5'))
        assert r['counts_exact'] == exact; counts['count_mismatches'] += not exact
        counts['old_new_supports_equal'] += fits['old3']['features'] == active
        assert r['query_seed'] == 144+i; rng = np.random.default_rng(144+i)
        queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        tree_cache = {}; checked_models = []
        for method,fit in fits.items():
            expectedknots = [0.,.5,1.] if method in ('old3','full3','same_new_support3') else [0.,.25,.5,.75,1.]
            np.testing.assert_allclose(fit['knots'],np.tile(expectedknots,(d,1)),atol=1e-12)
            assert fit['C'] == fit['pair_penalty'] == 1.
            fs = fit['features']; tree_key = (tuple(fs),tuple(expectedknots))
            if method != 'full5_fixed3':
                if tree_key not in tree_cache: tree_cache[tree_key] = independent_tree(A,labels,fs,fit['knots'])
                assert fit['edges'] == tree_cache[tree_key]
            if not any(fit == p for p in checked_models):
                diagnostics = refit(A,labels,fit)
                for key,value in diagnostics.items(): precision[key] = max(precision[key],value)
                checked_models.append(fit); counts['independent_classifier_refits'] += 1
            values = [BA(score(Q,fit),fit['classes'],truth) for Q in queries]
            pair,points,Z = leaf_margins(B,truth,fit,.01); margins = pair.min(axis=1); box = r['boxes'][method]
            np.testing.assert_allclose(box['margins'],margins,atol=1e-9,rtol=1e-9)
            rivals = np.array(box['rivals']); assert rivals.tolist() == pair.argmin(axis=1).tolist()
            witness = np.array(box['witness_raw']); lo,hi = np.maximum(0,B-.01),np.minimum(1,B+.01)
            assert witness.shape == B.shape and np.all(witness >= lo-1e-12) and np.all(witness <= hi+1e-12)
            wscores = score(witness,fit); own = np.argmax(truth[:,None] == np.array(fit['classes'])[None],axis=1)
            np.testing.assert_allclose(wscores[np.arange(len(B)),own]-wscores[np.arange(len(B)),rivals],margins,atol=1e-9,rtol=1e-9)
            for rival,value in lp_margins_first(B,truth,fit,.01).items():
                np.testing.assert_allclose(value,pair[0,rival],atol=1e-8,rtol=1e-8); counts['independent_lp_solves'] += 1
            equivalent(r['metrics'][method],{'clean':values[0],'noise':float(np.mean(values[1:])),
                'robust':coverage(margins > 1e-10,truth),'removed':1-len(fs)/d})
            counts['optimizer_caps'] += not fit['optimizer_converged']; counts['independent_leaf_replays'] += 1
            counts['raw_witness_replays'] += len(B); counts['query_predictions'] += 4
        counts['records'] += 1; print(name,i,j,'spline capacity audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'precision_maxima':precision,'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']]))
        for metric in ('clean','noise','robust','removed')} for m in METHODS}}
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
    compare = lambda first,second,metric:paired([r['methods'][first][metric]-r['methods'][second][metric] for r in rows])
    for metric in ('clean','noise','robust'):
        for key,first,second in [('basis_only','full5_fixed3','full3'),('five_state_tree','full5','full5_fixed3'),
            ('new_support_capacity','proposed5','same_new_support3'),('path_change','proposed5','old_support5'),('new_vs_old','proposed5','old3')]:
            equivalent(summary['ablations'][metric][key],compare(first,second,metric))
        for m in ('full5','fgs_native5','sfrip_native5','fgs_matched5','sfrip_matched5'):
            equivalent(summary['method_contrasts'][metric][m],compare('proposed5',m,metric))
    for m in METHODS:
        for metric in ('clean','noise','robust','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
    clean = summary['method_contrasts']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full5','fgs_matched5','sfrip_matched5')})
    equivalent(summary['matched_holm'],corrected); basis = summary['ablations']['clean']['basis_only']
    assert summary['representation_gate_passed'] == (basis['mean_pp'] >= .5 and basis['sign_flip_two_sided_p'] < .05)
    gate = all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched5','sfrip_matched5')) and clean['full5']['mean_pp'] >= 0 and counts['count_mismatches'] == counts['optimizer_caps'] == 0
    assert summary['method_gate_passed'] == gate and summary['goal_achieved'] is False
    result = {'passed':True,'counts':dict(counts),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'precision_maxima':{key:max(r['precision_maxima'][key] for r in rows) for key in rows[0]['precision_maxima']},
        'oracle_sha256':hashlib.sha256((ROOT/'scripts/spline_forest_audit_oracle.py').read_bytes()).hexdigest(),
        'scope':'all15_all128_independent_hat_interpolation_fits_trees_projection_FRS_geometry_leaf_LP_witnesses_queries',
        'limits':'Fixed_config_development_capacity_diagnostic_not_nested_SOTA_or_population_accuracy'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)


if __name__ == '__main__': main()
