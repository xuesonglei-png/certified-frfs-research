"""Independent entropy calibration, raw-score path, FRS and all box/refit checks."""
import argparse,json,itertools
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native,holm
from run_calibrated_forest_pilot import FOLDER,METHODS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from audit_nested_geometry_forest import BA,check_model
from audit_forest_compact_transfer import equivalent
from spline_forest_audit_oracle import score,leaf_margins,lp_margins_first
from run_forest_robustness_curve import coverage
from review_box_iterations import paired


def entropy(probabilities,axes):
    values = np.zeros_like(probabilities); positive = probabilities > 0
    values[positive] = -probabilities[positive]*np.log(probabilities[positive])
    return values.sum(axis=axes)

def mi(codes,truth,states,q):
    n,m = codes.shape; counts = np.zeros((m,states,q))
    np.add.at(counts,(np.arange(m)[None,:],codes,truth[:,None]),1)
    probabilities = counts/n
    return entropy(probabilities.sum(axis=2),1)+entropy(probabilities.sum(axis=1),1)-entropy(probabilities,(1,2))

def calibration(A,y,saved,counts):
    _,labels = np.unique(y,return_inverse=True); q = len(np.unique(y)); d = A.shape[1]
    bins = np.searchsorted([.25,.75],A,side='left'); edges = [(u,v) for u in range(d) for v in range(u+1,d)]
    assert saved['seed'] == 20261019 and saved['permutations'] == 31 and saved['alpha'] == .05
    if not edges:
        assert saved['edges'] == [] and saved['candidates'] == []; return []
    assert saved['score_decimal_places'] == 12
    pairs = np.column_stack([3*bins[:,u]+bins[:,v] for u,v in edges]); u,v = np.array(edges).T
    def stats(truth):
        single = mi(bins,truth,3,q); joint = mi(pairs,truth,9,q)
        return np.round(joint-single[u]-single[v],12),single,joint
    observed,single,joint = stats(labels); np.testing.assert_allclose(saved['single_MI'],single,atol=1e-12,rtol=1e-12)
    rng = np.random.default_rng(20261019); indices = [rng.permutation(len(A)).tolist() for _ in range(31)]
    assert saved['permutation_indices'] == indices
    maxima = [float(max(stats(labels[order])[0])) for order in indices]
    np.testing.assert_allclose(saved['null_maxima'],maxima,atol=1e-12,rtol=1e-12)
    expected = []
    for i,edge in enumerate(edges):
        p = (1+sum(m >= observed[i] for m in maxima))/32
        expected.append({'edge':list(edge),'synergy':float(observed[i]),'joint_MI':float(joint[i]),
            'maxT_p':p,'eligible':bool(observed[i] > 0 and p <= .05)})
    expected.sort(key=lambda row:(-row['synergy'],row['edge']))
    assert len(saved['candidates']) == len(expected)
    for want,actual in zip(expected,saved['candidates']):
        assert want['edge'] == actual['edge'] and want['maxT_p'] == actual['maxT_p'] and want['eligible'] == actual['eligible']
        np.testing.assert_allclose(actual['synergy'],want['synergy'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(actual['joint_MI'],want['joint_MI'],atol=1e-12,rtol=1e-12)
    groups = [{a} for a in range(d)]; selected = []
    for row in expected:
        if not row['eligible']: continue
        a,b = row['edge']; ga = next(g for g in groups if a in g); gb = next(g for g in groups if b in g)
        if ga is not gb: selected.append(row['edge']); groups.remove(ga); groups.remove(gb); groups.append(ga|gb)
    assert saved['edges'] == selected
    counts['entropy_calibrations'] += 1; counts['label_permutations'] += 31; counts['observed_pair_statistics'] += len(edges)
    return selected

def path_replay(A,y,stage,geometry,counts):
    teacher = stage['policies']['1.0']['fit']; d = A.shape[1]; q = len(teacher['classes'])
    assert stage['C'] == stage['pair_penalty'] == 1. and not stage['teacher_guards_enforced']
    mass = np.array([1/(q*sum(y == c)) for c in y]); center = mass@A
    target = score(A,{**teacher,'knots':np.tile([0,.5,1.],(d,1)).tolist()}); target -= target.mean(axis=1,keepdims=True)
    def loss(fs):
        replaced = A.copy(); omitted = [a for a in range(d) if a not in fs]; replaced[:,omitted] = center[omitted]
        value = score(replaced,{**teacher,'knots':np.tile([0,.5,1.],(d,1)).tolist()}); value -= value.mean(axis=1,keepdims=True)
        counts['raw_projection_probes'] += 1; return float(mass@np.mean((value-target)**2,axis=1))
    def check(record,fs,cost):
        got = geometry(fs)
        for key in ('features','weights','certificate'): equivalent(record[key],got[key])
        np.testing.assert_allclose(record['projection_loss'],cost,atol=1e-11,rtol=1e-9)
        return got['certificate']['feasible']
    path = stage['path']; np.testing.assert_allclose(path['center'],center,atol=1e-12); assert path['tie_decimals'] == 12
    active = list(range(d)); assert check(path['trace'][0],active,0); ti = ai = 0; accepted = False
    for rank in path['rankings']:
        assert rank['features'] == active and len(active) > int(np.ceil(d/2))
        choices = sorted([(loss([a for a in active if a != removed]),removed) for removed in active],key=lambda row:(round(row[0],12),row[1]))
        assert [c['removed'] for c in rank['choices']] == [a for _,a in choices]
        np.testing.assert_allclose([c['loss'] for c in rank['choices']],[value for value,_ in choices],atol=1e-11,rtol=1e-9)
        accepted = False
        for cost,removed in choices:
            fs = [a for a in active if a != removed]; attempt = path['attempts'][ai]; ai += 1
            assert attempt['removed'] == removed; ok = check(attempt,fs,cost); assert attempt['accepted'] == ok
            counts['attempts_replayed'] += 1
            if ok:
                ti += 1; assert check(path['trace'][ti],fs,cost); active = fs; accepted = True; break
        if not accepted: assert rank is path['rankings'][-1]
    assert ai == len(path['attempts']) and ti+1 == len(path['trace']) and (len(active) == int(np.ceil(d/2)) or not accepted)
    for fraction in (.5,.65,.8,1.):
        chosen = next(row for row in reversed(path['trace']) if len(row['features']) >= int(np.ceil(fraction*d)))
        policy = stage['policies'][str(fraction)]; equivalent(policy['candidate'],chosen)
        assert policy['fit']['features'] == chosen['features']

def audit(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source)
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text()); assert saved['complete']
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    fingerprints = {p.name:digest(p) for p in [artifact,Path(__file__),ROOT/'scripts/audit_nested_geometry_forest.py',ROOT/'scripts/spline_forest_audit_oracle.py']}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['fingerprints'] == fingerprints: return previous['result']
    X,y = load_dataset(name); counts = Counter(); expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(saved['records']) == len(expected)
    for r,(i,j,s,inner) in zip(saved['records'],expected):
        assert r['outer_index'] == i and r['inner_index'] == j and r['outer_reference_indices'] == s['reference_indices']
        for key in ('reference_indices','validation_indices'): assert r[key] == inner[key]
        ri,a,b = (np.array(r[k]) for k in ('outer_reference_indices','reference_indices','validation_indices')); assert set(a).isdisjoint(b)
        A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels,truth = y[ri[a]],y[ri[b]]; d = A.shape[1]; fits = r['fits']
        graph = calibration(A,labels,r['calibration'],counts)
        counts['selected_edges'] += len(graph); counts['eligible_edges'] += sum(c['eligible'] for c in r['calibration']['candidates']); counts['zero_edge_references'] += not graph
        original = np.ones(d)/np.sqrt(d); protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10; cache = {}
        def geometry(fs):
            key = tuple(fs)
            if key not in cache:
                weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
                embedded = np.zeros(d); embedded[fs] = weights
                cache[key] = {'features':fs,'weights':embedded.tolist(),'certificate':weight_certificate(A,labels,fs,original,weights,protected=protected)}
                counts['geometry_replays'] += 1
            return cache[key]
        stage = r['stage']; path_replay(A,labels,stage,geometry,counts); target = len(stage['policies']['0.8']['candidate']['features']); assert r['target_count'] == target
        assert {k:v for k,v in fits['new'].items() if k != 'knots'} == stage['policies']['0.8']['fit']
        assert {k:v for k,v in fits['full'].items() if k != 'knots'} == stage['policies']['1.0']['fit']
        oldstage = inner['stages'][7]
        for method,fraction in [('old_original','0.8'),('full_original','1.0')]:
            assert {k:v for k,v in fits[method].items() if k != 'knots'} == inner['models'][oldstage['policies'][fraction]['model_id']]
        assert fits['old_support']['features'] == fits['old_original']['features']
        for method,ci,prefix in [('fgs2023',4,'fgs'),('sfrip2025',3,'sfrip')]:
            reference = native['splits'][i]['inner'][j]['paths'][method][ci]; path = r['selector_paths'][method]
            assert path['selector_index'] == ci and path['config'] == GRIDS[method][ci] and path['ranking'] == reference['ranking']
            assert path['native_features'] == sorted(reference['native']['features']) and path['matched_features'] == sorted(reference['ranking'][:target])
            for suffix in ('native','matched'): assert fits[prefix+'_'+suffix]['features'] == path[suffix+'_features']
        exact = all(len(fits[m]['features']) == target for m in ('fgs_matched','sfrip_matched')); assert r['counts_exact'] == exact
        counts['count_mismatches'] += not exact; counts['changed_supports'] += fits['new']['features'] != fits['old_original']['features']
        for fraction,policy in stage['policies'].items():
            fit = policy['fit']; assert fit['edges'] == [e for e in graph if all(a in fit['features'] for a in e)]
            check_model(A,labels,fit,Counter()); counts['stage_policy_refits'] += 1
        assert r['query_seed'] == 144+i; rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for method,fit in fits.items():
            np.testing.assert_allclose(fit['knots'],np.tile([0.,.5,1.],(d,1)),atol=1e-12); assert fit['C'] == fit['pair_penalty'] == 1.
            if method not in ('old_original','full_original'): assert fit['edges'] == [e for e in graph if all(a in fit['features'] for a in e)]
            check_model(A,labels,{k:v for k,v in fit.items() if k != 'knots'},Counter()); counts['classifier_refits'] += 1; counts['optimizer_caps'] += not fit['optimizer_converged']
            values = [BA(score(Q,fit),fit['classes'],truth) for Q in queries]
            pair,_,_ = leaf_margins(B,truth,fit,.01); margins = pair.min(axis=1); box = r['boxes'][method]
            np.testing.assert_allclose(box['margins'],margins,atol=1e-9,rtol=1e-9); rivals = np.array(box['rivals']); assert rivals.tolist() == pair.argmin(axis=1).tolist()
            witness = np.array(box['witness_raw']); assert np.all(witness >= np.maximum(0,B-.01)-1e-12) and np.all(witness <= np.minimum(1,B+.01)+1e-12)
            ws = score(witness,fit); own = np.argmax(truth[:,None] == np.array(fit['classes'])[None],axis=1)
            np.testing.assert_allclose(ws[np.arange(len(B)),own]-ws[np.arange(len(B)),rivals],margins,atol=1e-9,rtol=1e-9)
            for rival,value in lp_margins_first(B,truth,fit,.01).items():
                np.testing.assert_allclose(value,pair[0,rival],atol=1e-8,rtol=1e-8); counts['box_LP'] += 1
            equivalent(r['metrics'][method],{'clean':values[0],'noise':float(np.mean(values[1:])),
                'robust':coverage(margins > 1e-10,truth),'removed':1-len(fit['features'])/d})
            counts['leaf_replays'] += 1; counts['witness_replays'] += len(B); counts['query_predictions'] += 4
        counts['records'] += 1; print(name,i,j,'calibrated forest audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean','noise','robust','removed')} for m in METHODS}}
    checkpoint.write_text(json.dumps({'fingerprints':fingerprints,'result':result},indent=2),encoding='utf-8'); return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    summary = None if args.only else json.loads((FOLDER/'complete.json').read_text()); names = [args.only] if args.only else [r['dataset'] for r in summary['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    if args.only: print(rows,flush=True)
    else:
        lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]; counts = sum((Counter(r['counts']) for r in rows),Counter())
        equivalent(summary['rows'],[{'dataset':r['dataset'],'methods':r['methods']} for r in rows])
        for k,v in summary['counts'].items(): assert counts[k] == v
        for m in METHODS:
            for metric in ('clean','noise','robust','removed'): equivalent(summary['means'][m][metric],float(np.mean([r['methods'][m][metric] for r in rows])))
        for metric in ('clean','noise','robust'):
            for m in METHODS[1:]: equivalent(summary['contrasts'][metric][m],paired([r['methods']['new'][metric]-r['methods'][m][metric] for r in rows]))
        clean = summary['contrasts']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')}); equivalent(summary['matched_holm'],corrected)
        gate = all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched')) and clean['full']['mean_pp'] >= 0 and summary['means']['new']['removed'] >= .1 and counts['count_mismatches'] == counts['optimizer_caps'] == 0
        assert summary['development_numeric_gate'] == gate and not summary['goal_achieved']
        result = {'passed':True,'counts':dict(counts),'auditor_sha256':digest(Path(__file__)),
            'scope':'independent_entropy_permutations_maxT_forest_raw_projection_paths_allcertificates_refits_boxes_queries',
            'limits':'global_null_only; generic_permutation_method; fixed_recent_settings; exposed_development'}
        (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
