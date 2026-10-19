"""Independent fixed-pool reconstruction, raw-RBF losses, ranking and FRS replay."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native
from screen_stable_support_pool import FOLDER,INPUT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from audit_forest_compact_transfer import equivalent


def audit(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source)
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text()); assert saved['complete'] and not saved['archived_query_labels_used']
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    upstream = json.loads((INPUT/(name+'.json')).read_text()); ua = json.loads((INPUT/(name+'-audit.json')).read_text())
    assert ua['fingerprints']['artifact_sha256'] == digest(INPUT/(name+'.json'))
    assert json.loads((INPUT/'complete-audit.json').read_text())['passed']
    fingerprint = {'artifact_sha256':digest(artifact),'auditor_sha256':digest(Path(__file__))}; checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior['fingerprints'] == fingerprint: return prior['result']
    X,y = load_dataset(name); counts = Counter(); maxima = {'hinge_error':0.}; min_gains,check_gains = [],[]
    expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(saved['rows']) == len(expected) == len(upstream['rows'])
    for r,old,(i,j,s,inner) in zip(saved['rows'],upstream['rows'],expected):
        assert r['outer'] == old['outer'] == i and r['inner'] == old['inner'] == j
        assert r['outer_reference_indices'] == s['reference_indices'] and r['reference_indices'] == inner['reference_indices']
        ri,a = np.array(s['reference_indices']),np.array(inner['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; d = raw.shape[1]
        path = r['path']; initial = old['path']['initial_features']; assert path['initial_features'] == initial and path['selection_seeds'] == [20261020,20261025]
        origins = {tuple(initial):{'old'}}; k = len(initial)
        omitted = sorted(set(range(d))-set(initial))
        for missing in omitted:
            for removed in initial: origins.setdefault(tuple(sorted((set(initial)-{removed})|{missing})),set()).add('old_one_exchange')
        provenance = []
        for method,grid in GRIDS.items():
            for ci,config in enumerate(grid):
                ref = native['splits'][i]['inner'][j]['paths'][method][ci]; fs = sorted(ref['ranking'][:k]); origin = method+'_'+str(ci)
                origins.setdefault(tuple(fs),set()).add(origin)
                provenance.append({'origin':origin,'method':method,'index':ci,'config':config,'ranking':ref['ranking'],'features':fs})
        equivalent(r['provenance'],provenance)
        assert [tuple(row['features']) for row in path['scored_pool']] == sorted(origins)
        inherited = {tuple(e['features']):e['risk'] for e in old['risk_evaluations']}; folds = {}; raw_cache = {}
        for seed in (20261020,20261025,20261026):
            splits = inner_splits(labels,seed=seed); assert len(splits) == len(r['folds'][str(seed)])
            folds[seed] = []
            for (ca,cb),fold in zip(splits,r['folds'][str(seed)]):
                assert fold['reference_indices'] == ca.tolist() and fold['validation_indices'] == cb.tolist() and set(ca).isdisjoint(cb)
                A,B = preprocess_fit(raw[ca],raw[cb]); truth = labels[cb]
                folds[seed].append((A,B,labels[ca],truth,np.array([1/(len(np.unique(truth))*sum(truth == c)) for c in truth])))
        def verify_loss(seed,fs,value):
            key = (seed,tuple(fs))
            if key in raw_cache: equivalent(value,raw_cache[key]); return
            if seed == 20261020 and tuple(fs) in inherited:
                assert value == inherited[tuple(fs)]; raw_cache[key] = value; counts['first_reused'] += 1; return
            errors,hinges = [],[]; assert len(value['fold_fits']) == len(folds[seed])
            for (A,B,train,truth,balance),fit in zip(folds[seed],value['fold_fits']):
                model = SVC(C=1.,kernel='rbf',gamma='scale',class_weight='balanced',decision_function_shape='ovr').fit(A[:,fs],train)
                assert model.fit_status_ == fit['fit_status'] == 0; np.testing.assert_allclose(fit['gamma'],model._gamma,rtol=1e-14,atol=1e-14)
                assert len(fit['n_iter']) == len(model.n_iter_) and min(fit['n_iter']) > 0
                pred = model.predict(B[:,fs]); assert fit['predictions'] == pred.tolist()
                errors.append(float(np.mean([np.mean(pred[truth == c] != c) for c in np.unique(truth)])))
                score = model.decision_function(B[:,fs]); index = np.searchsorted(model.classes_,truth)
                if score.ndim == 1: margin = (2*index-1)*score
                else: margin = np.array([score[t,c]-max(score[t,j] for j in range(len(model.classes_)) if j != c) for t,c in enumerate(index)])
                hinges.append(float(balance@np.maximum(1-margin,0))); counts[{20261020:'first_new_refits',20261025:'second_refits',20261026:'third_refits'}[seed]] += 1
            equivalent(value['fold_balanced_error'],errors); equivalent(value['balanced_error'],float(np.mean(errors)))
            np.testing.assert_allclose(value['fold_balanced_hinge'],hinges,atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(value['balanced_hinge'],np.mean(hinges),atol=1e-6,rtol=1e-6)
            maxima['hinge_error'] = max(maxima['hinge_error'],float(np.max(abs(np.array(hinges)-value['fold_balanced_hinge'])))); raw_cache[key] = value
        for row in path['scored_pool']:
            fs = row['features']; assert row['origins'] == sorted(origins[tuple(fs)]) and len(fs) == k
            for seed,value in zip((20261020,20261025),row['risk_by_partition']): verify_loss(seed,fs,value)
        baseline = [raw_cache[(seed,tuple(initial))] for seed in (20261020,20261025)]; equivalent(path['initial_risk_by_partition'],baseline)
        eligible = []
        for row in path['scored_pool']:
            values = row['risk_by_partition']; gains = [round(b['balanced_error'],12)-round(v['balanced_error'],12) for b,v in zip(baseline,values)]
            equivalent(row['gains'],gains); ok = all(g > 1e-12 for g in gains); assert row['eligible'] == ok
            if ok:
                key = (-round(min(gains),12),-round(sum(gains)/2,12),round(sum(v['balanced_hinge'] for v in values)/2,12),tuple(row['features']))
                eligible.append((key,row))
        eligible.sort(key=lambda p:p[0]); assert path['ranked_eligible'] == [row['features'] for _,row in eligible]
        A,_ = preprocess_fit(raw,raw); original = np.ones(d)/np.sqrt(d)
        protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10; geometries = {}
        def geometry(fs):
            key = tuple(fs)
            if key not in geometries:
                weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
                embedded = np.zeros(d); embedded[fs] = weights
                geometries[key] = {'features':fs,'weights':embedded.tolist(),'certificate':weight_certificate(A,labels,fs,original,weights,protected=protected)}
                counts['geometry_replays'] += 1
            return geometries[key]
        attempts = iter(path['attempts']); selected = initial; selected_values = baseline
        for _,row in eligible:
            actual = next(attempts); want = geometry(row['features']); ok = want['certificate']['feasible']
            equivalent(actual,{'features':row['features'],'gains':row['gains'],'candidate':want,'accepted':ok}); counts['attempts'] += 1; counts['rejections'] += not ok
            if ok: selected = row['features']; selected_values = row['risk_by_partition']; break
        assert next(attempts,None) is None and path['features'] == selected
        equivalent(path['candidate'],geometry(selected)); assert path['candidate']['certificate']['feasible']; equivalent(path['final_risk_by_partition'],selected_values)
        # Selection is fully reconstructed WITHOUT reading third-partition scores.
        for method,fs in [('old',initial),('new',selected)]: verify_loss(20261026,fs,r['third_check'][method])
        minimum = 100*min(b['balanced_error']-v['balanced_error'] for b,v in zip(baseline,selected_values))
        check = 100*(r['third_check']['old']['balanced_error']-r['third_check']['new']['balanced_error'])
        equivalent(r['minimum_selection_gain_pp'],minimum); equivalent(r['third_check_gain_pp'],check); min_gains.append(minimum); check_gains.append(check)
        want_counts = {'pool_size':len(origins),'first_reused':sum(fs in inherited for fs in origins),
            'first_new_refits':sum(len(folds[20261020]) for fs in origins if fs not in inherited),
            'second_refits':len(origins)*len(folds[20261025]),'third_refits':len({tuple(initial),tuple(selected)})*len(folds[20261026])}
        equivalent(r['counts'],want_counts); counts['pool_size'] += len(origins)
        counts['references'] += 1; counts['changed_supports'] += selected != initial; counts['third_improved_references'] += check > 1e-10
        counts['count_mismatches'] += len(selected) != k; counts['status_failures'] += 0
        print(name,i,j,'independent stable pool and third check passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'minimum_selection_gain_pp':float(np.mean(min_gains)),
        'third_check_gain_pp':float(np.mean(check_gains)),'precision_maxima':maxima}
    checkpoint.write_text(json.dumps({'fingerprints':fingerprint,'result':result},indent=2),encoding='utf-8'); return result


if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text()); names = [r['dataset'] for r in summary['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    lookup = {r['dataset']:r for r in rows}; counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in summary['counts'].items(): assert counts[k] == v
    for r in summary['rows']:
        for k in ('minimum_selection_gain_pp','third_check_gain_pp'): equivalent(r[k],lookup[r['dataset']][k])
    minimum,check = (float(np.mean([r[k] for r in rows])) for k in ('minimum_selection_gain_pp','third_check_gain_pp'))
    equivalent(summary['mean_minimum_selection_gain_pp'],minimum); equivalent(summary['mean_third_check_gain_pp'],check)
    gate = minimum >= .5 and check >= .5 and counts['third_improved_references'] >= 20 and counts['count_mismatches'] == counts['status_failures'] == 0
    assert summary['numeric_resource_gate'] == gate and not summary['goal_achieved'] and not summary['archived_query_labels_used']
    result = {'passed':True,'counts':dict(counts),'resource_gate':bool(gate),'mean_minimum_selection_gain_pp':minimum,'mean_third_check_gain_pp':check,
        'max_hinge_error':max(r['precision_maxima']['hinge_error'] for r in rows),'auditor_sha256':digest(Path(__file__)),
        'scope':'independent_raw_RBF_or_exact_audited_reuse_allfixedpool_sources_FRScert_selection_before_thirdcheck',
        'limits':'repeated_partitions_same_data_adaptive_selection_not_confirmation_or_SOTA'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
