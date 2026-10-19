"""Independent raw-RBF fits, ALL losses, deterministic swaps and FRS replay."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from screen_crossfit_kernel import FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from audit_forest_compact_transfer import equivalent


def raw_scores(model,X):
    values = model.decision_function(X)
    if values.ndim == 1: values = np.column_stack([np.zeros(len(X)),values])
    return values-values.mean(axis=1,keepdims=True)

def audit(name):
    threadpool_limits(1); source = source_verified(name); artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    assert saved['complete'] and not saved['query_labels_used']
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    fingerprints = {'artifact_sha256':digest(artifact),'auditor_sha256':digest(Path(__file__))}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior['fingerprints'] == fingerprints: return prior['result']
    X,y = load_dataset(name); counts = Counter(); gains = []; maxima = {'teacher_scores':0.,'score_mse':0.}
    expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(saved['rows']) == len(expected)
    for r,(i,j,s,inner) in zip(saved['rows'],expected):
        assert r['outer'] == i and r['inner'] == j and r['outer_reference_indices'] == s['reference_indices'] and r['reference_indices'] == inner['reference_indices']
        ri,a = np.array(r['outer_reference_indices']),np.array(r['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; d = raw.shape[1]
        splits = inner_splits(labels,seed=20261020); assert len(splits) == len(r['folds']); folds = []
        for (ca,cb),fold in zip(splits,r['folds']):
            assert fold['reference_indices'] == ca.tolist() and fold['validation_indices'] == cb.tolist() and set(ca).isdisjoint(cb)
            A,B = preprocess_fit(raw[ca],raw[cb]); train,truth = labels[ca],labels[cb]
            gamma = 1/(d*A.var()) if A.var() > 0 else 1.; equivalent(fold['gamma'],gamma)
            model = SVC(C=1.,kernel='rbf',gamma=gamma,class_weight='balanced',decision_function_shape='ovr').fit(A,train)
            assert model.fit_status_ == 0
            mass = np.array([1/(len(model.classes_)*sum(train == c)) for c in train]); center = mass@A
            original = raw_scores(model,B); balance = np.array([1/(len(np.unique(truth))*sum(truth == c)) for c in truth])
            equivalent(fold['center'],center.tolist()); equivalent(fold['balance'],balance.tolist())
            maxima['teacher_scores'] = max(maxima['teacher_scores'],float(abs(np.array(fold['original_scores'])-original).max()))
            np.testing.assert_allclose(fold['original_scores'],original,atol=1e-6,rtol=1e-6)
            folds.append((model,B,truth,center,original,balance)); counts['independent_raw_RBF_refits'] += 1
        lookup = {}; validated = {}
        for evaluation in r['risk_evaluations']:
            fs = evaluation['features']; key = tuple(fs); assert key not in lookup
            errors,mse = [],[]
            for model,B,truth,center,original,balance in folds:
                Q = B.copy(); omitted = [a for a in range(d) if a not in fs]; Q[:,omitted] = center[omitted]
                prediction = model.predict(Q); values = raw_scores(model,Q)
                errors.append(float(balance@(prediction != truth))); mse.append(float(balance@np.mean((values-original)**2,axis=1)))
            expected_risk = {'balanced_error':float(np.mean(errors)),'score_mse':float(np.mean(mse)),
                'fold_balanced_error':errors,'fold_score_mse':mse}
            reported = evaluation['risk']; equivalent(reported['balanced_error'],expected_risk['balanced_error'])
            equivalent(reported['fold_balanced_error'],errors)
            np.testing.assert_allclose(reported['fold_score_mse'],mse,atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(reported['score_mse'],expected_risk['score_mse'],atol=1e-6,rtol=1e-6)
            maxima['score_mse'] = max(maxima['score_mse'],abs(reported['score_mse']-expected_risk['score_mse']))
            lookup[key] = reported; validated[key] = expected_risk; counts['loss_evaluations'] += 1
        A,_ = preprocess_fit(raw,raw); original = np.ones(d)/np.sqrt(d)
        protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10; cache = {}
        def geometry(fs):
            key = tuple(fs)
            if key not in cache:
                w = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
                embedded = np.zeros(d); embedded[fs] = w
                cache[key] = {'features':fs,'weights':embedded.tolist(),'certificate':weight_certificate(A,labels,fs,original,w,protected=protected)}
                counts['geometry_replays'] += 1
            return cache[key]
        path = r['path']; initial = inner['models'][inner['stages'][7]['policies']['0.8']['model_id']]['features']
        assert path['initial_features'] == initial and path['pool'] == 8 and path['rounds'] == 2 and path['C'] == 1. and path['seed'] == 20261020
        def order(value): return (round(value['balanced_error'],12),round(value['score_mse'],12))
        current = initial.copy(); current_risk = lookup[tuple(current)]; stages = []; attempts = iter(path['attempts']); used = {tuple(current)}
        for step in range(2):
            candidates = []
            for remove in current:
                for add in range(d):
                    if add in current: continue
                    fs = sorted([a for a in current if a != remove]+[add]); used.add(tuple(fs)); risk = lookup[tuple(fs)]
                    if order(risk) < order(current_risk): candidates.append((order(risk),tuple(fs),remove,add,risk))
            candidates.sort(key=lambda row:(row[0],row[1]))
            stages.append({'round':step,'features':current.copy(),'risk':current_risk,'improving_candidates':len(candidates),
                'top_pool':[{'features':list(fs),'remove':remove,'add':add,'risk':risk} for _,fs,remove,add,risk in candidates[:8]]})
            accepted = None
            for _,fs,remove,add,risk in candidates[:8]:
                actual = next(attempts); want = geometry(list(fs)); ok = want['certificate']['feasible']
                equivalent(actual,{'round':step,'remove':remove,'add':add,'risk':risk,'candidate':want,'accepted':ok})
                counts['attempts'] += 1; counts['rejections'] += not ok
                if ok: accepted = list(fs); current_risk = risk; break
            if accepted is None: break
            current = accepted
        assert next(attempts,None) is None and used == set(lookup)
        equivalent(path['stages'],stages); assert path['features'] == current; equivalent(path['candidate'],geometry(current))
        equivalent(path['initial_risk'],lookup[tuple(initial)]); equivalent(path['final_risk'],current_risk)
        assert path['candidate']['certificate']['feasible'] and order(current_risk) <= order(lookup[tuple(initial)])
        gains.append(lookup[tuple(initial)]['balanced_error']-current_risk['balanced_error'])
        counts['references'] += 1; counts['error_improves'] += round(current_risk['balanced_error'],12) < round(lookup[tuple(initial)]['balanced_error'],12)
        counts['changed_supports'] += current != initial; counts['count_mismatches'] += len(current) != len(initial)
        print(name,i,j,'independent crossfit kernel screen passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'mean_internal_error_gain_pp':100*float(np.mean(gains)),'precision_maxima':maxima}
    checkpoint.write_text(json.dumps({'fingerprints':fingerprints,'result':result},indent=2),encoding='utf-8'); return result

if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text()); names = [r['dataset'] for r in summary['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]; counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in summary['counts'].items(): assert counts[k] == v
    for saved in summary['rows']: equivalent(saved['mean_internal_error_gain_pp'],lookup[saved['dataset']]['mean_internal_error_gain_pp'])
    mean = float(np.mean([r['mean_internal_error_gain_pp'] for r in rows])); equivalent(summary['mean_internal_error_gain_pp'],mean)
    gate = mean >= .5 and counts['error_improves'] >= 20 and counts['count_mismatches'] == 0
    assert summary['numeric_resource_gate'] == gate and not summary['query_labels_used'] and not summary['goal_achieved']
    result = {'passed':True,'counts':dict(counts),'resource_gate':bool(gate),'mean_internal_error_gain_pp':mean,
        'precision_maxima':{k:max(r['precision_maxima'][k] for r in rows) for k in rows[0]['precision_maxima']},
        'auditor_sha256':digest(Path(__file__)),'scope':'independent_raw_RBF_all_losses_full_swap_replay_FRS_provenance_training_only',
        'limits':'adaptively_selected_internal_validation_is_optimistic_not_test_prediction_advantage'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
