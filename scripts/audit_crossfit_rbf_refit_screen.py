"""Independent raw-RBF refits of ALL losses and deterministic selection replay."""
import argparse,json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from screen_crossfit_rbf_refit import FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from audit_forest_compact_transfer import equivalent


def audit(name):
    threadpool_limits(1); source = source_verified(name); artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    assert saved['complete'] and not saved['query_labels_used']
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    fingerprints = {'artifact_sha256':digest(artifact),'auditor_sha256':digest(Path(__file__))}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior['fingerprints'] == fingerprints: return prior['result']
    X,y = load_dataset(name); counts = Counter(); gains = []; max_hinge_error = 0.
    expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(saved['rows']) == len(expected)
    for r,(i,j,s,inner) in zip(saved['rows'],expected):
        assert r['outer'] == i and r['inner'] == j and r['outer_reference_indices'] == s['reference_indices'] and r['reference_indices'] == inner['reference_indices']
        ri,a = np.array(r['outer_reference_indices']),np.array(r['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; d = raw.shape[1]
        splits = inner_splits(labels,seed=20261020); assert len(splits) == len(r['folds']); folds = []
        for (ca,cb),fold in zip(splits,r['folds']):
            assert fold['reference_indices'] == ca.tolist() and fold['validation_indices'] == cb.tolist() and set(ca).isdisjoint(cb)
            A,B = preprocess_fit(raw[ca],raw[cb]); truth = labels[cb]
            folds.append((A,B,labels[ca],truth,np.array([1/(len(np.unique(truth))*sum(truth == c)) for c in truth])))
        lookup = {}
        for evaluation in r['risk_evaluations']:
            fs = evaluation['features']; key = tuple(fs); assert key not in lookup; assert len(fs) == len(r['path']['initial_features'])
            reported = evaluation['risk']; errors,hinges = [],[]; assert len(reported['fold_fits']) == len(folds)
            for (A,B,train,truth,balance),fit in zip(folds,reported['fold_fits']):
                model = SVC(C=1.,kernel='rbf',gamma='scale',class_weight='balanced',decision_function_shape='ovr').fit(A[:,fs],train)
                assert model.fit_status_ == fit['fit_status'] == 0
                np.testing.assert_allclose(fit['gamma'],model._gamma,rtol=1e-14,atol=1e-14)
                pred = model.predict(B[:,fs]); assert fit['predictions'] == pred.tolist()
                assert len(fit['n_iter']) == len(model.n_iter_) and min(fit['n_iter']) > 0
                errors.append(float(sum(np.mean(pred[truth == c] != c) for c in np.unique(truth))/len(np.unique(truth))))
                score = model.decision_function(B[:,fs]); index = np.searchsorted(model.classes_,truth)
                if score.ndim == 1: margin = (2*index-1)*score
                else:
                    margin = np.array([score[t,c]-max(score[t,k] for k in range(len(model.classes_)) if k != c) for t,c in enumerate(index)])
                hinges.append(float(balance@np.maximum(1-margin,0))); counts['independent_raw_RBF_refits'] += 1
                counts['status_failures'] += model.fit_status_ != 0
            equivalent(reported['balanced_error'],float(np.mean(errors))); equivalent(reported['fold_balanced_error'],errors)
            np.testing.assert_allclose(reported['fold_balanced_hinge'],hinges,atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(reported['balanced_hinge'],np.mean(hinges),atol=1e-6,rtol=1e-6)
            max_hinge_error = max(max_hinge_error,float(np.max(abs(np.array(hinges)-reported['fold_balanced_hinge']))))
            lookup[key] = reported; counts['loss_evaluations'] += 1
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
        def order(value): return (round(value['balanced_error'],12),round(value['balanced_hinge'],12))
        current = initial.copy(); current_risk = lookup[tuple(current)]; stages = []; attempts = iter(path['attempts']); used = {tuple(current)}
        for step in range(2):
            candidates = []; mean_improving = 0
            for remove in current:
                for add in range(d):
                    if add in current: continue
                    fs = sorted([a for a in current if a != remove]+[add]); used.add(tuple(fs)); risk = lookup[tuple(fs)]
                    mean_better = round(risk['balanced_error'],12) < round(current_risk['balanced_error'],12)
                    mean_improving += mean_better
                    if mean_better and all(round(a,12) <= round(b,12) for a,b in zip(risk['fold_balanced_error'],current_risk['fold_balanced_error'])):
                        candidates.append((order(risk),tuple(fs),remove,add,risk))
            candidates.sort(key=lambda row:(row[0],row[1]))
            stages.append({'round':step,'features':current.copy(),'risk':current_risk,'mean_improving_candidates':mean_improving,
                'foldwise_improving_candidates':len(candidates),'top_pool':[{'features':list(fs),'remove':remove,'add':add,'risk':risk} for _,fs,remove,add,risk in candidates[:8]]})
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
        print(name,i,j,'independent actual RBF refit screen passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'mean_internal_error_gain_pp':100*float(np.mean(gains)),'max_hinge_error':max_hinge_error}
    checkpoint.write_text(json.dumps({'fingerprints':fingerprints,'result':result},indent=2),encoding='utf-8'); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    if args.only: print(audit(args.only),flush=True)
    else:
        summary = json.loads((FOLDER/'complete.json').read_text()); names = [r['dataset'] for r in summary['rows']]
        with ProcessPoolExecutor(max_workers=2) as pool:
            rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
        lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]; counts = sum((Counter(r['counts']) for r in rows),Counter())
        for k,v in summary['counts'].items(): assert counts['independent_raw_RBF_refits' if k == 'refits' else k] == v
        for saved in summary['rows']: equivalent(saved['mean_internal_error_gain_pp'],lookup[saved['dataset']]['mean_internal_error_gain_pp'])
        mean = float(np.mean([r['mean_internal_error_gain_pp'] for r in rows])); equivalent(summary['mean_internal_error_gain_pp'],mean)
        gate = mean >= .5 and counts['error_improves'] >= 20 and counts['count_mismatches'] == counts['status_failures'] == 0
        assert summary['numeric_resource_gate'] == gate and not summary['query_labels_used'] and not summary['goal_achieved']
        result = {'passed':True,'counts':dict(counts),'resource_gate':bool(gate),'mean_internal_error_gain_pp':mean,
            'max_hinge_error':max(r['max_hinge_error'] for r in rows),'auditor_sha256':digest(Path(__file__)),
            'scope':'independent_raw_RBF_every_loss_all_foldwise_choices_FRS_provenance_training_only',
            'limits':'adaptive_selection_scores_not_unbiased_complete_selector_risk_or_prediction_advantage'}
        (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
