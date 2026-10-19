"""Every repartition raw-model refit, manual kernel score and paired summary."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from diagnose_refit_repartition import FOLDER,SOURCES
from audit_crossfit_rbf_refit_prediction import manual_svm_scores
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from audit_forest_compact_transfer import equivalent


def audit(name):
    threadpool_limits(1); original = source_verified(name); X,y = load_dataset(name)
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text()); assert saved['complete'] and not saved['archived_query_labels_used']
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    inputs = {}
    for method,path in SOURCES.items():
        inputs[method] = json.loads((path/(name+'.json')).read_text()); upstream = json.loads((path/(name+'-audit.json')).read_text())
        assert upstream['fingerprints']['artifact_sha256'] == digest(path/(name+'.json'))
    counts = Counter(); max_kernel_error = 0.; dataset_gains = {'selection':{m:[] for m in SOURCES},'repartition':{m:[] for m in SOURCES}}
    expected = [(i,j,s,f) for i,s in enumerate(original['splits']) for j,f in enumerate(s['inner'])]; assert len(expected) == len(saved['rows'])
    for index,(r,(i,j,s,inner)) in enumerate(zip(saved['rows'],expected)):
        assert r['outer'] == i and r['inner'] == j and r['seed'] == 20261025
        assert r['outer_reference_indices'] == s['reference_indices'] and r['reference_indices'] == inner['reference_indices']
        ri,a = np.array(s['reference_indices']),np.array(inner['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]
        supports = r['supports']; assert supports['old'] == inputs['refit']['rows'][index]['path']['initial_features']
        for m in SOURCES: assert supports[m] == inputs[m]['rows'][index]['path']['features'] and len(supports[m]) == len(supports['old'])
        splits = inner_splits(labels,seed=20261025); assert len(splits) == len(r['folds']); errors = {m:[] for m in supports}
        for (ca,cb),fold in zip(splits,r['folds']):
            assert fold['reference_indices'] == ca.tolist() and fold['validation_indices'] == cb.tolist() and set(ca).isdisjoint(cb)
            A,B = preprocess_fit(raw[ca],raw[cb]); truth = labels[cb]
            for m,fs in supports.items():
                model = SVC(C=1.,kernel='rbf',gamma='scale',class_weight='balanced',decision_function_shape='ovr').fit(A[:,fs],labels[ca])
                fit = fold['fits'][m]; assert fit['fit_status'] == model.fit_status_ == 0
                assert fit['classes'] == model.classes_.tolist() and fit['support'] == model.support_.tolist()
                equivalent(fit['gamma'],float(model._gamma)); assert fit['n_iter'] == model.n_iter_.tolist() and fit['n_support'] == model.n_support_.tolist()
                np.testing.assert_allclose(fit['support_vectors'],model.support_vectors_,atol=1e-12)
                np.testing.assert_allclose(fit['dual_coef'],model.dual_coef_,atol=1e-10); np.testing.assert_allclose(fit['intercept'],model.intercept_,atol=1e-10)
                pred = model.predict(B[:,fs]); score = model.decision_function(B[:,fs]); report = fold['evaluations'][m]
                assert report['predictions'] == pred.tolist(); np.testing.assert_allclose(report['scores'],score,atol=1e-10)
                manual = manual_svm_scores(B[:,fs],fit); np.testing.assert_allclose(manual,score,atol=1e-8)
                max_kernel_error = max(max_kernel_error,float(np.max(abs(manual-score))))
                error = float(np.mean([np.mean(pred[truth == c] != c) for c in np.unique(truth)])); equivalent(report['balanced_error'],error)
                errors[m].append(error); counts['models'] += 1
        mean = {m:float(np.mean(v)) for m,v in errors.items()}; equivalent(r['mean_errors'],mean)
        for m in SOURCES:
            selection = 100*(inputs[m]['rows'][index]['path']['initial_risk']['balanced_error']-inputs[m]['rows'][index]['path']['final_risk']['balanced_error'])
            repartition = 100*(mean['old']-mean[m]); equivalent(r['selection_gains_pp'][m],selection); equivalent(r['repartition_gains_pp'][m],repartition)
            dataset_gains['selection'][m].append(selection); dataset_gains['repartition'][m].append(repartition)
            selected = selection > 1e-10; counts[m+'_selected_improvements'] += selected
            counts[m+'_repartition_positive_of_selected'] += selected and repartition > 1e-10
            counts[m+'_repartition_negative_of_selected'] += selected and repartition < -1e-10
            counts[m+'_repartition_tied_of_selected'] += selected and abs(repartition) <= 1e-10
        counts['references'] += 1
    result = {'dataset':name,'counts':dict(counts),'max_kernel_error':max_kernel_error,
        'mean_gains_pp':{s:{m:float(np.mean(v)) for m,v in values.items()} for s,values in dataset_gains.items()}}
    (FOLDER/(name+'-audit.json')).write_text(json.dumps({'fingerprints':{'artifact_sha256':digest(artifact),'auditor_sha256':digest(ROOT/'scripts/audit_refit_repartition_diagnosis.py')},'result':result},indent=2),encoding='utf-8')
    print(name,'repartition audit passed',flush=True); return result


if __name__ == '__main__':
    saved = json.loads((FOLDER/'complete.json').read_text()); names = [r['dataset'] for r in saved['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    lookup = {r['dataset']:r for r in rows}; counts = sum((Counter(r['counts']) for r in rows),Counter())
    for k,v in saved['counts'].items(): assert counts[k] == v
    for row in saved['rows']: equivalent(row['mean_gains_pp'],lookup[row['dataset']]['mean_gains_pp'])
    for s,values in saved['dataset_equal_means_pp'].items():
        for m,v in values.items(): equivalent(v,float(np.mean([r['mean_gains_pp'][s][m] for r in rows])))
    result = {'passed':True,'counts':dict(counts),'max_manual_kernel_error':max(r['max_kernel_error'] for r in rows),
        'auditor_sha256':digest(ROOT/'scripts/audit_refit_repartition_diagnosis.py'),
        'scope':'all_frozen_supports_refitted_and_manual_kernel_replay_same_observation_repartition_only'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
