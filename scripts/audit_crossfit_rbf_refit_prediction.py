"""Independent fit/provenance, manual SVM kernels and unique-rank 5NN votes."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native,holm
from run_crossfit_rbf_refit_prediction import INPUT,FOLDER,METHODS,CLASSIFIERS,IMPUTATION
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from audit_forest_compact_transfer import equivalent
from review_box_iterations import paired

def BA(pred,truth):
    classes = sorted(set(truth.tolist()))
    return float(sum(float(np.sum((pred == c)&(truth == c)))/np.sum(truth == c) for c in classes)/len(classes))

def manual_svm_scores(X,fit):
    sv = np.array(fit['support_vectors']); kernel = np.exp(-fit['gamma']*np.sum((X[:,None,:]-sv[None,:,:])**2,axis=2))
    coef,intercept = np.array(fit['dual_coef']),np.array(fit['intercept']); q = len(fit['classes'])
    if q == 2: return kernel@coef[0]+intercept[0]
    boundaries = np.r_[0,np.cumsum(fit['n_support'])]; vote = np.zeros((len(X),q)); confidence = np.zeros((len(X),q)); pair = 0
    for i in range(q):
        for j in range(i+1,q):
            si,ei = boundaries[i:i+2]; sj,ej = boundaries[j:j+2]
            contrast = kernel[:,si:ei]@coef[j-1,si:ei]+kernel[:,sj:ej]@coef[i,sj:ej]+intercept[pair]
            vote[:,i] += contrast >= 0; vote[:,j] += contrast < 0
            confidence[:,i] += contrast; confidence[:,j] -= contrast; pair += 1
    return vote+confidence/(3*(abs(confidence)+1))

def audit(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source)
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text()); screen = json.loads((INPUT/(name+'.json')).read_text()); assert saved['complete']
    upstream = json.loads((INPUT/(name+'-audit.json')).read_text()); assert upstream['fingerprints']['artifact_sha256'] == digest(INPUT/(name+'.json'))
    imputed = json.loads((IMPUTATION/(name+'.json')).read_text())
    ia = json.loads((IMPUTATION/(name+'-audit.json')).read_text()); assert ia['fingerprints']['artifact_sha256'] == digest(IMPUTATION/(name+'.json'))
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha
    fingerprints = {'artifact_sha256':digest(artifact),'auditor_sha256':digest(Path(__file__))}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior['fingerprints'] == fingerprints: return prior['result']
    X,y = load_dataset(name); counts = Counter(); maxima = {'SVM_kernel_score_error':0.}
    expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(expected) == len(saved['records']) == len(screen['rows'])
    for r,sr,(i,j,s,inner) in zip(saved['records'],screen['rows'],expected):
        assert r['outer'] == sr['outer'] == i and r['inner'] == sr['inner'] == j and r['outer_reference_indices'] == s['reference_indices']
        for key in ('reference_indices','validation_indices'): assert r[key] == inner[key]
        ri,a,b = (np.array(r[k]) for k in ('outer_reference_indices','reference_indices','validation_indices'))
        assert set(a).isdisjoint(b); A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels,truth = y[ri[a]],y[ri[b]]; d = A.shape[1]
        supports = r['supports']; assert supports['new'] == sr['path']['features'] and supports['old'] == sr['path']['initial_features'] and supports['full'] == list(range(d))
        ir = next(row for row in imputed['rows'] if row['outer'] == i and row['inner'] == j)
        assert ir['reference_indices'] == r['reference_indices'] and supports['imputed'] == ir['path']['features']
        assert ir['path']['initial_features'] == supports['old'] and ir['path']['candidate']['certificate']['feasible']
        assert r['candidate'] == sr['path']['candidate']; fs = supports['new']; target = len(fs); assert r['target_count'] == target == len(supports['old'])
        original = np.ones(d)/np.sqrt(d); protected = MajorityTerms(A,labels,weights=original).bounds(range(d),allowed_enemies=2) > 1e-10
        weights = original if len(fs) == d else capped_sparse_projection(transfer_weights(A,fs,original)/np.sqrt(d),len(fs),2/d)*np.sqrt(d)
        embedded = np.zeros(d); embedded[fs] = weights
        equivalent(r['candidate'],{'features':fs,'weights':embedded.tolist(),'certificate':weight_certificate(A,labels,fs,original,weights,protected=protected)})
        assert r['candidate']['certificate']['feasible']; counts['FRS_replays'] += 1
        for method,ci,prefix in [('fgs2023',4,'fgs'),('sfrip2025',3,'sfrip')]:
            ref = native['splits'][i]['inner'][j]['paths'][method][ci]; path = r['selector_paths'][method]
            assert path['selector_index'] == ci and path['config'] == GRIDS[method][ci] and path['ranking'] == ref['ranking']
            assert path['native_features'] == sorted(ref['native']['features']) and path['matched_features'] == sorted(ref['ranking'][:target])
            for suffix in ('native','matched'): assert supports[prefix+'_'+suffix] == path[suffix+'_features']
        exact = all(len(supports[m]) == target for m in ('old','fgs_matched','sfrip_matched','imputed')); assert r['counts_exact'] == exact
        counts['count_mismatches'] += not exact; assert r['query_seed'] == 144+i
        rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        for method in METHODS:
            fs = supports[method]; fit = r['fits'][method]; assert fit['features'] == fs
            if not fs:
                counts['empty_native_supports'] += 1; assert fit['constant_class'] == min(labels)
                for family in CLASSIFIERS:
                    constant = np.full(len(B),fit['constant_class']); reported = r['evaluations'][method][family]
                    assert reported['predictions'] == [constant.tolist()]*4 and reported['scores'] == [None]*4
                    equivalent(r['metrics'][method][family],{'clean':BA(constant,truth),'noise':BA(constant,truth),'removed':1.})
                continue
            svm = SVC(C=1.,kernel='rbf',gamma='scale',class_weight='balanced',decision_function_shape='ovr').fit(A[:,fs],labels)
            knn = KNeighborsClassifier(n_neighbors=min(5,len(A)),weights='uniform',metric='minkowski',p=2,n_jobs=1).fit(A[:,fs],labels)
            sf = fit['svm']; assert sf['C'] == 1. and sf['class_weight'] == 'balanced' and sf['tol'] == .001 and sf['max_iter'] == -1
            assert sf['kernel'] == 'rbf' and sf['decision_function_shape'] == 'ovr' and sf['classes'] == svm.classes_.tolist()
            assert sf['support'] == svm.support_.tolist() and sf['n_support'] == svm.n_support_.tolist() and sf['n_iter'] == svm.n_iter_.tolist()
            equivalent(sf['gamma'],float(svm._gamma)); np.testing.assert_allclose(sf['support_vectors'],A[svm.support_][:,fs],atol=1e-12)
            np.testing.assert_allclose(sf['dual_coef'],svm.dual_coef_,atol=1e-8,rtol=1e-8); np.testing.assert_allclose(sf['intercept'],svm.intercept_,atol=1e-8,rtol=1e-8)
            assert sf['fit_status'] == svm.fit_status_; counts['svm_status_failures'] += sf['fit_status'] != 0
            assert fit['knn'] == {'neighbors':min(5,len(A)),'weights':'uniform','p':2,'metric':'minkowski','classes':knn.classes_.tolist()}
            counts['SVM_refits'] += 1; counts['KNN_refits'] += 1
            for family,model in [('svm',svm),('knn',knn)]:
                values = []; report = r['evaluations'][method][family]
                for k,Q in enumerate(queries):
                    prediction = model.predict(Q[:,fs]); assert report['predictions'][k] == prediction.tolist()
                    scores = model.decision_function(Q[:,fs]) if family == 'svm' else model.predict_proba(Q[:,fs])
                    np.testing.assert_allclose(report['scores'][k],scores,atol=1e-10,rtol=1e-10); values.append(BA(prediction,truth))
                    if family == 'svm':
                        manual = manual_svm_scores(Q[:,fs],sf); maximum = float(abs(manual-scores).max())
                        maxima['SVM_kernel_score_error'] = max(maxima['SVM_kernel_score_error'],maximum)
                        np.testing.assert_allclose(manual,scores,atol=1e-8,rtol=1e-8); counts['manual_SVM_kernel_replays'] += 1
                    else:
                        dist = np.sum((Q[:,None,fs]-A[None,:,fs])**2,axis=2); order = np.argsort(dist,axis=1,kind='stable'); neighbors = min(5,len(A))
                        if len(A) > neighbors:
                            sorted_d = np.take_along_axis(dist,order,axis=1); unique = sorted_d[:,neighbors]-sorted_d[:,neighbors-1] > 1e-12
                        else: unique = np.ones(len(Q),bool)
                        votes = np.stack([np.mean(labels[order[:,:neighbors]] == c,axis=1) for c in knn.classes_],axis=1)
                        np.testing.assert_allclose(scores[unique],votes[unique],atol=1e-12)
                        counts['manual_unique_rank_KNN_votes'] += int(unique.sum()); counts['KNN_boundary_tie_queries'] += int((~unique).sum())
                    counts['query_predictions'] += 1
                equivalent(r['metrics'][method][family],{'clean':values[0],'noise':float(np.mean(values[1:])), 'removed':1-len(fs)/d})
        counts['records'] += 1; print(name,i,j,'crossfit kernel prediction audit passed',flush=True)
    result = {'dataset':name,'counts':dict(counts),'precision_maxima':maxima,'methods':{m:{family:{metric:float(np.mean([r['metrics'][m][family][metric] for r in saved['records']]))
        for metric in ('clean','noise','removed')} for family in CLASSIFIERS} for m in METHODS}}
    checkpoint.write_text(json.dumps({'fingerprints':fingerprints,'result':result},indent=2),encoding='utf-8'); return result

if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text()); names = [r['dataset'] for r in summary['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(audit,n) for n in names])]
    lookup = {r['dataset']:r for r in rows}; rows = [lookup[n] for n in names]; counts = sum((Counter(r['counts']) for r in rows),Counter())
    equivalent(summary['rows'],[{'dataset':r['dataset'],'methods':r['methods']} for r in rows])
    for k,v in summary['counts'].items(): assert counts[k] == v
    for family in CLASSIFIERS:
        for m in METHODS:
            for metric in ('clean','noise','removed'): equivalent(summary['means'][family][m][metric],float(np.mean([r['methods'][m][family][metric] for r in rows])))
        for metric in ('clean','noise'):
            for m in METHODS[1:]: equivalent(summary['contrasts'][family][metric][m],paired([r['methods']['new'][family][metric]-r['methods'][m][family][metric] for r in rows]))
    clean = summary['contrasts']['svm']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')}); equivalent(summary['primary_holm'],corrected)
    gate = all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched')) and clean['full']['mean_pp'] >= 0 and summary['means']['svm']['new']['removed'] >= .1 and counts['count_mismatches'] == counts['svm_status_failures'] == 0
    assert summary['development_numeric_gate'] == gate and not summary['goal_achieved']
    result = {'passed':True,'counts':dict(counts),'precision_maxima':{k:max(r['precision_maxima'][k] for r in rows) for k in rows[0]['precision_maxima']},
        'auditor_sha256':digest(Path(__file__)),'scope':'independent_both_classifier_refits_fitted_SVM_manual_kernel_scores_unique_KNN_votes_FRS_allqueries',
        'limits':'exposed_development_fixed_recent_configs_no_SVM_box_or_population_guarantee; KNN_boundary_ties_follow_verified_sklearn'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
