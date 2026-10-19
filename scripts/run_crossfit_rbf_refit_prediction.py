"""Classical common SVM/5NN classifiers; selector isolated from query labels."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_nested_geometry_forest import GRIDS,verified_native,holm
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from review_box_iterations import paired

INPUT = ROOT/'results/crossfit-rbf-refit-screen'; FOLDER = ROOT/'results/crossfit-rbf-refit-prediction'
IMPUTATION = ROOT/'results/crossfit-kernel-screen'
METHODS = ('new','old','full','fgs_matched','sfrip_matched','fgs_native','sfrip_native','imputed')
CLASSIFIERS = ('svm','knn')

def BA(pred,truth): return float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)]))

def svm_record(model):
    return {'support':model.support_.tolist(),'support_vectors':model.support_vectors_.tolist(),
        'dual_coef':model.dual_coef_.tolist(),'intercept':model.intercept_.tolist(),'classes':model.classes_.tolist(),
        'n_support':model.n_support_.tolist(),'n_iter':model.n_iter_.tolist(),'gamma':float(model._gamma),
        'fit_status':int(model.fit_status_),'C':1.,'class_weight':'balanced','kernel':'rbf','tol':.001,
        'max_iter':-1,'decision_function_shape':'ovr'}

def run(name):
    threadpool_limits(1); source = source_verified(name); native = verified_native(name,source)
    screened = json.loads((INPUT/(name+'.json')).read_text()); screenaudit = json.loads((INPUT/(name+'-audit.json')).read_text())
    assert screenaudit['fingerprints']['artifact_sha256'] == digest(INPUT/(name+'.json'))
    fullaudit = json.loads((INPUT/'complete-audit.json').read_text()); assert fullaudit['passed'] and fullaudit['resource_gate']
    imputed = json.loads((IMPUTATION/(name+'.json')).read_text())
    imputed_audit = json.loads((IMPUTATION/(name+'-audit.json')).read_text())
    assert imputed_audit['fingerprints']['artifact_sha256'] == digest(IMPUTATION/(name+'.json'))
    hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'research/crossfit-rbf-refit-prediction-protocol.md',INPUT/(name+'.json'),INPUT/(name+'-audit.json'),IMPUTATION/(name+'.json'),IMPUTATION/(name+'-audit.json')):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    out = FOLDER/(name+'.json'); records = []
    if out.exists():
        prior = json.loads(out.read_text()); assert prior['fingerprints'] == hashes
        if prior['complete']: return
        records = prior['records']
    X,y = load_dataset(name); expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    assert len(expected) == len(screened['rows'])
    for index,((i,j,s,inner),r) in enumerate(zip(expected,screened['rows'])):
        if index < len(records): continue
        assert r['outer'] == i and r['inner'] == j
        ri,a,b = np.array(s['reference_indices']),np.array(inner['reference_indices']),np.array(inner['validation_indices'])
        assert set(a).isdisjoint(b); A,B = preprocess_fit(X[ri[a]],X[ri[b]]); labels = y[ri[a]]; d = A.shape[1]
        path = r['path']; target = len(path['features']); assert target == len(path['initial_features']) and path['candidate']['certificate']['feasible']
        ir = imputed['rows'][index]; assert ir['outer'] == i and ir['inner'] == j and ir['reference_indices'] == a.tolist()
        assert ir['path']['initial_features'] == path['initial_features'] and ir['path']['candidate']['certificate']['feasible']
        supports = {'new':path['features'],'old':path['initial_features'],'full':list(range(d)),'imputed':ir['path']['features']}; selectors = {}
        for method,ci,prefix in [('fgs2023',4,'fgs'),('sfrip2025',3,'sfrip')]:
            ref = native['splits'][i]['inner'][j]['paths'][method][ci]
            supports[prefix+'_native'] = sorted(ref['native']['features']); supports[prefix+'_matched'] = sorted(ref['ranking'][:target])
            selectors[method] = {'selector_index':ci,'config':GRIDS[method][ci],'ranking':ref['ranking'],
                'native_features':supports[prefix+'_native'],'matched_features':supports[prefix+'_matched']}
        models = {}; fits = {}
        for method in METHODS:
            fs = supports[method]
            if not fs:
                # Empty native support: explicit constant class prediction,
                # balanced-class tie chooses the smallest training class.
                models[method] = None; fits[method] = {'features':[],'constant_class':np.unique(labels)[0].item(),
                    'scope':'empty_native_support_balanced_constant_tie_not_full_fallback'}
            else:
                svm = SVC(C=1.,kernel='rbf',gamma='scale',class_weight='balanced',decision_function_shape='ovr').fit(A[:,fs],labels)
                knn = KNeighborsClassifier(n_neighbors=min(5,len(A)),weights='uniform',metric='minkowski',p=2,n_jobs=1).fit(A[:,fs],labels)
                models[method] = {'svm':svm,'knn':knn}; fits[method] = {'features':fs,'svm':svm_record(svm),
                    'knn':{'neighbors':min(5,len(A)),'weights':'uniform','p':2,'metric':'minkowski','classes':knn.classes_.tolist()}}
        exact = all(len(supports[m]) == target for m in ('old','fgs_matched','sfrip_matched','imputed'))
        # All support choices and trained classifiers have finalized above.
        rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        metrics,evaluations = {},{}
        for method in METHODS:
            fs = supports[method]; metrics[method] = {}; evaluations[method] = {}
            for family in CLASSIFIERS:
                if models[method] is None:
                    pred = [np.full(len(Q),fits[method]['constant_class']) for Q in queries]; scores = [None]*4
                else:
                    model = models[method][family]; pred = [model.predict(Q[:,fs]) for Q in queries]
                    scores = [(model.decision_function(Q[:,fs]) if family == 'svm' else model.predict_proba(Q[:,fs])).tolist() for Q in queries]
                values = [BA(p,y[ri[b]]) for p in pred]
                metrics[method][family] = {'clean':values[0],'noise':float(np.mean(values[1:])), 'removed':1-len(fs)/d}
                evaluations[method][family] = {'predictions':[p.tolist() for p in pred],'scores':scores}
        records.append({'outer':i,'inner':j,'outer_reference_indices':ri.tolist(),'reference_indices':a.tolist(),'validation_indices':b.tolist(),
            'query_seed':144+i,'target_count':target,'counts_exact':exact,'supports':supports,'candidate':path['candidate'],
            'selector_paths':selectors,'fits':fits,'metrics':metrics,'evaluations':evaluations})
        out.write_text(json.dumps({'dataset':name,'d':d,'fingerprints':hashes,'records':records,'complete':len(records) == len(expected),
            'scope':'same_count_common_SVM_5NN_classification_exposed_development_no_SVM_margin_certificate'},separators=(',',':')),encoding='utf-8')
        print(name,i,j,'SVM',{m:round(metrics[m]['svm']['clean'],5) for m in METHODS},
            'KNN',{m:round(metrics[m]['knn']['clean'],5) for m in METHODS[:5]},flush=True)

def summarize(names):
    rows = []; counts = {'records':0,'count_mismatches':0,'svm_status_failures':0,'empty_native_supports':0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        for r in saved['records']:
            counts['records'] += 1; counts['count_mismatches'] += not r['counts_exact']
            for fit in r['fits'].values():
                counts['empty_native_supports'] += not fit['features']
                if fit['features']: counts['svm_status_failures'] += fit['svm']['fit_status'] != 0
        rows.append({'dataset':name,'methods':{m:{family:{metric:float(np.mean([r['metrics'][m][family][metric] for r in saved['records']]))
            for metric in ('clean','noise','removed')} for family in CLASSIFIERS} for m in METHODS}})
    assert counts['records'] == 128 and len(rows) == 15
    result = {'rows':rows,'counts':counts,
        'means':{family:{m:{metric:float(np.mean([r['methods'][m][family][metric] for r in rows])) for metric in ('clean','noise','removed')} for m in METHODS} for family in CLASSIFIERS},
        'contrasts':{family:{metric:{m:paired([r['methods']['new'][family][metric]-r['methods'][m][family][metric] for r in rows]) for m in METHODS[1:]}
            for metric in ('clean','noise')} for family in CLASSIFIERS},
        'scope':'all15_all128_primary_SVM_secondary_KNN_common_fixedsettings_actualcount_exposed_development'}
    clean = result['contrasts']['svm']['clean']; corrected = holm({m:clean[m]['sign_flip_two_sided_p'] for m in ('full','fgs_matched','sfrip_matched')}); result['primary_holm'] = corrected
    result['development_numeric_gate'] = bool(all(clean[m]['mean_pp'] >= .5 and corrected[m] < .05 for m in ('fgs_matched','sfrip_matched')) and clean['full']['mean_pp'] >= 0
        and result['means']['svm']['new']['removed'] >= .1 and counts['count_mismatches'] == counts['svm_status_failures'] == 0)
    result['independent_audit_required'] = True; result['goal_achieved'] = False
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print({k:v for k,v in result.items() if k != 'rows'},flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args(); FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((INPUT/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
