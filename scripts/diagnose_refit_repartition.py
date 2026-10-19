"""Reference-only stability of fixed22/24 supports under one frozen resplit."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from run_crossfit_kernel_prediction import svm_record
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits

FOLDER = ROOT/'results/refit-repartition-diagnosis'
SOURCES = {'refit':ROOT/'results/crossfit-rbf-refit-screen','local':ROOT/'results/exhaustive-refit-screen'}


def run(name):
    threadpool_limits(1); original = source_verified(name); X,y = load_dataset(name); inputs = {}; hashes = dict(original['fingerprints'])
    for method,path in SOURCES.items():
        assert json.loads((path/'complete-audit.json').read_text())['passed']
        inputs[method] = json.loads((path/(name+'.json')).read_text())
        audit = json.loads((path/(name+'-audit.json')).read_text())
        assert audit['fingerprints']['artifact_sha256'] == digest(path/(name+'.json'))
        for rel,sha in inputs[method]['fingerprints'].items(): assert digest(ROOT/rel) == sha
        for p in (path/(name+'.json'),path/(name+'-audit.json')): hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    for p in (Path(__file__),ROOT/'research/refit-repartition-diagnosis-protocol.md',ROOT/'scripts/run_crossfit_kernel_prediction.py'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    artifact = FOLDER/(name+'.json')
    if artifact.exists():
        saved = json.loads(artifact.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']: return
    rows = []; expected = [(i,j,s,f) for i,s in enumerate(original['splits']) for j,f in enumerate(s['inner'])]
    for index,(i,j,s,inner) in enumerate(expected):
        ri,a = np.array(s['reference_indices']),np.array(inner['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]
        supports = {'old':inputs['refit']['rows'][index]['path']['initial_features']}
        for method,source in inputs.items():
            r = source['rows'][index]; assert r['outer'] == i and r['inner'] == j and r['reference_indices'] == a.tolist()
            supports[method] = r['path']['features']; assert len(supports[method]) == len(supports['old'])
        folds = []
        for ca,cb in inner_splits(labels,seed=20261025):
            A,B = preprocess_fit(raw[ca],raw[cb]); models = {}; fits = {}
            for method,fs in supports.items():
                svm = SVC(C=1.,kernel='rbf',gamma='scale',class_weight='balanced',decision_function_shape='ovr').fit(A[:,fs],labels[ca])
                assert svm.fit_status_ == 0; models[method] = svm; fits[method] = svm_record(svm)
            # All models fixed before reference-internal validation evaluation.
            evaluations = {}
            for method,model in models.items():
                pred = model.predict(B[:,supports[method]]); score = model.decision_function(B[:,supports[method]])
                error = float(np.mean([np.mean(pred[labels[cb] == c] != c) for c in np.unique(labels[cb])]))
                evaluations[method] = {'predictions':pred.tolist(),'scores':score.tolist(),'balanced_error':error}
            folds.append({'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),'fits':fits,'evaluations':evaluations})
        errors = {m:float(np.mean([f['evaluations'][m]['balanced_error'] for f in folds])) for m in supports}
        rows.append({'outer':i,'inner':j,'outer_reference_indices':ri.tolist(),'reference_indices':a.tolist(),'supports':supports,
            'folds':folds,'mean_errors':errors,'seed':20261025,'repartition_gains_pp':{m:100*(errors['old']-errors[m]) for m in SOURCES},
            'selection_gains_pp':{m:100*(inputs[m]['rows'][index]['path']['initial_risk']['balanced_error']-inputs[m]['rows'][index]['path']['final_risk']['balanced_error']) for m in SOURCES}})
        print(name,i,j,'repartition gains',rows[-1]['repartition_gains_pp'],flush=True)
    artifact.write_text(json.dumps({'dataset':name,'rows':rows,'fingerprints':hashes,'complete':True,
        'archived_query_labels_used':False,'scope':'same_observation_repartition_sensitivity_not_confirmation'},separators=(',',':')),encoding='utf-8')


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = [r['dataset'] for r in json.loads((SOURCES['refit']/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]): f.result()
    rows = []; counts = Counter()
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        counts['references'] += len(saved['rows']); counts['models'] += sum(3*len(r['folds']) for r in saved['rows'])
        row = {'dataset':name,'mean_gains_pp':{stage:{m:float(np.mean([r[stage+'_gains_pp'][m] for r in saved['rows']])) for m in SOURCES} for stage in ('selection','repartition')}}
        rows.append(row)
        for r in saved['rows']:
            for m in SOURCES:
                selected = r['selection_gains_pp'][m] > 1e-10
                counts[m+'_selected_improvements'] += selected
                counts[m+'_repartition_positive_of_selected'] += selected and r['repartition_gains_pp'][m] > 1e-10
                counts[m+'_repartition_negative_of_selected'] += selected and r['repartition_gains_pp'][m] < -1e-10
                counts[m+'_repartition_tied_of_selected'] += selected and abs(r['repartition_gains_pp'][m]) <= 1e-10
    assert counts['references'] == 128 and len(rows) == 15
    result = {'rows':rows,'counts':dict(counts),'dataset_equal_means_pp':{stage:{m:float(np.mean([r['mean_gains_pp'][stage][m] for r in rows])) for m in SOURCES} for stage in ('selection','repartition')},
        'goal_achieved':False,'archived_query_labels_used':False,'independent_audit_required':True,
        'scope':'fixed_support_same_data_repartition_sensitivity_no_p_values_or_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
