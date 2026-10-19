"""Fold-isolated full-support bagged classifier mechanism, all15/all128."""
import json,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.orthogonal_forest_logistic import fit_orthogonal_forest,orthogonal_scores
from certified_frfs.bagged_fixed_forest import fit_bagged_fixed_forest

FOLDER = ROOT/'results/bagged-forest-mechanism-screen'


def evaluate(Q,truth,fit):
    scores = orthogonal_scores(Q,fit); classes = np.array(fit['classes']); pred = classes[scores.argmax(axis=1)]
    indices = np.searchsorted(classes,truth); losses = logsumexp(scores,axis=1)-scores[np.arange(len(Q)),indices]
    return {'scores':scores.tolist(),'predictions':pred.tolist(),
        'balanced_accuracy':float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)])),
        'balanced_logloss':float(np.mean([np.mean(losses[truth == c]) for c in np.unique(truth)]))}


def run(name):
    threadpool_limits(1); source = source_verified(name); X,y = load_dataset(name); hashes = dict(source['fingerprints'])
    for p in (Path(__file__),ROOT/'src/certified_frfs/bagged_fixed_forest.py',ROOT/'src/certified_frfs/nested_teacher_regularized.py',
        ROOT/'research/bagged-forest-mechanism-screen-protocol.md'): hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    artifact = FOLDER/(name+'.json'); rows = []
    if artifact.exists():
        prior = json.loads(artifact.read_text()); assert prior['fingerprints'] == hashes
        if prior['complete']: return
        rows = prior['rows']
    expected = [(i,j,s,inner) for i,s in enumerate(source['splits']) for j,inner in enumerate(s['inner'])]
    for index,(i,j,s,inner) in enumerate(expected):
        if index < len(rows): continue
        ri,a = np.array(s['reference_indices']),np.array(inner['reference_indices']); raw,labels = X[ri[a]],y[ri[a]]; start = time.perf_counter(); folds = []
        for ca,cb in inner_splits(labels,seed=20261027):
            A,B = preprocess_fit(raw[ca],raw[cb]); train,truth = labels[ca],labels[cb]; features = list(range(raw.shape[1]))
            tree = interaction_tree(A,train)
            single = fit_orthogonal_forest(A,train,features,tree['edges'],C=1.,pair_penalty=1.)
            bag = fit_bagged_fixed_forest(A,train,features,tree['edges'],C=1.,pair_penalty=1.,members=12,seed=20261027)
            evaluations = {'single':evaluate(B,truth,single),'bagged':evaluate(B,truth,bag['fit'])}
            folds.append({'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),'tree':tree,'single':single,'bag':bag,'evaluations':evaluations})
        metrics = {m:{k:float(np.mean([f['evaluations'][m][k] for f in folds])) for k in ('balanced_accuracy','balanced_logloss')} for m in ('single','bagged')}
        rows.append({'outer':i,'inner':j,'outer_reference_indices':ri.tolist(),'reference_indices':a.tolist(),
            'folds':folds,'metrics':metrics,'seconds':time.perf_counter()-start})
        artifact.write_text(json.dumps({'dataset':name,'fingerprints':hashes,'rows':rows,'complete':len(rows) == len(expected),
            'archived_query_labels_used':False,'scope':'fold_isolated_full_classifier_mechanism_not_reduct_superiority'},separators=(',',':')),encoding='utf-8')
        print(name,i,j,'BAgainpp',round(100*(metrics['bagged']['balanced_accuracy']-metrics['single']['balanced_accuracy']),6),
            'loglossgain',round(metrics['single']['balanced_logloss']-metrics['bagged']['balanced_logloss'],6),'seconds',round(rows[-1]['seconds'],2),flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True); names = [r['dataset'] for r in json.loads((ROOT/'results/crossfit-rbf-refit-screen/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]): f.result()
    counts = Counter(); rows = []
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete'] and not saved['archived_query_labels_used']
        for r in saved['rows']:
            counts['references'] += 1; counts['positive_BA_references'] += r['metrics']['bagged']['balanced_accuracy']-r['metrics']['single']['balanced_accuracy'] > 1e-12
            for fold in r['folds']:
                counts['internal_folds'] += 1; counts['single_fits'] += 1; counts['member_fits'] += len(fold['bag']['members'])
                counts['single_caps'] += not fold['single']['optimizer_converged']; counts['member_caps'] += sum(not f['optimizer_converged'] for f in fold['bag']['members'])
        rows.append({'dataset':name,'metrics':{m:{k:float(np.mean([r['metrics'][m][k] for r in saved['rows']])) for k in ('balanced_accuracy','balanced_logloss')} for m in ('single','bagged')}})
    assert counts['references'] == 128 and len(rows) == 15
    means = {m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in ('balanced_accuracy','balanced_logloss')} for m in ('single','bagged')}
    gain = 100*(means['bagged']['balanced_accuracy']-means['single']['balanced_accuracy']); loggain = means['single']['balanced_logloss']-means['bagged']['balanced_logloss']
    gate = gain >= .5 and loggain >= 0 and counts['positive_BA_references'] >= 20 and counts['single_caps'] == counts['member_caps'] == 0
    result = {'rows':rows,'means':means,'counts':dict(counts),'BA_gain_pp':gain,'logloss_gain':loggain,'numeric_resource_gate':bool(gate),
        'goal_achieved':False,'archived_query_labels_used':False,'independent_audit_required':True,
        'scope':'isolated_internal_full_classifier_only_not_recent_baseline_or_support_quality'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
